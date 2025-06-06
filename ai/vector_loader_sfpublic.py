import os
import sys
import json
import re
import time
import uuid
import logging
from openai import OpenAI
import qdrant_client
from qdrant_client.http import models as rest
from dotenv import load_dotenv
import tiktoken  # For counting tokens
from tools.data_processing import format_columns, serialize_columns, convert_to_timestamp
from tools.db_utils import get_postgres_connection
import psycopg2.extras
import shutil

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "vector_loader.log"), mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API key not found in environment variables.")
    raise ValueError("OpenAI API key not found in environment variables.")

client = OpenAI(api_key=openai_api_key)

# ------------------------------
# Model Configuration
# ------------------------------
EMBEDDING_MODEL = "text-embedding-3-large"

# ------------------------------
# Qdrant Setup
# ------------------------------
try:
    qdrant = qdrant_client.QdrantClient(host='localhost', port=6333)
    logger.info("Connected to Qdrant at localhost:6333")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    raise

def split_text_into_chunks(text, max_tokens=8192):
    """
    Splits text into chunks that fit within the token limit.
    Uses tiktoken to count tokens accurately.
    """
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    logger.debug(f"Total tokens in text: {total_tokens}")

    chunks = []
    for i in range(0, total_tokens, max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        logger.debug(f"Created chunk {len(chunks)} with tokens {i} to {i + len(chunk_tokens)}")

    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks

def get_embedding(text, retries=3, delay=5):
    """
    Generate embeddings for text, handling chunking for large inputs.
    Returns the average embedding of all chunks.
    """
    embeddings = []
    chunks = list(split_text_into_chunks(text, max_tokens=8192))
    logger.info(f"Generating embeddings for {len(chunks)} chunks.")
    
    for idx, chunk in enumerate(chunks, start=1):
        logger.debug(f"Processing chunk {idx}/{len(chunks)}")
        for attempt in range(1, retries + 1):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL, 
                    input=chunk
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                logger.debug(f"Successfully generated embedding for chunk {idx}")
                break  # If successful, exit retry loop
            except Exception as e:
                logger.warning(f"Attempt {attempt} - Error generating embedding for chunk {idx}: {e}")
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries reached for chunk {idx}. Skipping chunk.")
        else:
            logger.error(f"Failed to generate embedding for chunk {idx} after {retries} attempts.")
    
    if not embeddings:
        logger.error("No embeddings were generated.")
        return None
    
    # Average embeddings across chunks
    averaged_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
    logger.info("Averaged embeddings across all chunks.")
    return averaged_embedding

def recreate_collection(collection_name, vector_size):
    """Delete if exists and recreate the collection."""
    try:
        # Check if collection exists
        if qdrant.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' exists, deleting...")
            try:
                qdrant.delete_collection(collection_name)
                # Wait longer for deletion to complete
                time.sleep(5)  # Increased from 2 to 5 seconds
                
                # Verify collection is actually deleted
                max_retries = 3
                for attempt in range(max_retries):
                    if not qdrant.collection_exists(collection_name):
                        logger.info(f"Collection '{collection_name}' successfully deleted.")
                        break
                    if attempt < max_retries - 1:
                        logger.warning(f"Collection still exists, waiting longer... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(5)  # Wait another 5 seconds
                    else:
                        raise Exception(f"Collection '{collection_name}' still exists after deletion attempts")
            except Exception as e:
                logger.error(f"Error during collection deletion: {e}")
                raise
        
        # Create new collection
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
        try:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=vector_size,
                ),
                timeout=120  # Increased timeout to 120 seconds
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Failed to recreate collection '{collection_name}': {e}")
        raise

def load_datasets_from_db():
    """Load all active datasets from the PostgreSQL database."""
    try:
        connection = get_postgres_connection()
        if not connection:
            logger.error("Failed to connect to database")
            return []
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query all active datasets from the database
        cursor.execute("""
            SELECT 
                id,
                endpoint,
                url,
                title,
                category,
                description,
                publishing_department,
                rows_updated_at,
                columns,
                metadata,
                created_at,
                updated_at
            FROM datasets 
            WHERE is_active = true
            ORDER BY title
        """)
        
        datasets = cursor.fetchall()
        cursor.close()
        connection.close()
        
        logger.info(f"Loaded {len(datasets)} active datasets from database")
        return datasets
        
    except Exception as e:
        logger.error(f"Error loading datasets from database: {e}")
        return []

def load_sf_public_data():
    """Load SF Public Data into Qdrant from PostgreSQL database."""
    collection_name = 'SFPublicData'
    
    # Get sample embedding to determine vector size
    sample_embedding = get_embedding("Sample text to determine vector size.")
    if not sample_embedding:
        logger.error("Failed to get a sample embedding for vector size determination.")
        return False
    vector_size = len(sample_embedding)
    
    # Create or recreate collection
    recreate_collection(collection_name, vector_size)

    # Load datasets from database
    datasets = load_datasets_from_db()
    if not datasets:
        logger.warning("No datasets found in database")
        return False

    logger.info(f"Processing {len(datasets)} datasets from database for SFPublicData")
    points_to_upsert = []

    for idx, dataset in enumerate(datasets, start=1):
        logger.info(f"Processing dataset {idx}/{len(datasets)}: {dataset['title']}")

        try:
            # Extract data from database record
            title = dataset.get('title', '')
            description = dataset.get('description', '')
            columns = dataset.get('columns', []) or []  # Handle None
            url = dataset.get('url', '')
            endpoint = dataset.get('endpoint', '')
            category = dataset.get('category', '')
            publishing_department = dataset.get('publishing_department', '')
            last_updated_date = dataset.get('rows_updated_at', '')
            metadata = dataset.get('metadata', {}) or {}  # Handle None

            # Extract additional fields from metadata if available
            page_text = metadata.get('page_text', '')
            queries = metadata.get('queries', [])
            report_category = metadata.get('report_category', '')
            periodic = metadata.get('periodic', '')
            district_level = metadata.get('district_level', '')

            # Format text for embedding
            combined_text_parts = []
            if title:
                combined_text_parts.append(f"Title: {title}")
            if url:
                combined_text_parts.append(f"URL: {url}")
            if endpoint:
                combined_text_parts.append(f"Endpoint: {endpoint}")
            if description:
                combined_text_parts.append(f"Description: {description}")
            if page_text:
                combined_text_parts.append(f"Content: {page_text}")
            if columns:
                columns_formatted = format_columns(columns)
                combined_text_parts.append(f"Columns: {columns_formatted}")
            if report_category:
                combined_text_parts.append(f"Report Category: {report_category}")
            if publishing_department:
                combined_text_parts.append(f"Publishing Department: {publishing_department}")
            if last_updated_date:
                combined_text_parts.append(f"Last Updated: {last_updated_date}")
            if periodic:
                combined_text_parts.append(f"Periodic: {periodic}")
            if district_level:
                combined_text_parts.append(f"District Level: {district_level}")
            if queries:
                combined_text_parts.append(f"Queries: {queries}")

            combined_text = "\n".join(combined_text_parts)

            # Generate embedding
            embedding = get_embedding(combined_text)
            if not embedding:
                logger.error(f"Failed to generate embedding for dataset '{title}'. Skipping.")
                continue

            # Prepare payload
            payload = {
                'title': title,
                'description': description,
                'url': url,
                'endpoint': endpoint,
                'columns': serialize_columns(columns),
                'column_names': [col.get('name', '').lower() for col in columns if isinstance(col, dict)],
                'category': category.lower() if category else '',
                'publishing_department': str(publishing_department).lower() if publishing_department else '',
                'last_updated_date': convert_to_timestamp(last_updated_date) if last_updated_date else None,
                'queries': queries,
                'database_id': dataset.get('id'),  # Store the database ID for reference
            }

            # Remove None values from payload
            payload = {k: v for k, v in payload.items() if v is not None}

            point = rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points_to_upsert.append(point)

            # Batch upload every 100 points
            if len(points_to_upsert) >= 100:
                try:
                    qdrant.upsert(collection_name=collection_name, points=points_to_upsert)
                    logger.info(f"Successfully upserted batch of {len(points_to_upsert)} documents")
                    points_to_upsert = []
                except Exception as e:
                    logger.error(f"Error upserting batch to Qdrant: {e}")

        except Exception as e:
            logger.error(f"Error processing dataset '{dataset.get('title', 'Unknown')}': {e}")
            continue

    # Upload any remaining points
    if points_to_upsert:
        try:
            qdrant.upsert(collection_name=collection_name, points=points_to_upsert)
            logger.info(f"Successfully upserted final batch of {len(points_to_upsert)} documents")
        except Exception as e:
            logger.error(f"Error upserting final batch to Qdrant: {e}")

    return True

if __name__ == '__main__':
    load_sf_public_data() 