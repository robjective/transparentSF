import os
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import logging
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from tools.db_utils import get_postgres_connection
import psycopg2
import psycopg2.extras

# Initialize APIRouter
router = APIRouter()

# Templates will be set by the main app
templates = None

def set_templates(t):
    """Set the templates instance for this router"""
    global templates
    templates = t
    logging.info("Templates set in database admin router")

# Get logger
logger = logging.getLogger(__name__)

@router.get("/database-admin")
async def database_admin_page(request: Request):
    """Serve the database admin interface."""
    logger.debug("Database admin page route called")
    if templates is None:
        logger.error("Templates not initialized in database admin router")
        raise RuntimeError("Templates not initialized")
    
    logger.debug("Serving database_admin.html template")
    return templates.TemplateResponse("database_admin.html", {
        "request": request
    })

@router.get("/api/database-info")
async def get_database_info():
    """Get database information including status, size, and last backup time."""
    logger.debug("Get database info called")
    
    try:
        # Get database connection and basic info
        connection = get_postgres_connection()
        if not connection:
            raise Exception("Could not connect to database")
        
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get database size
        cursor.execute("""
            SELECT pg_size_pretty(pg_database_size(current_database())) as size;
        """)
        size_result = cursor.fetchone()
        db_size = size_result['size'] if size_result else 'Unknown'
        
        # Get table count
        cursor.execute("""
            SELECT count(*) as table_count 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        table_result = cursor.fetchone()
        table_count = table_result['table_count'] if table_result else 0
        
        cursor.close()
        connection.close()
        
        # Check for last backup file
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backups_dir = os.path.join(script_dir, 'backups')
        last_backup = "Never"
        
        if os.path.exists(backups_dir):
            backup_files = [f for f in os.listdir(backups_dir) if f.endswith('.sql') or f.endswith('.gz')]
            if backup_files:
                # Get the most recent backup file
                backup_files.sort(key=lambda x: os.path.getmtime(os.path.join(backups_dir, x)), reverse=True)
                latest_backup = backup_files[0]
                backup_time = os.path.getmtime(os.path.join(backups_dir, latest_backup))
                last_backup = datetime.fromtimestamp(backup_time).strftime("%Y-%m-%d %H:%M:%S")
        
        return JSONResponse({
            "status": "success",
            "db_status": f"Connected ({table_count} tables)",
            "db_size": db_size,
            "last_backup": last_backup
        })
        
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        return JSONResponse({
            "status": "error",
            "db_status": "Connection Error",
            "db_size": "Unknown",
            "last_backup": "Unknown",
            "message": str(e)
        })

@router.post("/api/database-backup")
async def create_database_backup():
    """Create a database backup and return download URL."""
    logger.debug("Create database backup called")
    
    try:
        # Get database connection info
        connection = get_postgres_connection()
        if not connection:
            raise Exception("Could not connect to database")
        
        # Get connection parameters
        conn_params = connection.get_dsn_parameters()
        connection.close()
        
        # Create backups directory if it doesn't exist
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backups_dir = os.path.join(script_dir, 'backups')
        os.makedirs(backups_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"database_backup_{timestamp}.sql"
        backup_path = os.path.join(backups_dir, backup_filename)
        
        # Check if DATABASE_URL is available (common for managed services like Replit PostgreSQL)
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            # Use DATABASE_URL directly for pg_dump
            logger.info("Using DATABASE_URL for pg_dump")
            pg_dump_cmd = [
                'pg_dump',
                '--verbose',
                '--no-owner',
                '--no-privileges',
                '--clean',
                '--if-exists',
                '--format=plain',
                '--file', backup_path,
                database_url
            ]
            
            # Set environment variables
            env = os.environ.copy()
            
        else:
            # Use individual connection parameters (fallback for local setups)
            logger.info("Using individual connection parameters for pg_dump")
            pg_dump_cmd = [
                'pg_dump',
                '--verbose',
                '--no-owner',
                '--no-privileges',
                '--clean',
                '--if-exists',
                '--format=plain',
                '--file', backup_path
            ]
            
            # Add connection parameters
            if 'host' in conn_params:
                pg_dump_cmd.extend(['--host', conn_params['host']])
            if 'port' in conn_params:
                pg_dump_cmd.extend(['--port', conn_params['port']])
            if 'user' in conn_params:
                pg_dump_cmd.extend(['--username', conn_params['user']])
            if 'dbname' in conn_params:
                pg_dump_cmd.append(conn_params['dbname'])
            
            # Set environment variables for password
            env = os.environ.copy()
            if 'password' in conn_params:
                env['PGPASSWORD'] = conn_params['password']
        
        logger.info(f"Running pg_dump command: {' '.join(pg_dump_cmd[:-2])} [database/file]")
        
        # Run pg_dump
        result = subprocess.run(
            pg_dump_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"pg_dump failed: {result.stderr}")
            raise Exception(f"Backup failed: {result.stderr}")
        
        # Verify backup file was created
        if not os.path.exists(backup_path):
            raise Exception("Backup file was not created")
        
        # Get file size for verification
        file_size = os.path.getsize(backup_path)
        if file_size == 0:
            raise Exception("Backup file is empty")
        
        logger.info(f"Database backup created successfully: {backup_path} ({file_size} bytes)")
        
        # Return success with download URL
        return JSONResponse({
            "status": "success",
            "message": f"Database backup created successfully ({file_size:,} bytes)",
            "filename": backup_filename,
            "download_url": f"/backend/api/download-backup/{backup_filename}",
            "file_size": file_size
        })
        
    except subprocess.TimeoutExpired:
        logger.error("Database backup timed out")
        return JSONResponse({
            "status": "error",
            "message": "Database backup timed out (5 minute limit exceeded)"
        }, status_code=500)
    except Exception as e:
        logger.error(f"Error creating database backup: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/download-backup/{filename}")
async def download_backup(filename: str):
    """Download a backup file."""
    logger.debug(f"Download backup called for filename: {filename}")
    
    try:
        # Security check - ensure filename is safe
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backups_dir = os.path.join(script_dir, 'backups')
        backup_path = os.path.join(backups_dir, filename)
        
        if not os.path.exists(backup_path):
            raise HTTPException(status_code=404, detail="Backup file not found")
        
        return FileResponse(
            backup_path,
            media_type='application/octet-stream',
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading backup {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/database-restore")
async def restore_database(backup_file: UploadFile = File(...)):
    """Restore database from uploaded backup file."""
    logger.debug(f"Database restore called with file: {backup_file.filename}")
    
    try:
        # Validate file
        if not backup_file.filename:
            raise Exception("No file provided")
        
        # Check file extension
        allowed_extensions = ['.sql', '.gz', '.tar', '.zip']
        file_ext = os.path.splitext(backup_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise Exception(f"Unsupported file type: {file_ext}. Allowed types: {', '.join(allowed_extensions)}")
        
        # Create temporary file for the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Read and save uploaded file
            contents = await backup_file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        logger.info(f"Uploaded file saved to temporary location: {temp_file_path}")
        
        try:
            # Get database connection info
            connection = get_postgres_connection()
            if not connection:
                raise Exception("Could not connect to database")
            
            conn_params = connection.get_dsn_parameters()
            connection.close()
            
            # Handle different file types
            if file_ext == '.sql':
                # Direct SQL file
                sql_file_path = temp_file_path
            elif file_ext == '.gz':
                # Gzipped SQL file - decompress it
                import gzip
                sql_file_path = temp_file_path + '_decompressed.sql'
                with gzip.open(temp_file_path, 'rb') as f_in:
                    with open(sql_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                raise Exception(f"File type {file_ext} not yet supported for restore")
            
            # Verify SQL file exists and has content
            if not os.path.exists(sql_file_path):
                raise Exception("Could not process backup file")
            
            file_size = os.path.getsize(sql_file_path)
            if file_size == 0:
                raise Exception("Backup file appears to be empty")
            
            logger.info(f"Processing SQL file: {sql_file_path} ({file_size} bytes)")
            
            # Check if DATABASE_URL is available (common for managed services like Replit PostgreSQL)
            database_url = os.getenv("DATABASE_URL")
            
            if database_url:
                # Use DATABASE_URL directly for psql
                logger.info("Using DATABASE_URL for psql restore")
                psql_cmd = [
                    'psql',
                    '--quiet',
                    '--file', sql_file_path,
                    database_url
                ]
                
                # Set environment variables
                env = os.environ.copy()
                
            else:
                # Use individual connection parameters (fallback for local setups)
                logger.info("Using individual connection parameters for psql restore")
                psql_cmd = [
                    'psql',
                    '--quiet',
                    '--file', sql_file_path
                ]
                
                # Add connection parameters
                if 'host' in conn_params:
                    psql_cmd.extend(['--host', conn_params['host']])
                if 'port' in conn_params:
                    psql_cmd.extend(['--port', conn_params['port']])
                if 'user' in conn_params:
                    psql_cmd.extend(['--username', conn_params['user']])
                if 'dbname' in conn_params:
                    psql_cmd.extend(['--dbname', conn_params['dbname']])
                
                # Set environment variables for password
                env = os.environ.copy()
                if 'password' in conn_params:
                    env['PGPASSWORD'] = conn_params['password']
            
            logger.info(f"Running psql restore command")
            
            # Run psql restore
            result = subprocess.run(
                psql_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Database restore failed: {result.stderr}")
                raise Exception(f"Database restore failed: {result.stderr}")
            
            logger.info("Database restore completed successfully")
            
            return JSONResponse({
                "status": "success",
                "message": f"Database restored successfully from {backup_file.filename}",
                "restored_size": file_size
            })
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if file_ext == '.gz' and 'sql_file_path' in locals() and sql_file_path != temp_file_path:
                    if os.path.exists(sql_file_path):
                        os.unlink(sql_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temporary files: {cleanup_error}")
        
    except subprocess.TimeoutExpired:
        logger.error("Database restore timed out")
        return JSONResponse({
            "status": "error",
            "message": "Database restore timed out (10 minute limit exceeded)"
        }, status_code=500)
    except Exception as e:
        logger.error(f"Error restoring database: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/session-logs-info")
async def get_session_logs_info():
    """Get information about session logs including count and latest session."""
    logger.debug("Get session logs info called")
    
    try:
        # Get the path to the session logs directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(script_dir, 'logs', 'sessions')
        
        total_sessions = 0
        latest_session = None
        logs_directory = logs_dir
        
        if os.path.exists(logs_dir):
            # Count session log files (all JSON files in sessions directory)
            session_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
            total_sessions = len(session_files)
            
            # Find the latest session
            if session_files:
                session_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
                latest_file = session_files[0]
                latest_time = os.path.getmtime(os.path.join(logs_dir, latest_file))
                latest_session = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M:%S")
        else:
            logs_directory = "Directory not found"
        
        return JSONResponse({
            "status": "success",
            "total_sessions": total_sessions,
            "latest_session": latest_session,
            "logs_directory": logs_directory
        })
        
    except Exception as e:
        logger.error(f"Error getting session logs info: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/api/download-session-logs")
async def download_session_logs():
    """Download all session logs as a ZIP file."""
    logger.debug("Download session logs called")
    
    try:
        # Get the path to the session logs directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(script_dir, 'logs', 'sessions')
        
        if not os.path.exists(logs_dir):
            return JSONResponse({
                "status": "error",
                "message": "Session logs directory not found"
            }, status_code=404)
        
        # Find all session log files (all JSON files in sessions directory)
        session_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
        
        if not session_files:
            return JSONResponse({
                "status": "error",
                "message": "No session log files found"
            }, status_code=404)
        
        # Create a temporary ZIP file
        temp_zip_path = tempfile.mktemp(suffix='.zip')
        
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for session_file in session_files:
                file_path = os.path.join(logs_dir, session_file)
                # Add file to ZIP with just the filename (no path)
                zipf.write(file_path, session_file)
        
        # Return the ZIP file
        return FileResponse(
            path=temp_zip_path,
            filename=f"session_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=session_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            }
        )
        
    except Exception as e:
        logger.error(f"Error downloading session logs: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)