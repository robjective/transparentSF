import json
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import psycopg2.extras

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from ai.anomalyAnalyzer import get_anomaly_details

def generate_report_text(report_ids, execute_with_connection, load_prompts, AGENT_MODEL, client, logger=None):
    """
    Step 3.5: Generate the final report_text for each prioritized item using the LLM.
    Updates the monthly_reporting table with the final narrative for each item.
    Args:
        report_ids: List of report IDs to process
        execute_with_connection: Function to execute DB operations
        load_prompts: Function to load prompt templates
        AGENT_MODEL: Model name for the LLM
        client: LLM client
        logger: Optional logger
    Returns:
        Status dictionary
    """
    # Use the root logger if no logger is provided
    if logger is None:
        logger = logging.getLogger(__name__)
        # Ensure we're using the root logger's configuration
        logger.propagate = True
    
    logger.info(f"Generating report_text for {len(report_ids)} items")

    prompts = load_prompts()
    prompt_template = prompts['monthly_report']['generate_report_text']['prompt']
    system_message = prompts['monthly_report']['generate_report_text']['system']

    def generate_report_text_operation(connection):
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        updated_count = 0
        # Prepare log file
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'report_text_prompt.log')
        chart_log_file = os.path.join(logs_dir, 'chart_metadata.log')
        
        for report_id in report_ids:
            cursor.execute("SELECT * FROM monthly_reporting WHERE id = %s", (report_id,))
            item = cursor.fetchone()
            if not item:
                logger.warning(f"Report item with ID {report_id} not found for report_text generation")
                continue
            # Gather all fields
            rationale = item.get("rationale", "") if isinstance(item, dict) else item["rationale"]
            explanation = item.get("explanation", "") if isinstance(item, dict) else item["explanation"]
            trend_analysis = ""
            follow_up = ""
            charts = ""
            citations = ""
            perplexity_context = ""
            metadata = item.get("metadata") if isinstance(item, dict) else item["metadata"]
            
            # Log chart metadata
            if metadata:
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                trend_analysis = metadata.get("trend_analysis", "")
                follow_up = metadata.get("follow_up", "")
                if "charts" in metadata:
                    charts = metadata["charts"]
                    # Process each chart to get anomaly details
                    processed_charts = []
                    for chart in charts:
                        if isinstance(chart, str) and chart.startswith("[CHART:anomaly:"):
                            try:
                                # Extract anomaly ID from chart reference
                                anomaly_id = int(chart.split(":")[-1].rstrip("]"))
                                # Get anomaly details
                                context_variables = {}
                                anomaly_details = get_anomaly_details(context_variables, anomaly_id)
                                
                                if anomaly_details.get("status") == "success" and anomaly_details.get("anomaly"):
                                    anomaly = anomaly_details["anomaly"]
                                    # Get field_name and caption from anomaly details
                                    field_name = anomaly.get("field_name", "")
                                    caption = anomaly.get("metadata", {}).get("caption", "")
                                    
                                    # Create enhanced chart reference with metadata
                                    chart_with_metadata = {
                                        "reference": chart,
                                        "field_name": field_name,
                                        "caption": caption
                                    }
                                    processed_charts.append(chart_with_metadata)
                                else:
                                    # If we couldn't get anomaly details, keep the original chart reference
                                    processed_charts.append(chart)
                            except Exception as e:
                                logger.error(f"Error processing anomaly chart {chart}: {e}")
                                processed_charts.append(chart)
                        else:
                            processed_charts.append(chart)
                    
                    # Update charts with processed data
                    charts = processed_charts
                    
                    # Log chart metadata
                    try:
                        with open(chart_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                            f.write(f"Report ID: {report_id}\n")
                            f.write(f"Metric: {item.get('metric_name', '') if isinstance(item, dict) else item['metric_name']}\n")
                            f.write(f"Processed Chart Data:\n{json.dumps(charts, indent=2)}\n")
                        logger.info(f"Logged chart metadata for report ID {report_id} to {chart_log_file}")
                    except Exception as log_err:
                        logger.error(f"Failed to write chart metadata to log: {log_err}")
                
                if "perplexity_context" in metadata:
                    perplexity_context = metadata["perplexity_context"]
                if "perplexity_response" in metadata and "citations" in metadata["perplexity_response"]:
                    citations = metadata["perplexity_response"]["citations"]
            # Format as string for prompt
            charts_str = json.dumps(charts) if charts else ""
            # Format citations as a numbered list for the prompt
            citations_list = []
            if citations:
                if isinstance(citations, str):
                    try:
                        citations = json.loads(citations)
                    except Exception:
                        citations = [citations]
                if isinstance(citations, list):
                    for idx, citation in enumerate(citations, 1):
                        if isinstance(citation, str):
                            citations_list.append(f"{idx}. {citation}")
                        elif isinstance(citation, dict):
                            title = citation.get("title", "Untitled")
                            url = citation.get("url", citation.get("link", "No URL"))
                            citations_list.append(f"{idx}. {title}: {url}")
                        else:
                            citations_list.append(f"{idx}. {str(citation)}")
            citations_str = "\n".join(citations_list) if citations_list else ""
            prompt = prompt_template.format(
                rationale=rationale,
                explanation=explanation,
                trend_analysis=trend_analysis,
                follow_up=follow_up,
                charts=charts_str,
                citations=citations_str,
                perplexity_context=perplexity_context
            )
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[{"role": "system", "content": system_message},
                         {"role": "user", "content": prompt}],
                temperature=0.2
            )
            report_text = response.choices[0].message.content
            # Log prompt and response to file
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Report ID: {report_id}\n")
                    f.write(f"Metric: {item.get('metric_name', '') if isinstance(item, dict) else item['metric_name']}\n")
                    f.write(f"Prompt:\n{prompt}\n")
                    f.write(f"Response:\n{report_text}\n")
            except Exception as log_err:
                logger.error(f"Failed to write report_text prompt/response to log: {log_err}")
            
            # Update metadata with processed charts
            if isinstance(metadata, dict):
                metadata["charts"] = charts
                metadata_json = json.dumps(metadata)
            else:
                metadata_json = metadata
                
            cursor.execute(
                "UPDATE monthly_reporting SET report_text = %s, metadata = %s WHERE id = %s",
                (report_text, metadata_json, report_id)
            )
            updated_count += 1
        connection.commit()
        cursor.close()
        return updated_count

    result = execute_with_connection(generate_report_text_operation)
    if isinstance(result, dict) and result.get("status") == "success":
        logger.info(f"Successfully generated report_text for {result['result']} items")
        return {"status": "success", "message": f"Report text generated for {result['result']} items"}
    elif isinstance(result, int):
        logger.info(f"Successfully generated report_text for {result} items")
        return {"status": "success", "message": f"Report text generated for {result} items"}
    else:
        return {"status": "error", "message": str(result)} 