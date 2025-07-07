from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
import os
from typing import Optional, List
import traceback

# Import the weekly analysis functions from the new refactored module
from tools.analysis.weekly import run_weekly_analysis
# from tools.analysis.weekly import generate_weekly_newsletter

# Create router
router = APIRouter(prefix="/api/weekly-analysis", tags=["weekly-analysis"])

# Set up logging
logger = logging.getLogger(__name__)

@router.get("")
async def run_weekly_analysis_endpoint(
    metrics: Optional[str] = None,
    include_districts: bool = False
):
    """
    Run weekly analysis for specified metrics or all default metrics.
    
    Args:
        metrics: Comma-separated list of metric IDs to analyze. If not provided, uses default metrics.
        include_districts: Whether to process district-level data
    """
    try:
        # Convert metrics string to list if provided
        metrics_list = None
        if metrics:
            metrics_list = [m.strip() for m in metrics.split(",")]
            logger.info(f"Running weekly analysis for specified metrics: {metrics_list}")
        else:
            logger.info("No specific metrics provided, using default set")
        
        # Run the weekly analysis using the refactored module
        results = run_weekly_analysis(
            metrics_list=metrics_list,
            process_districts=include_districts
        )
        
        if not results:
            return JSONResponse({
                "success": False,
                "error": "No results generated from weekly analysis"
            })
        
        # Generate newsletter using the refactored module
        # Newsletter generation temporarily disabled
        # newsletter_path = generate_weekly_newsletter(results)
        
        return JSONResponse({
            "success": True,
            "message": f"Weekly analysis completed successfully for {len(results)} metrics",
            "results": {
                "total_metrics": len(results),
                # "newsletter_path": newsletter_path,
                "metric_ids": [r.get('metric_id') for r in results]
            }
        })
        
    except Exception as e:
        logger.error(f"Error in weekly analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@router.get("/report")
async def get_weekly_report():
    """Get the latest weekly report."""
    try:
        # Get the weekly output directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weekly_dir = os.path.join(script_dir, 'output', 'weekly')
        
        if not os.path.exists(weekly_dir):
            raise HTTPException(status_code=404, detail="Weekly reports directory not found")
        
        # Find the latest newsletter file
        newsletter_files = [f for f in os.listdir(weekly_dir) if f.startswith('weekly_newsletter_') and f.endswith('.md')]
        
        if not newsletter_files:
            raise HTTPException(status_code=404, detail="No weekly reports found")
        
        # Get the latest newsletter file
        latest_newsletter = sorted(newsletter_files)[-1]
        newsletter_path = os.path.join(weekly_dir, latest_newsletter)
        
        # Read the newsletter content
        with open(newsletter_path, 'r') as f:
            content = f.read()
        
        return JSONResponse({
            "success": True,
            "content": content,
            "filename": latest_newsletter
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting weekly report: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@router.get("/metrics/{metric_id}")
async def get_metric_weekly_analysis(metric_id: str, district: Optional[int] = None):
    """
    Get weekly analysis for a specific metric.
    
    Args:
        metric_id: The ID of the metric to get analysis for
        district: Optional district number to get district-specific analysis
    """
    try:
        # Get the weekly output directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weekly_dir = os.path.join(script_dir, 'output', 'weekly')
        
        # Determine the directory to look in
        if district is not None:
            metric_dir = os.path.join(weekly_dir, str(district))
        else:
            metric_dir = os.path.join(weekly_dir, "0")  # Citywide analysis
            
        if not os.path.exists(metric_dir):
            district_info = f"district {district}" if district is not None else "citywide"
            raise HTTPException(status_code=404, detail=f"Analysis directory not found for {district_info}")
        
        # Look for the metric's JSON file
        json_path = os.path.join(metric_dir, f"{metric_id}.json")
        
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"No weekly analysis found for metric {metric_id}")
        
        # Read the JSON file
        with open(json_path, 'r') as f:
            analysis_data = f.read()
        
        return JSONResponse({
            "success": True,
            "data": analysis_data
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting metric weekly analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500) 