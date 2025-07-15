import asyncio
import threading
import time
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BackgroundJob:
    """Represents a background job with status tracking."""
    
    def __init__(self, job_id: str, job_type: str, description: str):
        self.job_id = job_id
        self.job_type = job_type
        self.description = description
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.created_at = datetime.now()
        
    def start(self):
        """Mark the job as started."""
        self.status = "running"
        self.start_time = datetime.now()
        logger.info(f"Job {self.job_id} ({self.job_type}) started")
        
    def complete(self, result: Any = None):
        """Mark the job as completed."""
        self.status = "completed"
        self.result = result
        self.end_time = datetime.now()
        self.progress = 100
        logger.info(f"Job {self.job_id} ({self.job_type}) completed")
        
    def fail(self, error: str):
        """Mark the job as failed."""
        self.status = "failed"
        self.error = error
        self.end_time = datetime.now()
        logger.error(f"Job {self.job_id} ({self.job_type}) failed: {error}")
        
    def update_progress(self, progress: int):
        """Update the job progress (0-100)."""
        self.progress = max(0, min(100, progress))
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "created_at": self.created_at.isoformat(),
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }

class BackgroundJobManager:
    """Manages background jobs and their execution."""
    
    def __init__(self):
        self.jobs: Dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        
    def create_job(self, job_type: str, description: str) -> str:
        """Create a new background job."""
        job_id = str(uuid.uuid4())
        job = BackgroundJob(job_id, job_type, description)
        
        with self._lock:
            self.jobs[job_id] = job
            
        logger.info(f"Created background job {job_id} ({job_type}): {description}")
        return job_id
        
    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        """Get a job by ID."""
        with self._lock:
            return self.jobs.get(job_id)
            
    def get_all_jobs(self) -> Dict[str, BackgroundJob]:
        """Get all jobs."""
        with self._lock:
            return self.jobs.copy()
            
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed/failed jobs."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job.status in ["completed", "failed"] and job.created_at.timestamp() < cutoff_time:
                    jobs_to_remove.append(job_id)
                    
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
            
    async def run_job(self, job_id: str, func: Callable, *args, **kwargs):
        """Run a job in the background."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
            
        job.start()
        
        try:
            # Run the function in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
            job.complete(result)
        except Exception as e:
            job.fail(str(e))
            raise

# Global job manager instance
job_manager = BackgroundJobManager()

# Start cleanup task
async def cleanup_old_jobs_periodic():
    """Periodically cleanup old jobs."""
    while True:
        try:
            job_manager.cleanup_old_jobs()
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")
        
        # Wait 1 hour before next cleanup
        await asyncio.sleep(3600)

# Start the cleanup task when the module is imported
def start_cleanup_task():
    """Start the periodic cleanup task."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(cleanup_old_jobs_periodic())
    else:
        # If no event loop is running, we'll start it when needed
        pass 