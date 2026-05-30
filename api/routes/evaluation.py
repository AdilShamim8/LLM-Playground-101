"""
Evaluation API endpoints.
Run benchmarks and compute metrics via REST API.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
import time

from loguru import logger

router = APIRouter()

# In-memory job store
eval_jobs: dict[str, dict] = {}


class EvalRequest(BaseModel):
    model_name: str
    task: str  # "perplexity" | "bleu" | "rouge" | "mmlu" | "gsm8k"
    texts: Optional[list[str]] = None
    hypotheses: Optional[list[str]] = None
    references: Optional[list[str]] = None
    num_examples: int = 100


class EvalResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[dict] = None


async def _run_eval_job(
    job_id: str,
    request: EvalRequest
):
    """Background evaluation job."""
    eval_jobs[job_id]["status"] = "running"
    t0 = time.time()

    try:
        results = {}

        if request.task == "bleu":
            from evaluation.metrics import BLEUScore
            bleu = BLEUScore()
            hyps = request.hypotheses or []
            refs = request.references or []
            results = bleu.compute(
                hyps, [[r] for r in refs]
            )

        elif request.task == "rouge":
            from evaluation.metrics import ROUGEScore
            rouge = ROUGEScore()
            results = rouge.compute(
                request.hypotheses or [],
                request.references or []
            )

        else:
            results = {
                "task": request.task,
                "note": "Task queued. Results pending."
            }

        elapsed = time.time() - t0
        eval_jobs[job_id].update({
            "status": "completed",
            "results": results,
            "elapsed": elapsed
        })
        logger.info(
            f"Eval job {job_id} done in {elapsed:.2f}s"
        )

    except Exception as e:
        eval_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"Eval job {job_id} failed: {e}")


@router.post("/evaluate", response_model=EvalResponse)
async def start_evaluation(
    request: EvalRequest,
    background_tasks: BackgroundTasks
):
    """Start an evaluation job in the background."""
    job_id = f"eval-{uuid.uuid4().hex[:8]}"
    eval_jobs[job_id] = {
        "status": "pending",
        "task": request.task,
        "model": request.model_name,
        "created_at": time.time(),
    }
    background_tasks.add_task(_run_eval_job, job_id, request)
    return EvalResponse(job_id=job_id, status="pending")


@router.get("/evaluate/{job_id}", response_model=EvalResponse)
async def get_evaluation_result(job_id: str):
    """Get evaluation job status and results."""
    if job_id not in eval_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = eval_jobs[job_id]
    return EvalResponse(
        job_id=job_id,
        status=job["status"],
        results=job.get("results")
    )