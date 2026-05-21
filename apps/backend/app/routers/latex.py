"""LaTeX resume PDF generation endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.database import db
from app.schemas import LatexResumeRequest
from app.services.latex_resume import LatexRenderError, generate_latex_resume_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/latex", tags=["LaTeX"])


@router.post("/resume-pdf")
async def generate_latex_resume_pdf_endpoint(request: LatexResumeRequest) -> Response:
    """Generate a JD-tailored LaTeX resume PDF for the given resume."""
    resume = db.get_resume(request.resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    job = db.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")

    try:
        pdf_bytes = await generate_latex_resume_pdf(
            resume=resume,
            job_description=job["content"],
            max_experiences=request.max_experiences,
            max_projects=request.max_projects,
        )
    except LatexRenderError as exc:
        logger.error("LaTeX PDF generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to generate LaTeX PDF. Please try again.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected LaTeX PDF error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error generating LaTeX PDF.",
        ) from exc

    headers = {
        "Content-Disposition": f'attachment; filename="latex_resume_{request.resume_id}.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
