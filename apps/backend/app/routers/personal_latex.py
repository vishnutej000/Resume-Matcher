"""Personal single-page LaTeX resume PDF generation endpoint."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.database import db
from app.schemas import PersonalLatexRequest
from app.services.latex_resume import LatexRenderError
from app.services.personal_resume import generate_personal_latex_resume_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/latex", tags=["LaTeX"])


@router.post("/personal-resume-pdf")
async def generate_personal_resume_pdf_endpoint(request: PersonalLatexRequest) -> Response:
    """Generate a personal single-page LaTeX resume PDF.

    When ``job_id`` is supplied the most relevant 2-3 experience and project
    entries are selected for that job description.  When omitted the most-
    recently-listed entries are used (useful for the master resume view).
    """
    job_description: str | None = None
    if request.job_id:
        job = db.get_job(request.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job description not found")
        job_description = job["content"]

    try:
        pdf_bytes = await generate_personal_latex_resume_pdf(
            job_description=job_description,
            max_experiences=request.max_experiences,
            max_projects=request.max_projects,
        )
    except LatexRenderError as exc:
        logger.error("Personal LaTeX PDF generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to generate personal LaTeX PDF. Please try again.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected personal LaTeX PDF error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error generating personal LaTeX PDF.",
        ) from exc

    job_suffix = request.job_id or "general"
    headers = {
        "Content-Disposition": f'attachment; filename="resume_{job_suffix}.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
