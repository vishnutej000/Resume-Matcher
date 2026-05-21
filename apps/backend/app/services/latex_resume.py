"""LaTeX resume rendering with JD-based selection."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.database import db
from app.llm import complete_json
from app.prompts import LATEX_SELECT_ENTRIES_PROMPT
from app.schemas import PersonalInfo, ResumeData, normalize_resume_data

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INFORMATION_DIR = _REPO_ROOT / "assets" / "information"
_TEMPLATE_PATH = _REPO_ROOT / "assets" / "latex-templates" / "prime-template.tex"

MAX_JD_LENGTH = 2000
DEFAULT_MIN_ITEMS = 2
DEFAULT_MAX_ITEMS = 3

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?above",
    r"forget\s+(everything|all)",
    r"new\s+instructions?:",
    r"system\s*:",
    r"<\s*/?\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"\[\s*/\s*INST\s*\]",
]


class LatexRenderError(Exception):
    """Raised when LaTeX PDF generation fails."""

    pass


@dataclass(frozen=True)
class ExperienceEntry:
    entry_id: int
    title: str
    company: str
    duration: str | None
    location: str | None
    role: str | None
    tags: list[str]
    bullets: list[str]
    raw_text: str


@dataclass(frozen=True)
class ProjectEntry:
    entry_id: int
    name: str
    role: str | None
    years: str | None
    tech_stack: list[str]
    tags: list[str]
    bullets: list[str]
    raw_text: str


_EXPERIENCE_HEADING_RE = re.compile(r"^##\s+(?P<title>.+)$", re.MULTILINE)
_PROJECT_HEADING_RE = re.compile(r"^#\s+(?P<title>.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^[\-\*\u2022]\s+(.*)$")


def _sanitize_for_prompt(text: str) -> str:
    sanitized = text
    for pattern in _INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


def _truncate_job_description(job_description: str) -> str:
    trimmed = job_description.strip()
    if len(trimmed) <= MAX_JD_LENGTH:
        return trimmed
    logger.warning("Job description truncated from %d to %d characters", len(trimmed), MAX_JD_LENGTH)
    return trimmed[:MAX_JD_LENGTH]


def _strip_markdown_decorators(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.strip("*").strip()
    cleaned = cleaned.strip("_").strip()
    return cleaned


def _extract_sections(text: str, heading_pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group("title").strip()
        body = text[start:end].strip()
        sections.append((heading, body))
    return sections


def _split_heading(heading: str) -> tuple[str, str]:
    parts = re.split(r"\s+(?:\u2014|\u2013|-)\s+", heading, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return heading.strip(), ""


def _parse_tags(value: str) -> list[str]:
    return [tag.strip() for tag in re.split(r"[,;]", value) if tag.strip()]


def _extract_bullets_or_paragraphs(lines: list[str]) -> list[str]:
    bullet_items: list[str] = []
    paragraph_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            paragraph_lines.append("")
            continue
        match = _BULLET_RE.match(stripped)
        if match:
            bullet_items.append(match.group(1).strip())
        else:
            paragraph_lines.append(stripped)

    if bullet_items:
        return bullet_items

    paragraphs: list[str] = []
    current: list[str] = []
    for line in paragraph_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())

    return paragraphs


def load_experience_entries(path: Path | None = None) -> list[ExperienceEntry]:
    source_path = path or (_INFORMATION_DIR / "experience.md")
    if not source_path.exists():
        raise LatexRenderError(f"Experience file not found: {source_path}")

    text = source_path.read_text(encoding="utf-8")
    sections = _extract_sections(text, _EXPERIENCE_HEADING_RE)
    entries: list[ExperienceEntry] = []

    for idx, (heading, body) in enumerate(sections, start=1):
        title, company = _split_heading(heading)
        duration: str | None = None
        location: str | None = None
        role: str | None = None
        tags: list[str] = []
        content_lines: list[str] = []

        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped == "---":
                content_lines.append("")
                continue

            clean_line = _strip_markdown_decorators(stripped)
            lower = clean_line.lower()

            if lower.startswith("duration:"):
                duration = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("years:"):
                duration = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("location:"):
                location = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("role:"):
                role = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("tags:"):
                tags = _parse_tags(clean_line.split(":", 1)[1])
                continue

            content_lines.append(clean_line)

        bullets = _extract_bullets_or_paragraphs(content_lines)
        raw_text = " ".join(
            [
                title,
                company,
                role or "",
                duration or "",
                location or "",
                " ".join(tags),
                " ".join(bullets),
            ]
        ).strip()

        entries.append(
            ExperienceEntry(
                entry_id=idx,
                title=title,
                company=company,
                duration=duration,
                location=location,
                role=role,
                tags=tags,
                bullets=[b for b in bullets if b],
                raw_text=raw_text,
            )
        )

    return entries


def load_project_entries(path: Path | None = None) -> list[ProjectEntry]:
    source_path = path or (_INFORMATION_DIR / "projects.md")
    if not source_path.exists():
        raise LatexRenderError(f"Projects file not found: {source_path}")

    text = source_path.read_text(encoding="utf-8")
    sections = _extract_sections(text, _PROJECT_HEADING_RE)
    entries: list[ProjectEntry] = []

    for idx, (heading, body) in enumerate(sections, start=1):
        name = heading.strip()
        role: str | None = None
        years: str | None = None
        tech_stack: list[str] = []
        tags: list[str] = []
        content_lines: list[str] = []

        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped == "---":
                content_lines.append("")
                continue

            clean_line = _strip_markdown_decorators(stripped)
            lower = clean_line.lower()

            if lower.startswith("role:"):
                role = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("tags:"):
                tags = _parse_tags(clean_line.split(":", 1)[1])
                continue
            if lower.startswith("duration:") or lower.startswith("years:"):
                years = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("tech stack:"):
                tech_stack = _parse_tags(clean_line.split(":", 1)[1])
                continue

            content_lines.append(clean_line)

        bullets = _extract_bullets_or_paragraphs(content_lines)
        raw_text = " ".join(
            [
                name,
                role or "",
                years or "",
                " ".join(tech_stack),
                " ".join(tags),
                " ".join(bullets),
            ]
        ).strip()

        entries.append(
            ProjectEntry(
                entry_id=idx,
                name=name,
                role=role,
                years=years,
                tech_stack=tech_stack,
                tags=tags,
                bullets=[b for b in bullets if b],
                raw_text=raw_text,
            )
        )

    return entries


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9+#.]+", text.lower()) if token}


def _score_entries(entries: list[ExperienceEntry | ProjectEntry], job_description: str) -> list[int]:
    jd_tokens = _tokenize(job_description)
    scores: list[int] = []
    for entry in entries:
        entry_tokens = _tokenize(entry.raw_text)
        scores.append(len(entry_tokens & jd_tokens))
    return scores


def _select_by_score(
    entries: list[ExperienceEntry | ProjectEntry],
    job_description: str,
    max_items: int,
    min_items: int,
) -> list[ExperienceEntry | ProjectEntry]:
    if not entries:
        return []
    scores = _score_entries(entries, job_description)
    ranked = sorted(
        zip(entries, scores),
        key=lambda pair: (pair[1], -pair[0].entry_id),
        reverse=True,
    )
    picked = [entry for entry, _score in ranked]
    return picked[: max(min_items, min(max_items, len(picked)))]


async def _select_entries_with_llm(
    entries: list[ExperienceEntry | ProjectEntry],
    job_description: str,
    max_items: int,
    entry_type: str,
) -> list[int]:
    payload = []
    for entry in entries:
        if isinstance(entry, ExperienceEntry):
            payload.append(
                {
                    "id": entry.entry_id,
                    "title": entry.title,
                    "company": entry.company,
                    "role": entry.role,
                    "tags": entry.tags,
                    "summary": " ".join(entry.bullets),
                }
            )
        else:
            payload.append(
                {
                    "id": entry.entry_id,
                    "name": entry.name,
                    "role": entry.role,
                    "tech_stack": entry.tech_stack,
                    "tags": entry.tags,
                    "summary": " ".join(entry.bullets),
                }
            )

    prompt = LATEX_SELECT_ENTRIES_PROMPT.format(
        entry_type=entry_type,
        job_description=job_description,
        max_items=max_items,
        entries_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )

    result = await complete_json(
        prompt=prompt,
        system_prompt="You are a strict JSON selection engine.",
        max_tokens=600,
        retries=2,
        schema_type="keywords",
    )

    selected = result.get("selected_ids", []) if isinstance(result, dict) else []
    selected_ids: list[int] = []
    for value in selected:
        if isinstance(value, int):
            selected_ids.append(value)
        elif isinstance(value, str) and value.isdigit():
            selected_ids.append(int(value))
    return selected_ids


async def select_relevant_entries(
    entries: list[ExperienceEntry | ProjectEntry],
    job_description: str,
    max_items: int,
    min_items: int,
    entry_type: str,
) -> list[ExperienceEntry | ProjectEntry]:
    if not entries:
        return []
    if len(entries) <= max_items:
        return entries

    sanitized = _sanitize_for_prompt(_truncate_job_description(job_description))

    try:
        selected_ids = await _select_entries_with_llm(
            entries,
            sanitized,
            max_items,
            entry_type,
        )
        lookup = {entry.entry_id: entry for entry in entries}
        selected = [lookup[entry_id] for entry_id in selected_ids if entry_id in lookup]
    except Exception as exc:
        logger.warning("LLM selection failed (%s): %s", entry_type, exc)
        selected = []

    if len(selected) < min_items:
        selected = _select_by_score(entries, job_description, max_items, min_items)

    return selected[: max_items]


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _build_contact_line(info: PersonalInfo) -> str:
    pieces: list[str] = []

    def add_piece(icon: str, href: str, label: str) -> None:
        safe_href = _escape_latex(href)
        safe_label = _escape_latex(label)
        pieces.append(f"{icon} \\href{{{safe_href}}}{{{safe_label}}}")

    if info.phone:
        add_piece("\\faPhone", f"tel:{info.phone}", info.phone)
    if info.email:
        add_piece("\\faEnvelope", f"mailto:{info.email}", info.email)
    if info.linkedin:
        url = _normalize_url(info.linkedin)
        add_piece("\\faLinkedin", url, "LinkedIn")
    if info.github:
        url = _normalize_url(info.github)
        add_piece("\\faGithub", url, "GitHub")
    if info.website:
        url = _normalize_url(info.website)
        add_piece("\\faGlobe", url, "Portfolio")

    if not pieces:
        return ""

    return " \\quad\\textbar\\quad ".join(pieces) + " \\\\[3pt]"


def _normalize_url(value: str) -> str:
    trimmed = value.strip()
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed
    return f"https://{trimmed}"


def _format_heading(left: str, right: str | None, bold: bool = True) -> str:
    safe_left = _escape_latex(_normalize_whitespace(left))
    if bold:
        safe_left = f"\\textbf{{{safe_left}}}"
    if right:
        safe_right = _escape_latex(_normalize_whitespace(right))
        return f"{safe_left} \\hfill {safe_right}"
    return safe_left


def _format_subheading(left: str, right: str | None) -> str:
    safe_left = _escape_latex(_normalize_whitespace(left)) if left else ""
    safe_right = _escape_latex(_normalize_whitespace(right)) if right else ""
    if safe_left:
        safe_left = f"\\textit{{{safe_left}}}"
    if safe_right:
        safe_right = f"\\textit{{{safe_right}}}"
    if safe_left and safe_right:
        return f"{safe_left} \\hfill {safe_right}"
    return safe_left or safe_right


def _format_highlights(items: list[str]) -> str:
    if not items:
        return ""
    lines = ["\\begin{highlights}"]
    for item in items:
        safe_item = _escape_latex(_normalize_whitespace(item))
        lines.append(f"\\item {safe_item}")
    lines.append("\\end{highlights}")
    return "\n".join(lines)


def _build_summary_section(summary: str) -> str:
    if not summary.strip():
        return ""
    safe_summary = _escape_latex(_normalize_whitespace(summary))
    return f"\\section{{Summary}}\n{safe_summary}\n"


def _build_education_section(education: list[Any]) -> str:
    if not education:
        return ""
    lines: list[str] = ["\\section{Education}"]
    for entry in education:
        institution = getattr(entry, "institution", "") or ""
        degree = getattr(entry, "degree", "") or ""
        years = getattr(entry, "years", "") or ""
        lines.append(_format_heading(institution, None))
        sub = _format_subheading(degree, years)
        if sub:
            lines.append(sub)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_experience_section(experiences: list[ExperienceEntry]) -> str:
    if not experiences:
        return ""
    lines: list[str] = ["\\section{Experience}"]
    for entry in experiences:
        lines.append(_format_heading(entry.company or entry.title, entry.location))
        role_text = entry.role or entry.title
        sub = _format_subheading(role_text, entry.duration)
        if sub:
            lines.append(sub)
        highlights = _format_highlights(entry.bullets)
        if highlights:
            lines.append(highlights)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_projects_section(projects: list[ProjectEntry]) -> str:
    if not projects:
        return ""
    lines: list[str] = ["\\section{Projects}"]
    for entry in projects:
        tech_stack = ", ".join(entry.tech_stack)
        right = f"Tech Stack: {tech_stack}" if tech_stack else ""
        lines.append(_format_heading(entry.name, right))
        sub = _format_subheading(entry.role or "", entry.years)
        if sub:
            lines.append(sub)
        highlights = _format_highlights(entry.bullets)
        if highlights:
            lines.append(highlights)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_skills_section(additional: Any) -> str:
    lines: list[str] = []

    technical = getattr(additional, "technicalSkills", []) or []
    languages = getattr(additional, "languages", []) or []
    certifications = getattr(additional, "certificationsTraining", []) or []
    awards = getattr(additional, "awards", []) or []

    if technical:
        lines.append(
            f"\\skillitem{{Technical}}{{{_escape_latex(', '.join(technical))}}}"
        )
    if languages:
        lines.append(
            f"\\skillitem{{Languages}}{{{_escape_latex(', '.join(languages))}}}"
        )
    if certifications:
        lines.append(
            f"\\skillitem{{Certifications}}{{{_escape_latex(', '.join(certifications))}}}"
        )
    if awards:
        lines.append(
            f"\\skillitem{{Awards}}{{{_escape_latex(', '.join(awards))}}}"
        )

    if not lines:
        return ""

    return "\\section{Skills}\n" + "\\\n".join(lines) + "\n"


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _resolve_resume_data(resume: dict[str, Any]) -> ResumeData:
    master = db.get_master_resume()
    base_data = _load_resume_data(master) if master else None
    if not base_data:
        base_data = _load_resume_data(resume)

    if not base_data:
        raise LatexRenderError("Resume data unavailable. Upload a master resume first.")

    normalized = normalize_resume_data(base_data)
    return ResumeData.model_validate(normalized)


def _load_resume_data(resume: dict[str, Any] | None) -> dict[str, Any] | None:
    if not resume:
        return None
    data = resume.get("processed_data")
    if isinstance(data, dict):
        return data
    if resume.get("content_type") == "json":
        content = resume.get("content", "")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.warning("Failed to parse resume JSON content for %s", resume.get("resume_id"))
    return None


def _resolve_latex_engine(preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["pdflatex", "tectonic", "xelatex", "lualatex"])
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    raise LatexRenderError(
        "No LaTeX engine found. Install pdflatex, tectonic, xelatex, or lualatex."
    )


def _compile_latex_to_pdf(latex: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "resume.tex"
        tex_path.write_text(latex, encoding="utf-8")

        engine = _resolve_latex_engine()
        if engine == "tectonic":
            command = [engine, "--outdir", tmpdir, str(tex_path)]
        else:
            command = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                tmpdir,
                str(tex_path),
            ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=40,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("LaTeX compilation timed out: %s", exc)
            raise LatexRenderError("LaTeX compilation timed out.") from exc
        except FileNotFoundError as exc:
            logger.error("LaTeX engine not found: %s", exc)
            raise LatexRenderError("LaTeX engine not found.") from exc
        except subprocess.CalledProcessError as exc:
            logger.error("LaTeX compilation failed: %s", exc.stdout)
            logger.error("LaTeX compilation stderr: %s", exc.stderr)
            raise LatexRenderError("LaTeX compilation failed.") from exc

        pdf_path = Path(tmpdir) / "resume.pdf"
        if not pdf_path.exists():
            raise LatexRenderError("LaTeX engine did not produce a PDF.")
        return pdf_path.read_bytes()


def build_latex_document(
    *,
    resume_data: ResumeData,
    experiences: list[ExperienceEntry],
    projects: list[ProjectEntry],
    template_path: Path | None = None,
) -> str:
    path = template_path or _TEMPLATE_PATH
    if not path.exists():
        raise LatexRenderError(f"LaTeX template not found: {path}")

    template = path.read_text(encoding="utf-8")

    name = _escape_latex(resume_data.personalInfo.name or "")
    title = _escape_latex(resume_data.personalInfo.title or "")
    contact_line = _build_contact_line(resume_data.personalInfo)

    summary_section = _build_summary_section(resume_data.summary)
    education_section = _build_education_section(resume_data.education)
    experience_section = _build_experience_section(experiences)
    projects_section = _build_projects_section(projects)
    skills_section = _build_skills_section(resume_data.additional)

    return _render_template(
        template,
        {
            "NAME": name,
            "TITLE": title,
            "CONTACT_LINE": contact_line,
            "SUMMARY_SECTION": summary_section,
            "EDUCATION_SECTION": education_section,
            "EXPERIENCE_SECTION": experience_section,
            "PROJECTS_SECTION": projects_section,
            "SKILLS_SECTION": skills_section,
        },
    )


async def generate_latex_resume_pdf(
    *,
    resume: dict[str, Any],
    job_description: str,
    max_experiences: int = DEFAULT_MAX_ITEMS,
    max_projects: int = DEFAULT_MAX_ITEMS,
) -> bytes:
    resume_data = _resolve_resume_data(resume)

    experience_entries = load_experience_entries()
    project_entries = load_project_entries()

    max_experiences = max(DEFAULT_MIN_ITEMS, min(DEFAULT_MAX_ITEMS, max_experiences))
    max_projects = max(DEFAULT_MIN_ITEMS, min(DEFAULT_MAX_ITEMS, max_projects))

    selected_experiences, selected_projects = await asyncio.gather(
        select_relevant_entries(
            experience_entries,
            job_description,
            max_experiences,
            DEFAULT_MIN_ITEMS,
            "experience",
        ),
        select_relevant_entries(
            project_entries,
            job_description,
            max_projects,
            DEFAULT_MIN_ITEMS,
            "project",
        ),
    )

    latex = build_latex_document(
        resume_data=resume_data,
        experiences=selected_experiences,
        projects=selected_projects,
    )

    return _compile_latex_to_pdf(latex)
