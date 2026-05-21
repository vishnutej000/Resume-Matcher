"""Personal single-page LaTeX resume with role-aware JD selection.

Uses personal-template.tex (which mirrors tempo.tex exactly) as the base.
Only the Experience and Projects sections are dynamically filled from the
MD source files based on job-description relevance + declared role matching.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.llm import complete
from app.services.latex_resume import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_MIN_ITEMS,
    ExperienceEntry,
    LatexRenderError,
    ProjectEntry,
    _compile_latex_to_pdf,
    _escape_latex,
    _format_highlights,
    _format_subheading,
    _normalize_whitespace,
    _render_template,
    _sanitize_for_prompt,
    _select_entries_with_llm,
    _tokenize,
    _truncate_job_description,
    load_experience_entries,
    load_project_entries,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PERSONAL_TEMPLATE_PATH = _REPO_ROOT / "assets" / "latex-templates" / "personal-template.tex"

# One bullet per entry keeps the output identical to tempo.tex style (single page).
_MAX_BULLETS_PER_ENTRY = 1

# How many candidates to pre-select before auto-fit trims to 1 page.
_AUTO_FIT_POOL = 4

# Master skills list — full set of skills; JD-matching ones float to the top when a JD is given.
_MASTER_SKILLS: dict[str, list[str]] = {
    "Languages": ["Python", "Golang", "TypeScript", "JavaScript", "HTML", "CSS", "SQL", "Dart", "Bash/Shell"],
    "Frameworks": ["FastAPI", "Django", "Node.js", "Express.js", "React.js", "Next.js", "Flutter", "React Native"],
    "Databases": ["PostgreSQL", "MongoDB", "Vector DBs", "Firebase"],
    "Cloud & DevOps": ["GCP", "Docker", "Kubernetes", "Helm charts", "Terraform", "Linux", "Git", "CI/CD"],
    "Specialized": ["Machine Learning (NLP)", "RESTful API Design", "Finetuning AI model using LoRA", "LangGraph"],
}

# JD category detection — checked in priority order (most specific first).
_JD_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "android": ["android developer", "android application", "android sdk", "kotlin", "android studio", "android"],
    "mobile": ["mobile application", "flutter", "react native", "ios developer", "mobile developer"],
    "ml": ["machine learning", "deep learning", "nlp", "computer vision", "ai engineer", "data scientist", "llm"],
    "security": ["cybersecurity", "penetration testing", "siem", "soc analyst", "security engineer", "infosec", "threat detection"],
    "devops": ["devops", "kubernetes", "terraform", "ci/cd", "infrastructure engineer", "cloud engineer", "sre", "platform engineer"],
    "backend": ["backend developer", "api developer", "server-side", "fastapi", "django", "golang", "node.js"],
    "frontend": ["frontend developer", "ui developer", "react developer", "angular developer", "vue developer"],
    "fullstack": ["full stack", "fullstack", "full-stack"],
}

# Role-specific tailored summaries — written in first person to fit the role's language.
_SUMMARY_BY_ROLE: dict[str, str] = {
    "android": (
        "A Computer Science undergraduate with hands-on mobile application development experience "
        "using Flutter and cross-platform frameworks, with a strong foundation in software architecture, "
        "REST API consumption, and collaborative remote development. Brings mobile lifecycle knowledge, "
        "state management patterns, and a track record of building and shipping production mobile apps "
        "--- eager to apply these skills to native Android development."
    ),
    "mobile": (
        "A Computer Science undergraduate experienced in mobile application development using Flutter "
        "and cross-platform frameworks. Builds production-grade apps with REST API integration, state "
        "management, and deployment across Android and iOS. Backed by strong backend and cloud "
        "engineering skills spanning the full mobile development lifecycle."
    ),
    "ml": (
        "A Computer Science undergraduate with applied machine learning and AI engineering experience. "
        "Fine-tunes LLMs with LoRA, builds NLP pipelines from dataset synthesis through deployment, "
        "and delivers ML models as full-stack web applications using Python, LangGraph, and cloud "
        "infrastructure on GCP."
    ),
    "security": (
        "A Computer Science undergraduate with cybersecurity and AI security engineering experience. "
        "Built an ML-based Smishing Analyzer from scratch covering dataset synthesis, fine-tuning, "
        "and deployment. Contributes to SIEM-based threat detection workflows and manages cloud-native "
        "infrastructure on GCP and Azure."
    ),
    "devops": (
        "A Computer Science undergraduate with cloud and DevOps engineering experience. Manages "
        "infrastructure on GCP and Azure using Terraform and Kubernetes, containerises applications "
        "with Docker, and supports CI/CD pipelines for high-availability production deployments."
    ),
    "backend": (
        "A Computer Science undergraduate specialising in backend and API development. Builds scalable "
        "RESTful services with FastAPI and Django, manages PostgreSQL and MongoDB databases, and deploys "
        "containerised workloads with Docker and Kubernetes on GCP."
    ),
    "frontend": (
        "A Computer Science undergraduate specialising in frontend and full-stack web development. "
        "Builds responsive interfaces with React.js, Next.js, and TypeScript, backed by RESTful Python "
        "APIs and cloud-native infrastructure on GCP and Azure."
    ),
    "fullstack": (
        "A Computer Science undergraduate who builds and deploys full-stack web applications using "
        "React.js, Next.js, Node.js, and Python. Integrates REST APIs, manages SQL and NoSQL databases, "
        "and deploys containerised services on GCP and Azure."
    ),
    "default": (
        "A Computer Science undergraduate who builds and deploys full-stack and back-end applications. "
        "Specialises in creating containerised services with Docker and Kubernetes, developing RESTful "
        "APIs in Python, and architecting solutions using SQL and NoSQL databases on cloud platforms."
    ),
}


def _normalize_unicode_for_latex(text: str) -> str:
    """Replace common Unicode chars that pdflatex cannot handle with ASCII/LaTeX equivalents."""
    return (
        text.replace("—", "---")   # em dash  —
            .replace("–", "--")    # en dash  –
            .replace("‘", "`")     # left single quote  '
            .replace("’", "'")     # right single quote  '
            .replace("“", "``")    # left double quote  "
            .replace("”", "''")    # right double quote  "
            .replace("é", "\\'e")  # é
            .replace("à", "\\`a")  # à
            .replace("•", r"\textbullet{}")  # bullet •
    )


_CANDIDATE_PROFILE = """
Kollati Vishnu Teja — B.Tech Computer Science & Design, 2022–2026

INTERNSHIP EXPERIENCE:
• Web Developer @ Space ECE (Jul–Nov 2025, Pune): PHP, Laravel, HTML, Python — web apps, secure login/auth, admin dashboards, website responsiveness.
• Mobile App Developer @ Sanjeevani Vidhya Vikas (Aug–Nov 2025, Remote): Flutter, Dart, Python backend — transcription/translation app, read-aloud accessibility, login/auth, reminder app.
• Integration Testing Engineer @ Achala IT (Oct–Nov 2025, Hyderabad): Python — validated 5,000+ AI Drishti devices, integration testing, bug identification, performance testing.
• Security & AI Engineer @ Pi Sigma Cybersecurity (Sep 2025–Present, Hybrid): Built Smishing Analyzer ML model end-to-end (dataset synthesis → LoRA fine-tuning → full-stack deployment), SIEM threat detection for SOC teams, cloud infra on GCP/Azure/DigitalOcean with Terraform & Kubernetes, Django + Next.js full-stack apps.

PROJECTS:
• Cloust (Frontend Dev) — React Native, Spring Boot, Azure, Docker: cloud media storage mobile app.
• Notes App (Full Stack Dev) — MongoDB, Express.js, React.js, Node.js: MERN notes management web app.
• Neuro Nexus (Backend Dev) — Python, Next.js, ML, LLMs, Docker: healthcare accessibility web app (GDG Hackathon).
• Deep Detect — Python, ML: object detection system.
• Handwritten Text Detection — Python, ML, Computer Vision.
• Fake News Detection — Python, ML, NLP: text classification model.

SKILLS: Python, JavaScript, TypeScript, Golang, HTML, CSS, SQL, Dart, Bash/Shell;
Frameworks: FastAPI, Django, Node.js, Express.js, React.js, Next.js, Flutter, React Native;
Databases: PostgreSQL, MongoDB, Firebase;
Cloud & DevOps: Docker, Kubernetes, Terraform, GCP, Azure, Git, CI/CD;
Specialised: ML/NLP, LoRA fine-tuning, LangGraph, RESTful API Design, SIEM.
""".strip()

_SUMMARY_SYSTEM_PROMPT = (
    "You are an expert resume writer. Write only the requested summary text — "
    "no headers, no bullet points, no labels, no extra commentary."
)

_SUMMARY_USER_PROMPT = """Write a 2-3 sentence professional resume summary for the candidate below, \
tailored to the specific job description. Lead with what is MOST relevant to this role. \
Be honest — do not claim skills the candidate doesn't have, but frame transferable skills \
competitively. Active voice, professional tone. Maximum 55 words.

CANDIDATE:
{profile}

JOB DESCRIPTION:
{jd}"""


def _detect_jd_category(job_description: str) -> str:
    """Return the best-matching role category for this JD."""
    jd_lower = job_description.lower()
    for category, keywords in _JD_CATEGORY_KEYWORDS.items():
        if any(kw in jd_lower for kw in keywords):
            return category
    return "default"


def _category_summary_section(job_description: str | None = None) -> str:
    """Fallback: category-based summary when LLM is unavailable."""
    category = _detect_jd_category(job_description) if job_description else "default"
    text = _SUMMARY_BY_ROLE.get(category, _SUMMARY_BY_ROLE["default"])
    return f"\\section{{Summary}}\n{_escape_latex(text)}"


async def _build_personal_summary_section(job_description: str | None = None) -> str:
    """Return a \\section{Summary} block tailored to any JD via LLM.

    Falls back to the category-based template if the LLM call fails.
    """
    if not job_description:
        return _category_summary_section(None)

    prompt = _SUMMARY_USER_PROMPT.format(
        profile=_CANDIDATE_PROFILE,
        jd=_truncate_job_description(job_description),
    )
    try:
        text = await complete(
            prompt,
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            max_tokens=120,
            temperature=0.4,
        )
        text = _normalize_unicode_for_latex(text.strip().strip('"').strip())
        if text:
            return f"\\section{{Summary}}\n{_escape_latex(text)}"
    except Exception as exc:
        logger.warning("LLM summary generation failed: %s", exc)

    return _category_summary_section(job_description)


# Role categories for bonus scoring.
# Each overlapping category between the entry's declared role and the JD earns +5 points.
_ROLE_CATEGORIES: dict[str, list[str]] = {
    "backend": [
        "backend", "back-end", "server-side", "api developer", "fastapi",
        "django", "flask", "express", "golang", "node.js",
    ],
    "frontend": [
        "frontend", "front-end", "ui developer", "react developer",
        "vue", "angular", "next.js", "nextjs",
    ],
    "fullstack": ["full stack", "full-stack", "fullstack"],
    "mobile": ["mobile", "flutter", "react native", "ios developer", "android developer"],
    "devops": [
        "devops", "sre", "kubernetes", "platform engineer",
        "cloud engineer", "terraform", "infrastructure",
    ],
    "security": [
        "security engineer", "cybersecurity", "penetration", "siem",
        "soc analyst", "threat", "infosec",
    ],
    "ml": [
        "machine learning", "ml engineer", "ai engineer", "nlp",
        "deep learning", "data scientist", "computer vision",
    ],
    "testing": ["test engineer", "qa engineer", "quality assurance", "sdet", "automation tester"],
}


def _role_bonus(role: str | None, jd_text: str) -> int:
    """Extra score when the entry's declared role aligns categorically with the JD."""
    if not role:
        return 0
    jd_lower = jd_text.lower()
    role_lower = role.lower()
    bonus = 0
    for _, keywords in _ROLE_CATEGORIES.items():
        if any(kw in jd_lower for kw in keywords) and any(kw in role_lower for kw in keywords):
            bonus += 5
    return bonus


def _pick_jd_best_role(role: str | None, job_description: str | None) -> str:
    """From a comma-separated role list, return the role whose words appear most in the JD."""
    if not role:
        return ""
    parts = [r.strip() for r in role.split(",") if r.strip()]
    if not parts:
        return ""
    if not job_description or len(parts) == 1:
        return parts[0]
    jd_lower = job_description.lower()
    best, best_score = parts[0], -1
    for r in parts:
        score = sum(1 for w in r.lower().split() if len(w) > 2 and w in jd_lower)
        if score > best_score:
            best_score, best = score, r
    return best


def _build_personal_skills_section(job_description: str | None = None) -> str:
    """Skills section with JD-relevant skills sorted to the front of each category."""
    lines: list[str] = ["\\section{Skills}", ""]
    items = list(_MASTER_SKILLS.items())
    for i, (category, skills) in enumerate(items):
        if job_description:
            jd_lower = job_description.lower()
            ordered = sorted(
                skills,
                key=lambda s, jd=jd_lower: (
                    0 if any(w in jd for w in s.lower().replace(".", " ").split() if len(w) > 1) else 1,
                    skills.index(s),
                ),
            )
        else:
            ordered = skills
        safe_cat = _escape_latex(category)
        safe_vals = ", ".join(_escape_latex(s) for s in ordered)
        suffix = "\\\\[2pt]" if i < len(items) - 1 else ""
        lines.append(f"\\skillitem{{{safe_cat}}}{{{safe_vals}}}{suffix}")
    return "\n".join(lines)


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count pages in a compiled PDF by scanning raw bytes."""
    import re
    matches = re.findall(rb'/Type\s*/Page(?!s)', pdf_bytes)
    return max(1, len(matches))


def _score_with_role(
    entries: list[ExperienceEntry | ProjectEntry],
    job_description: str,
) -> list[int]:
    jd_tokens = _tokenize(job_description)
    return [
        len(_tokenize(e.raw_text) & jd_tokens) + _role_bonus(e.role, job_description)
        for e in entries
    ]


def _fallback_select(
    entries: list[ExperienceEntry | ProjectEntry],
    job_description: str,
    max_items: int,
    min_items: int,
) -> list[ExperienceEntry | ProjectEntry]:
    scores = _score_with_role(entries, job_description)
    ranked = sorted(
        zip(entries, scores),
        key=lambda pair: (pair[1], -pair[0].entry_id),
        reverse=True,
    )
    count = max(min_items, min(max_items, len(ranked)))
    return [e for e, _ in ranked[:count]]


async def _select_personal_entries(
    entries: list[ExperienceEntry | ProjectEntry],
    job_description: str,
    max_items: int,
    min_items: int,
    entry_type: str,
) -> list[ExperienceEntry | ProjectEntry]:
    """LLM-first selection with role-aware score fallback.

    Always returns exactly max_items entries when available — if the LLM picks
    fewer, the remaining slots are filled with the best-scoring unselected entries.
    """
    if not entries:
        return []
    if len(entries) <= max_items:
        return entries

    sanitized = _sanitize_for_prompt(_truncate_job_description(job_description))
    selected: list[ExperienceEntry | ProjectEntry] = []
    try:
        ids = await _select_entries_with_llm(entries, sanitized, max_items, entry_type)
        lookup = {e.entry_id: e for e in entries}
        selected = [lookup[i] for i in ids if i in lookup]
    except Exception as exc:
        logger.warning("LLM selection failed (%s): %s", entry_type, exc)

    # Fill any remaining slots with the best-scoring entries the LLM didn't pick
    if len(selected) < max_items:
        selected_ids = {e.entry_id for e in selected}
        remaining = [e for e in entries if e.entry_id not in selected_ids]
        scores = _score_with_role(remaining, job_description)
        ranked = sorted(zip(remaining, scores), key=lambda p: (-p[1], p[0].entry_id))
        while len(selected) < max_items and ranked:
            selected.append(ranked.pop(0)[0])

    # Absolute fallback: replace everything with scored selection
    if len(selected) < min_items:
        selected = _fallback_select(entries, job_description, max_items, min_items)

    return selected[:max_items]


def _build_personal_experience_section(
    experiences: list[ExperienceEntry],
    job_description: str | None = None,
) -> str:
    """Experience section matching tempo.tex format: company \\hfill location \\\\ role \\hfill date."""
    if not experiences:
        return ""

    lines: list[str] = ["\\section{Experience}", ""]
    for entry in experiences:
        company = _escape_latex(_normalize_whitespace(entry.company or entry.title))
        location = _escape_latex(_normalize_whitespace(entry.location or ""))

        if location:
            lines.append(f"\\textbf{{{company}}} \\hfill {location} \\\\")
        else:
            lines.append(f"\\textbf{{{company}}} \\\\")

        # Pick the role whose words appear most in the JD (e.g. "Full Stack Developer"
        # instead of "Security Engineer" when the JD is a fullstack role).
        display_role = _pick_jd_best_role(entry.role, job_description)

        sub = _format_subheading(display_role or entry.title, entry.duration)
        if sub:
            lines.append(sub)

        highlights = _format_highlights(entry.bullets[:_MAX_BULLETS_PER_ENTRY])
        if highlights:
            lines.append(highlights)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_personal_projects_section(
    projects: list[ProjectEntry],
    job_description: str | None = None,
) -> str:
    """Projects section matching tempo.tex format: Name (Role) \\hfill Tech Stack: ..."""
    if not projects:
        return ""

    lines: list[str] = ["\\section{Projects}"]
    for entry in projects:
        display_role = _pick_jd_best_role(entry.role, job_description)
        raw_name = f"{entry.name} ({display_role})" if display_role else entry.name
        safe_name = _escape_latex(_normalize_whitespace(raw_name))

        tech_stack = ", ".join(entry.tech_stack)
        if tech_stack:
            safe_tech = _escape_latex(_normalize_whitespace(tech_stack))
            lines.append(
                f"\\textbf{{{safe_name}}} \\hfill \\textit{{Tech Stack: {safe_tech}}}"
            )
        else:
            lines.append(f"\\textbf{{{safe_name}}}")

        highlights = _format_highlights(entry.bullets[:_MAX_BULLETS_PER_ENTRY])
        if highlights:
            lines.append(highlights)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_personal_latex_document(
    experiences: list[ExperienceEntry],
    projects: list[ProjectEntry],
    job_description: str | None = None,
    summary_section: str | None = None,
    template_path: Path | None = None,
) -> str:
    path = template_path or _PERSONAL_TEMPLATE_PATH
    if not path.exists():
        raise LatexRenderError(f"Personal LaTeX template not found: {path}")

    template = path.read_text(encoding="utf-8")
    return _render_template(
        template,
        {
            "SUMMARY_SECTION": summary_section or _category_summary_section(job_description),
            "EXPERIENCE_SECTION": _build_personal_experience_section(experiences, job_description),
            "PROJECTS_SECTION": _build_personal_projects_section(projects, job_description),
            "SKILLS_SECTION": _build_personal_skills_section(job_description),
        },
    )


def _auto_fit_configs(max_e: int, max_p: int) -> list[tuple[int, int]]:
    """Generate (n_exp, n_proj) configs ordered from most content to least.

    Tries the largest total first so the page is as full as possible.
    Prefers keeping more experience entries when reducing.
    """
    seen: set[tuple[int, int]] = set()
    configs: list[tuple[int, int]] = []
    # Generate all valid combos and sort descending by total, then by n_exp (keep exp)
    for total in range(max_e + max_p, DEFAULT_MIN_ITEMS * 2 - 1, -1):
        for ne in range(max_e, DEFAULT_MIN_ITEMS - 1, -1):
            np = total - ne
            if DEFAULT_MIN_ITEMS <= np <= max_p and (ne, np) not in seen:
                configs.append((ne, np))
                seen.add((ne, np))
    return configs


async def generate_personal_latex_resume_pdf(
    *,
    job_description: str | None = None,
    max_experiences: int = DEFAULT_MAX_ITEMS,
    max_projects: int = DEFAULT_MAX_ITEMS,
) -> bytes:
    """Generate a single-page personal LaTeX resume.

    Selects a pool of up to _AUTO_FIT_POOL candidates per section, then tries
    progressively smaller subsets until the result fits in exactly one page.
    If the largest pool fits, that version (most content) is returned so the
    page is as full as possible.
    """
    experience_entries = load_experience_entries()
    project_entries = load_project_entries()

    if job_description:
        # Run LLM calls in parallel: entry selection + summary generation
        (exp_pool, proj_pool), summary_section = await asyncio.gather(
            asyncio.gather(
                _select_personal_entries(
                    experience_entries,
                    job_description,
                    _AUTO_FIT_POOL,
                    DEFAULT_MIN_ITEMS,
                    "experience",
                ),
                _select_personal_entries(
                    project_entries,
                    job_description,
                    _AUTO_FIT_POOL,
                    DEFAULT_MIN_ITEMS,
                    "project",
                ),
            ),
            _build_personal_summary_section(job_description),
        )
    else:
        # No JD — take the most recently listed entries (bottom of each MD file).
        exp_pool = experience_entries[-_AUTO_FIT_POOL:]
        proj_pool = project_entries[-_AUTO_FIT_POOL:]
        summary_section = _category_summary_section(None)

    # Sort pools so the most JD-relevant entry always appears first on the page.
    if job_description:
        exp_pool = sorted(
            exp_pool,
            key=lambda e: _score_with_role([e], job_description)[0],
            reverse=True,
        )
        proj_pool = sorted(
            proj_pool,
            key=lambda e: _score_with_role([e], job_description)[0],
            reverse=True,
        )

    max_e = min(_AUTO_FIT_POOL, len(exp_pool))
    max_p = min(_AUTO_FIT_POOL, len(proj_pool))
    configs = _auto_fit_configs(max_e, max_p)

    last_pdf: bytes | None = None
    for n_exp, n_proj in configs:
        exps = list(exp_pool[:n_exp])
        projs = list(proj_pool[:n_proj])
        latex = build_personal_latex_document(exps, projs, job_description, summary_section)
        pdf_bytes = _compile_latex_to_pdf(latex)
        pages = _count_pdf_pages(pdf_bytes)
        logger.debug("Auto-fit (%d exp, %d proj) → %d page(s)", n_exp, n_proj, pages)
        if pages == 1:
            return pdf_bytes
        # Overflows — save as fallback and try the next smaller config
        last_pdf = pdf_bytes

    # Nothing fit in 1 page; return the smallest config's output
    return last_pdf or pdf_bytes  # type: ignore[return-value]
