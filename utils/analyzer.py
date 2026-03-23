import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[assignment]


def _section_word_count(sections: Dict[str, str], key: str) -> int:
    text = (sections.get(key) or "").strip()
    if not text:
        return 0
    return len(text.split())


def _normalize_token(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        nx = _normalize_token(x)
        if not nx or nx in seen:
            continue
        seen.add(nx)
        out.append(x.strip())
    return out


def _contains_any(text_lower: str, needles: Sequence[str]) -> bool:
    return any(n in text_lower for n in needles if n)


def extract_important_keywords(text: str, top_n: int = 18) -> List[str]:
    """
    Extract "important" keywords/phrases using TF-IDF (unigrams + bigrams).
    Falls back to a simple token heuristic if sklearn isn't available.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    if TfidfVectorizer is None:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}", raw.lower())
        # naive frequency
        freq: Dict[str, int] = {}
        for t in tokens:
            if len(t) <= 2:
                continue
            freq[t] = freq.get(t, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in ranked[:top_n]]

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=2500,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}\b",
    )
    X = vec.fit_transform([raw])
    if X.nnz == 0:
        return []
    feature_names = vec.get_feature_names_out()
    scores = X.toarray()[0]
    ranked_idx = scores.argsort()[::-1]
    keywords: List[str] = []
    for i in ranked_idx:
        if scores[i] <= 0:
            break
        kw = feature_names[i].strip()
        if len(kw) <= 2:
            continue
        keywords.append(kw)
        if len(keywords) >= top_n:
            break
    return _unique_preserve_order(keywords)


@dataclass(frozen=True)
class JobMatchResult:
    match_pct: int
    matched_keywords: List[str]
    missing_keywords: List[str]
    extracted_jd_keywords: List[str]


def compute_job_description_match(
    resume_text: str,
    job_description: str,
    role_skills: Iterable[str] = (),
    role_keywords: Optional[Iterable[str]] = None,
) -> JobMatchResult:
    """
    Compare resume text against a pasted job description.

    - Extracts keywords from the JD (TF-IDF) + includes role skills/keywords as "expected terms"
    - Calculates match % based on presence of expected terms in resume text
    """
    resume_lower = (resume_text or "").lower()
    jd_raw = (job_description or "").strip()
    if not jd_raw:
        return JobMatchResult(
            match_pct=0,
            matched_keywords=[],
            missing_keywords=[],
            extracted_jd_keywords=[],
        )

    jd_keywords = extract_important_keywords(jd_raw, top_n=22)
    expected_terms = _unique_preserve_order(
        list(jd_keywords)
        + list(role_skills or [])
        + list(role_keywords or [])
    )

    matched = [t for t in expected_terms if _normalize_token(t) and _normalize_token(t) in resume_lower]
    missing = [t for t in expected_terms if _normalize_token(t) and _normalize_token(t) not in resume_lower]

    denom = max(1, len(expected_terms))
    match_pct = int(round(100 * (len(matched) / float(denom))))
    match_pct = max(0, min(100, match_pct))

    return JobMatchResult(
        match_pct=match_pct,
        matched_keywords=matched[:30],
        missing_keywords=missing[:30],
        extracted_jd_keywords=jd_keywords,
    )


def compute_ats_score(
    sections: Dict[str, str],
    resume_skills: Iterable[str],
    role_skills: Iterable[str],
    role_keywords: Optional[Iterable[str]] = None,
) -> Tuple[int, Dict[str, int]]:
    """
    ATS compatibility score out of 100 based on:
    - keyword presence (role skills + role keywords)
    - formatting (bullets, section breaks, links)
    - section completeness (core sections present)

    Returns (ats_score, breakdown_dict).
    """
    raw_text = (sections.get("raw_text") or "")
    raw_lower = raw_text.lower()

    # Weights
    w_keyword = 45
    w_format = 25
    w_complete = 30

    role_skills_set = {_normalize_token(s) for s in role_skills or [] if _normalize_token(s)}
    role_keywords_set = {_normalize_token(k) for k in (role_keywords or []) if _normalize_token(k)}
    expected = sorted(role_skills_set | role_keywords_set)

    keyword_ratio = 0.0
    if expected:
        hits = sum(1 for t in expected if t in raw_lower)
        keyword_ratio = hits / float(len(expected))
    else:
        # fall back to detected skills density
        keyword_ratio = min(1.0, len(list(resume_skills or [])) / 20.0)

    keyword_score = int(round(w_keyword * keyword_ratio))

    # Formatting heuristics (text-only, but still useful)
    bullet_like = raw_text.count("\n-") + raw_text.count("\n•") + raw_text.count("\n–") + raw_text.count("\n*")
    has_bullets = bullet_like >= 3
    has_section_breaks = raw_text.count("\n\n") >= 3
    has_links = _contains_any(raw_lower, ["linkedin.com", "github.com", "gitlab.com", "bitbucket.org", "portfolio", "http://", "https://"])
    line_lengths = [len(l.strip()) for l in raw_text.splitlines() if l.strip()]
    avg_line = (sum(line_lengths) / float(len(line_lengths))) if line_lengths else 0.0
    avoids_wall_of_text = avg_line <= 140  # rough ATS-friendly signal

    format_points = 0
    format_points += 10 if has_bullets else 0
    format_points += 8 if has_section_breaks else 0
    format_points += 4 if has_links else 0
    format_points += 3 if avoids_wall_of_text else 0
    format_score = int(round(w_format * (format_points / 25.0)))

    # Completeness: presence of sections
    keys_to_check = ["name", "skills", "education", "projects", "experience"]
    present_count = sum(1 for k in keys_to_check if (sections.get(k) or "").strip())
    completeness_ratio = present_count / float(len(keys_to_check))
    completeness_score = int(round(w_complete * completeness_ratio))

    total = max(0, min(100, keyword_score + format_score + completeness_score))
    breakdown = {
        "keyword_presence": keyword_score,
        "formatting": format_score,
        "section_completeness": completeness_score,
    }
    return total, breakdown


def compute_resume_score_breakdown(
    sections: Dict[str, str],
    resume_skills: Iterable[str],
    role_skills: Iterable[str],
    role_keywords: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """
    Compute a breakdown that sums to an overall score out of 100.
    The breakdown is returned as:
      skills_score, experience_score, projects_score, overall_score
    """
    resume_skills_set = {s.lower() for s in resume_skills}
    role_skills_list = list(role_skills)
    role_skills_set = {s.lower() for s in role_skills_list}

    weights = {
        "skills": 40,
        "keywords": 20,
        "projects": 15,
        "experience": 15,
        "completeness": 10,
    }

    skill_match_ratio = 0.0
    if role_skills_set:
        matched = resume_skills_set & role_skills_set
        skill_match_ratio = len(matched) / float(len(role_skills_set))
    skills_score = int(round(skill_match_ratio * weights["skills"]))

    raw_text = (sections.get("raw_text") or "").lower()
    keywords = [k.lower() for k in (role_keywords or []) if k]
    keyword_ratio = 0.0
    if keywords:
        hits = sum(1 for kw in keywords if kw in raw_text)
        keyword_ratio = hits / float(len(keywords))
    else:
        keyword_ratio = skill_match_ratio
    keywords_score = int(round(keyword_ratio * weights["keywords"]))

    project_words = _section_word_count(sections, "projects")
    if project_words == 0:
        projects_score = 0
    elif project_words >= 60:
        projects_score = weights["projects"]
    else:
        projects_score = int(round(weights["projects"] * (project_words / 60.0)))

    experience_words = _section_word_count(sections, "experience")
    if experience_words == 0:
        experience_score = 0
    elif experience_words >= 80:
        experience_score = weights["experience"]
    else:
        experience_score = int(round(weights["experience"] * (experience_words / 80.0)))

    keys_to_check = ["name", "skills", "education", "projects", "experience"]
    present_count = sum(1 for k in keys_to_check if (sections.get(k) or "").strip())
    completeness_ratio = present_count / float(len(keys_to_check))
    completeness_score = int(round(completeness_ratio * weights["completeness"]))

    overall = skills_score + keywords_score + projects_score + experience_score + completeness_score
    overall = max(0, min(100, int(overall)))

    return {
        "skills_score": max(0, min(weights["skills"], skills_score)),
        "experience_score": max(0, min(weights["experience"], experience_score)),
        "projects_score": max(0, min(weights["projects"], projects_score)),
        # include keyword/completeness internally; UI can stay simple
        "overall_score": overall,
    }


def compute_resume_score(
    sections: Dict[str, str],
    resume_skills: Iterable[str],
    role_skills: Iterable[str],
    role_keywords: Optional[Iterable[str]] = None,
) -> int:
    """
    Compute an overall resume score out of 100 based on:
    - skill relevance
    - keyword presence
    - project section depth
    - experience section depth
    - overall resume completeness
    """
    breakdown = compute_resume_score_breakdown(
        sections=sections,
        resume_skills=resume_skills,
        role_skills=role_skills,
        role_keywords=role_keywords,
    )
    return breakdown["overall_score"]


def generate_suggestions(
    sections: Dict[str, str],
    resume_skills: Iterable[str],
    missing_skills: Iterable[str],
    role: str,
    role_keywords: Optional[Iterable[str]] = None,
    job_description: str = "",
    jd_missing_keywords: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Generate human-readable suggestions for improving the resume based on
    extracted sections, skills, and missing skills.
    """
    suggestions: List[str] = []
    raw_text = (sections.get("raw_text") or "").lower()
    jd_missing = [k for k in (jd_missing_keywords or []) if k]

    if missing_skills:
        suggestions.append(
            "Consider adding or strengthening the following important skills for your target role: "
            + ", ".join(sorted(missing_skills))
            + "."
        )

    if jd_missing:
        suggestions.append(
            "Tailor your resume to the job description by naturally incorporating these missing terms (only if you truly have the experience): "
            + ", ".join(_unique_preserve_order(jd_missing)[:12])
            + "."
        )

    if "github.com" not in raw_text and "gitlab.com" not in raw_text and "bitbucket.org" not in raw_text:
        suggestions.append(
            "Include a link to your GitHub or code repository to showcase projects and contributions."
        )

    if "linkedin.com" not in raw_text and "portfolio" not in raw_text and "behance.net" not in raw_text:
        suggestions.append(
            "Add a LinkedIn or portfolio link so recruiters can quickly learn more about you."
        )

    experience_text = sections.get("experience") or ""
    if "%" not in experience_text and "increased" not in experience_text.lower() and "reduced" not in experience_text.lower():
        suggestions.append(
            "Add measurable achievements in your experience section (e.g., 'increased performance by 20%' or "
            "'reduced processing time by 30%')."
        )
    if _section_word_count(sections, "experience") > 0 and not _contains_any(
        experience_text.lower(), ["led", "owned", "shipped", "delivered", "launched", "optimized", "implemented"]
    ):
        suggestions.append(
            "Make experience bullets more impact-driven: start with strong action verbs (e.g., 'Led', 'Owned', 'Shipped', 'Optimized') and include scope + outcome."
        )

    projects_text = sections.get("projects") or ""
    if _section_word_count(sections, "projects") < 40:
        suggestions.append(
            "Expand your project descriptions with more detail about technologies used, your role, and impact."
        )
    if projects_text and not _contains_any(projects_text.lower(), ["api", "performance", "latency", "users", "scale", "testing", "ci/cd", "deployment"]):
        suggestions.append(
            "Improve project descriptions by adding 1–2 specifics per project: tech stack, what you built, and a measurable result (users, latency, cost, reliability)."
        )

    if _section_word_count(sections, "education") == 0:
        suggestions.append(
            "Include an education section with your degree(s), institution(s), and graduation year(s)."
        )

    if "certification" not in raw_text and "certificate" not in raw_text:
        suggestions.append(
            f"Consider adding relevant certifications to strengthen your profile as a {role}."
        )

    if not resume_skills:
        suggestions.append(
            "Add a dedicated skills section that clearly lists your technical and soft skills."
        )

    # ATS formatting suggestions
    if "•" not in raw_text and "–" not in raw_text and "-" not in raw_text:
        suggestions.append(
            "Use bullet points (• or -) for experience and projects to improve ATS readability."
        )

    if "\n\n" not in raw_text or raw_text.count("\n\n") < 3:
        suggestions.append(
            "Improve formatting for ATS: add clear section breaks and avoid complex layouts."
        )

    if len(raw_text) < 300:
        suggestions.append(
            "Your resume may be too brief. Expand sections with relevant details for better ATS matching."
        )

    # Improve keyword usage
    keywords_list = list(role_keywords or [])
    if keywords_list:
        missing_kw = [kw for kw in keywords_list if kw.lower() not in raw_text]
        if len(missing_kw) > len(keywords_list) // 2:
            suggestions.append(
                "Improve keyword usage: include more role-specific terms from job descriptions."
            )

    # De-duplicate while preserving order.
    seen = set()
    unique_suggestions: List[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    return unique_suggestions

