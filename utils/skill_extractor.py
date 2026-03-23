"""
Skill extraction using spaCy NLP, TF-IDF, and substring matching.
"""
import json
import re
from typing import Dict, List, Iterable, Any, Set, Tuple

import pandas as pd

try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = None
except ImportError:
    _NLP = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]


def load_job_skills(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load job/role skills configuration from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    records = []
    for role, cfg in data.items():
        for skill in cfg.get("skills", []):
            records.append({"role": role, "skill": skill})
    _ = pd.DataFrame.from_records(records, columns=["role", "skill"])
    return data


def _collect_all_skills(job_skills: Dict[str, Dict[str, Any]]) -> List[str]:
    skills: List[str] = []
    for cfg in job_skills.values():
        for skill in cfg.get("skills", []):
            if skill not in skills:
                skills.append(skill)
    return skills


def _normalize_skill(skill: str) -> str:
    return skill.strip().lower()


def _extract_skills_with_spacy(resume_text: str, all_skills: List[str]) -> List[str]:
    """
    Use spaCy NLP to extract skills: noun chunks, entities (PRODUCT, ORG),
    and pattern-based extraction (e.g., "proficient in X", "experience with X").
    """
    if not _NLP or not resume_text.strip():
        return []

    doc = _NLP(resume_text[:15000])  # Limit for performance
    matched: Set[str] = set()
    resume_lower = resume_text.lower()

    # 1. Match known skills from noun chunks and entities
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if len(chunk_text) < 2 or len(chunk_text) > 50:
            continue
        for skill in all_skills:
            if _normalize_skill(skill) in chunk_text.lower() or chunk_text.lower() in _normalize_skill(skill):
                matched.add(skill)

    # 2. PRODUCT entities often include tech (e.g., "Python", "React")
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG", "GPE"):
            ent_lower = ent.text.lower()
            for skill in all_skills:
                if _normalize_skill(skill) == ent_lower or _normalize_skill(skill) in ent_lower:
                    matched.add(skill)

    # 3. Pattern-based: "skills: X, Y, Z" or "proficient in X" or "experience with X"
    patterns = [
        r"(?:proficient|experienced|skilled|expert)\s+(?:in|with)\s+([A-Za-z0-9\.\#\+\-\s]+?)(?:\.|,|$)",
        r"(?:knowledge|experience)\s+(?:of|in|with)\s+([A-Za-z0-9\.\#\+\-\s]+?)(?:\.|,|$)",
        r"(?:technologies?|tools?|languages?)\s*:?\s*([A-Za-z0-9\.\#\+\-\s,]+?)(?:\n|$)",
        r"skills?\s*:?\s*([A-Za-z0-9\.\#\+\-\s,]+?)(?:\n\n|\n[A-Z]|$)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, resume_text, re.IGNORECASE):
            candidates = re.split(r"[,;|/]", m.group(1))
            for c in candidates:
                c = c.strip()
                if 2 <= len(c) <= 40:
                    for skill in all_skills:
                        if _normalize_skill(skill) == _normalize_skill(c) or _normalize_skill(skill) in _normalize_skill(c):
                            matched.add(skill)

    return sorted(matched, key=lambda s: s.lower())


def extract_resume_skills(resume_text: str, job_skills: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Extract skills from resume using spaCy NLP, TF-IDF similarity, and substring matching.
    """
    resume_text = resume_text or ""
    resume_text_lower = resume_text.lower()

    all_skills = _collect_all_skills(job_skills)
    if not resume_text_lower.strip() or not all_skills:
        return []

    matched: Set[str] = set()

    # 1. Direct substring matches (case-insensitive)
    for skill in all_skills:
        if _normalize_skill(skill) and _normalize_skill(skill) in resume_text_lower:
            matched.add(skill)

    # 2. spaCy NLP extraction
    if _NLP:
        spacy_matched = _extract_skills_with_spacy(resume_text, all_skills)
        matched.update(spacy_matched)

    # 3. TF-IDF + cosine similarity for semantic variations
    if TfidfVectorizer is not None and cosine_similarity is not None:
        corpus: List[str] = [resume_text] + all_skills
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        resume_vec = tfidf_matrix[0:1]
        skill_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(resume_vec, skill_vecs)[0]
        for skill, score in zip(all_skills, similarities):
            if score >= 0.15 and skill not in matched:
                matched.add(skill)

    return sorted(matched, key=lambda s: s.lower())


def get_missing_skills(required_skills: Iterable[str], resume_skills: Iterable[str]) -> List[str]:
    """Determine which required skills are not present in the resume."""
    resume_set = {_normalize_skill(s) for s in resume_skills}
    missing = [s for s in required_skills if _normalize_skill(s) not in resume_set]
    return sorted(missing, key=lambda s: s.lower())


def get_keywords_found_in_resume(
    resume_text: str,
    role_skills: Iterable[str],
    role_keywords: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """
    Return (skills_found, keywords_found) for highlighting in the resume text.
    """
    resume_lower = (resume_text or "").lower()
    skills_found: List[str] = []
    keywords_found: List[str] = []

    for skill in role_skills:
        if _normalize_skill(skill) in resume_lower:
            skills_found.append(skill)

    for kw in role_keywords:
        if kw and kw.lower() in resume_lower:
            keywords_found.append(kw)

    return (skills_found, keywords_found)
