"""
Skills database for each target role.
Loads and provides role-specific skills and keywords for resume analysis.
"""
import os
from typing import Dict, List, Any

from utils.skill_extractor import load_job_skills

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JOB_SKILLS_PATH = os.path.join(DATA_DIR, "job_skills.json")

# Cached job skills - loaded once at module import
_JOB_SKILLS: Dict[str, Dict[str, Any]] | None = None


def get_job_skills() -> Dict[str, Dict[str, Any]]:
    """Load and return job skills configuration (cached)."""
    global _JOB_SKILLS
    if _JOB_SKILLS is None:
        _JOB_SKILLS = load_job_skills(JOB_SKILLS_PATH)
    return _JOB_SKILLS


def get_roles() -> List[str]:
    """Return sorted list of available target roles."""
    return sorted(get_job_skills().keys())


def get_role_skills(role: str) -> List[str]:
    """Return skills required for a given role."""
    config = get_job_skills().get(role, {})
    return config.get("skills", [])


def get_role_keywords(role: str) -> List[str]:
    """Return keywords for a given role."""
    config = get_job_skills().get(role, {})
    return config.get("keywords", [])
