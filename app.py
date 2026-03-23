"""
SkillLens AI - AI-powered resume intelligence for smarter careers.
"""
import io
import os
from datetime import datetime
from typing import Dict, List, Any

from flask import Flask, render_template, request, redirect, url_for, flash, Response, session

from skills_database import get_job_skills, get_roles, get_role_skills, get_role_keywords
from resume_parser import extract_text_from_resume, parse_resume_sections
from utils.skill_extractor import (
    extract_resume_skills,
    get_missing_skills,
    get_keywords_found_in_resume,
)
from utils.analyzer import (
    compute_ats_score,
    compute_job_description_match,
    compute_resume_score,
    compute_resume_score_breakdown,
    extract_important_keywords,
    generate_suggestions,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def create_app() -> Flask:
    """Application factory for SkillLens AI."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    def allowed_file(filename: str) -> bool:
        _, ext = os.path.splitext(filename.lower())
        return ext in ALLOWED_EXTENSIONS

    @app.route("/", methods=["GET"])
    def index():
        roles = get_roles()
        return render_template("index.html", roles=roles)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        uploaded_file = request.files.get("resume")
        selected_role = request.form.get("job_role") or "Universal"
        job_description = (request.form.get("job_description") or "").strip()

        # Validation: No file
        if not uploaded_file or uploaded_file.filename == "":
            flash("Please upload a resume file (PDF or DOCX).", "error")
            return redirect(url_for("index"))

        # Validation: Unsupported file type
        if not allowed_file(uploaded_file.filename):
            flash("Unsupported file type. Please upload a PDF or DOCX file.", "error")
            return redirect(url_for("index"))

        # Validation: Invalid role
        job_skills = get_job_skills()
        if selected_role not in job_skills:
            selected_role = "Universal"

        try:
            file_bytes = uploaded_file.read()
        except Exception as exc:
            flash(f"Could not read the uploaded file: {exc}", "error")
            return redirect(url_for("index"))

        # Validation: Empty file
        if not file_bytes:
            flash("Uploaded file appears to be empty.", "error")
            return redirect(url_for("index"))

        # Validation: File size (handled by MAX_CONTENT_LENGTH, but double-check)
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            flash(f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.", "error")
            return redirect(url_for("index"))

        try:
            _, ext = os.path.splitext(uploaded_file.filename.lower())
            resume_text: str = extract_text_from_resume(file_bytes, ext)
        except Exception as exc:
            flash(f"Could not parse the resume. Please ensure it's a valid PDF or DOCX: {exc}", "error")
            return redirect(url_for("index"))

        # Validation: No extractable text
        if not resume_text or not resume_text.strip():
            flash("No readable text could be extracted from the resume. Try a different file.", "error")
            return redirect(url_for("index"))

        # Parse and analyze
        sections = parse_resume_sections(resume_text)
        required_skills = get_role_skills(selected_role)
        role_keywords = get_role_keywords(selected_role)

        extracted_skills = extract_resume_skills(resume_text, job_skills)
        missing_skills = get_missing_skills(required_skills, extracted_skills)

        score = compute_resume_score(
            sections=sections,
            resume_skills=extracted_skills,
            role_skills=required_skills,
            role_keywords=role_keywords,
        )

        score_breakdown = compute_resume_score_breakdown(
            sections=sections,
            resume_skills=extracted_skills,
            role_skills=required_skills,
            role_keywords=role_keywords,
        )

        ats_score, ats_breakdown = compute_ats_score(
            sections=sections,
            resume_skills=extracted_skills,
            role_skills=required_skills,
            role_keywords=role_keywords,
        )

        jd_match = compute_job_description_match(
            resume_text=resume_text,
            job_description=job_description,
            role_skills=required_skills,
            role_keywords=role_keywords,
        )

        suggestions = generate_suggestions(
            sections=sections,
            resume_skills=extracted_skills,
            missing_skills=missing_skills,
            role=selected_role,
            role_keywords=role_keywords,
            job_description=job_description,
            jd_missing_keywords=jd_match.missing_keywords if job_description else [],
        )

        # Skill match percentage
        skill_match_pct = (
            round(100 * len(extracted_skills) / len(required_skills))
            if required_skills
            else 100
        )
        skill_match_pct = min(100, skill_match_pct)

        # Keywords found for highlighting
        skills_found, keywords_found = get_keywords_found_in_resume(
            resume_text, required_skills, role_keywords
        )

        # Keyword analysis (resume-only + JD-aware)
        resume_keywords = extract_important_keywords(resume_text, top_n=18)
        resume_keywords_matched = (
            [k for k in resume_keywords if (k or "").lower() in resume_text.lower()]
            if resume_keywords
            else []
        )

        candidate_name = sections.get("name") or "Your Resume"

        # Store in session for report download
        session["last_analysis"] = {
            "name": candidate_name,
            "role": selected_role,
            "score": score,
            "ats_score": ats_score,
            "score_breakdown": score_breakdown,
            "ats_breakdown": ats_breakdown,
            "jd_match_pct": jd_match.match_pct if job_description else 0,
            "jd_missing_keywords": jd_match.missing_keywords if job_description else [],
            "skill_match_pct": skill_match_pct,
            "extracted_skills": extracted_skills,
            "missing_skills": missing_skills,
            "suggestions": suggestions,
        }

        return render_template(
            "results.html",
            name=candidate_name,
            role=selected_role,
            score=score,
            ats_score=ats_score,
            score_breakdown=score_breakdown,
            ats_breakdown=ats_breakdown,
            skill_match_pct=skill_match_pct,
            extracted_skills=extracted_skills,
            missing_skills=missing_skills,
            suggestions=suggestions,
            skills_found=skills_found,
            keywords_found=keywords_found,
            resume_keywords=resume_keywords,
            jd_text=job_description,
            jd_match_pct=jd_match.match_pct if job_description else None,
            jd_missing_keywords=jd_match.missing_keywords if job_description else [],
            jd_matched_keywords=jd_match.matched_keywords if job_description else [],
            raw_text=resume_text[:2000],  # For keyword highlight preview
        )

    @app.route("/download-report", methods=["GET"])
    def download_report():
        """Generate and download analysis report as HTML."""
        data = session.get("last_analysis")
        if not data:
            flash("No analysis available. Please analyze a resume first.", "error")
            return redirect(url_for("index"))

        html = render_template(
            "report_template.html",
            name=data.get("name", "Resume"),
            role=data.get("role", ""),
            score=data.get("score", 0),
            ats_score=data.get("ats_score", data.get("score", 0)),
            score_breakdown=data.get("score_breakdown", {}),
            jd_match_pct=data.get("jd_match_pct", 0),
            skill_match_pct=data.get("skill_match_pct", 0),
            extracted_skills=data.get("extracted_skills", []),
            missing_skills=data.get("missing_skills", []),
            suggestions=data.get("suggestions", []),
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

        filename = f"resume_analysis_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
        return Response(
            html,
            mimetype="text/html",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.errorhandler(413)
    def request_entity_too_large(error):
        flash(f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.", "error")
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(debug=True)
