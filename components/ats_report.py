from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st


def _as_url(value: str | None, kind: str = "url") -> str | None:
    if not value:
        return None
    value = value.strip()
    if kind == "email":
        return f"mailto:{value}"
    if kind == "phone":
        digits = re.sub(r"[^\d+]", "", value)
        return f"tel:{digits}" if digits else None
    if value.startswith(("http://", "https://")):
        return value
    return f"https://{value}"


def _link_card(label: str, value: str | None, kind: str = "url") -> str:
    href = _as_url(value, kind)
    if not value or not href:
        return f'<div class="careerai-link-card missing"><span>{html.escape(label)}</span><strong>Missing</strong></div>'
    safe_value = html.escape(value)
    safe_href = html.escape(href, quote=True)
    return f'<a class="careerai-link-card" href="{safe_href}" target="_blank" rel="noopener noreferrer"><span>{html.escape(label)}</span><strong>{safe_value}</strong></a>'



def calculate_ats_score(resume_data: dict[str, Any], tech_skills: list[str]) -> dict[str, Any]:
    keywords_found = int(resume_data.get("action_keywords", resume_data.get("kw_hits", 0)) or 0)
    skills_found = len(tech_skills)
    certs = len(resume_data.get("certifications", []) or [])

    checks = [
        ("Email", bool(resume_data.get("email"))),
        ("Phone", bool(resume_data.get("phone"))),
        ("LinkedIn", bool(resume_data.get("linkedin"))),
        ("Portfolio/GitHub", bool(resume_data.get("github")) or bool(resume_data.get("portfolio"))),
        ("Length (300-1000 words)", 300 <= int(resume_data.get("word_count", 0) or 0) <= 1000),
        ("Sections (3+)", int(resume_data.get("sections", 0) or 0) >= 3),
        ("Bullets (5+)", int(resume_data.get("bullet_count", resume_data.get("bullets", 0)) or 0) >= 5),
    ]
    checks_passed = sum(1 for _, passed in checks if passed)
    total_checks = len(checks)

    keyword_match = min(keywords_found / 10, 1) * 40
    skill_coverage = min(skills_found / 15, 1) * 30
    formatting = (checks_passed / total_checks) * 20 if total_checks else 0
    certs_bonus = min(certs * 5, 15)
    final_score = min(keyword_match + skill_coverage + formatting + certs_bonus, 100)

    return {
        "score": final_score,
        "keyword_match": keyword_match,
        "skill_coverage": skill_coverage,
        "formatting": formatting,
        "certs_bonus": certs_bonus,
        "checks": checks,
        "checks_passed": checks_passed,
        "total_checks": total_checks,
        "keywords_found": keywords_found,
        "skills_found": skills_found,
        "certs": certs,
    }


def render_ats_report(resume_data: dict[str, Any], tech_skills: list[str]) -> None:
    score = calculate_ats_score(resume_data, tech_skills)
    certifications = resume_data.get("certifications", []) or []

    st.markdown(
        f"""
        <section class="careerai-report-hero">
          <div>
            <div class="careerai-kicker">ATS diagnostics</div>
            <h2>ATS Report</h2>
            <p>Clickable contact links, certification signals, and a transparent scoring breakdown.</p>
          </div>
          <div class="careerai-score-orb">
            <strong>{score['score']:.1f}</strong>
            <span>/100</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col_score, col_links = st.columns([1, 1.25])

    with col_score:
        st.markdown(
            f"""
            <div class="careerai-panel">
              <h3>Score Breakdown</h3>
              <div class="careerai-breakdown-row"><span>Keyword Match</span><strong>{score['keyword_match']:.1f}/40</strong></div>
              <div class="careerai-breakdown-row"><span>Skill Coverage</span><strong>{score['skill_coverage']:.1f}/30</strong></div>
              <div class="careerai-breakdown-row"><span>Formatting</span><strong>{score['formatting']:.1f}/20</strong></div>
              <div class="careerai-breakdown-row"><span>Certificate Bonus</span><strong>+{score['certs_bonus']:.1f}/15</strong></div>
              <p>{score['keywords_found']} action keywords, {score['skills_found']} skills, {score['checks_passed']}/{score['total_checks']} checks passed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_links:
        links_html = "".join([
            _link_card("Email", resume_data.get("email"), "email"),
            _link_card("Phone", resume_data.get("phone"), "phone"),
            _link_card("LinkedIn", resume_data.get("linkedin")),
            _link_card("GitHub", resume_data.get("github")),
            _link_card("Portfolio", resume_data.get("portfolio")),
        ])
        st.markdown(
            f'<div class="careerai-panel"><h3>Clickable Links</h3><div class="careerai-link-grid">{links_html}</div></div>',
            unsafe_allow_html=True,
        )


    col_checks, col_certs = st.columns(2)

    with col_checks:
        rows = "".join(
            f"<div class='careerai-check-row {'ok' if passed else 'missing'}'><span>{'OK' if passed else 'NO'}</span>{html.escape(label)}</div>"
            for label, passed in score["checks"]
        )
        st.markdown(f"<div class='careerai-panel'><h3>ATS Checklist</h3>{rows}</div>", unsafe_allow_html=True)

    with col_certs:
        if certifications:
            cert_rows = "".join(f"<li>{html.escape(cert)}</li>" for cert in certifications)
        else:
            cert_rows = "<li>No certificate lines found.</li>"
        st.markdown(f"<div class='careerai-panel'><h3>Certificates</h3><ol>{cert_rows}</ol></div>", unsafe_allow_html=True)
