from __future__ import annotations

from typing import Any

import streamlit as st


def _as_url(value: str | None, scheme: str = "https://") -> str | None:
    if not value:
        return None
    value = value.strip()
    if value.startswith(("http://", "https://", "mailto:", "tel:")):
        return value
    return f"{scheme}{value}"


def _link_line(label: str, value: str | None, kind: str = "url") -> str:
    if not value:
        return f"- **{label}:** ❌"
    if kind == "email":
        href = _as_url(value, "mailto:")
    elif kind == "phone":
        href = _as_url(value, "tel:")
    else:
        href = _as_url(value)
    return f"- **{label}:** [{value}]({href})"


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
    cert_deduction = 0
    final_score = min(keyword_match + skill_coverage + formatting - cert_deduction + certs_bonus, 100)

    return {
        "score": final_score,
        "keyword_match": keyword_match,
        "skill_coverage": skill_coverage,
        "formatting": formatting,
        "certs_bonus": certs_bonus,
        "cert_deduction": cert_deduction,
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
        """
        <div class="fade-up" style="margin-bottom:1.5rem;">
          <h2 style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:var(--text);margin:0 0 6px;">
            ATS Report
          </h2>
          <p style="font-size:13px;color:rgba(128,128,128,0.8);">
            Clickable contact links, certification signals, and a transparent scoring breakdown.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_score, col_links = st.columns([1, 1.25])

    with col_score:
        st.markdown(f"### Final ATS Score: **{score['score']:.1f}/100**")
        st.write(f"- **Keyword Match:** {score['keyword_match']:.1f}/40 ({score['keywords_found']} found)")
        st.write(f"- **Skill Coverage:** {score['skill_coverage']:.1f}/30 ({score['skills_found']} skills)")
        st.write(f"- **Formatting:** {score['formatting']:.1f}/20 ({score['checks_passed']}/{score['total_checks']} checks)")
        st.write(f"- **Certificate Bonus:** +{score['certs_bonus']:.1f}/15 ({score['certs']} found)")

    with col_links:
        st.markdown("### Clickable Links Extracted")
        st.markdown(
            "\n".join(
                [
                    _link_line("Email", resume_data.get("email"), "email"),
                    _link_line("Phone", resume_data.get("phone"), "phone"),
                    _link_line("LinkedIn", resume_data.get("linkedin")),
                    _link_line("GitHub", resume_data.get("github")),
                    _link_line("Portfolio", resume_data.get("portfolio")),
                ]
            )
        )

    col_checks, col_certs = st.columns(2)

    with col_checks:
        st.markdown("### ATS Checklist")
        for label, passed in score["checks"]:
            st.write(f"{'✅' if passed else '❌'} {label}")

    with col_certs:
        st.markdown("### Certificates")
        if certifications:
            for index, cert in enumerate(certifications, 1):
                st.write(f"{index}. {cert}")
        else:
            st.write("No certificate lines found.")
