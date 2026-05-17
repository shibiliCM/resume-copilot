from __future__ import annotations

import re
from io import BytesIO
from typing import Any, BinaryIO

import pdfplumber


CERT_KEYWORDS = (
    "certificate",
    "certified",
    "certification",
    "credential",
    "course",
    "bootcamp",
    "nanodegree",
)

ACTION_KEYWORDS = (
    "developed",
    "designed",
    "managed",
    "implemented",
    "improved",
    "optimized",
    "built",
    "led",
    "launched",
    "delivered",
    "reduced",
    "increased",
    "scaled",
    "collaborated",
    "architected",
    "deployed",
    "automated",
    "created",
    "achieved",
)

COMMON_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
    "live.com",
}


def _read_pdf_text(pdf_bytes: bytes | BinaryIO) -> str:
    text_parts: list[str] = []
    source: bytes | BinaryIO
    source = BytesIO(pdf_bytes) if isinstance(pdf_bytes, (bytes, bytearray)) else pdf_bytes
    if hasattr(source, "seek"):
        source.seek(0)

    try:
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
    finally:
        if hasattr(source, "seek"):
            source.seek(0)

    return "\n".join(text_parts)


def _first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(0).strip() if match else None


def _extract_name(text: str) -> str:
    for line in text.splitlines()[:8]:
        cleaned = re.sub(r"\s+", " ", line).strip(" -|")
        if not cleaned:
            continue
        if re.search(r"@|linkedin|github|resume|curriculum|vitae|\+?\d", cleaned, re.IGNORECASE):
            continue
        words = cleaned.split()
        if 1 <= len(words) <= 4 and all(re.search(r"[A-Za-z]", word) for word in words):
            return cleaned
    return ""


def _extract_certifications(text: str) -> list[str]:
    certifications: list[str] = []
    seen: set[str] = set()
    lines = [re.sub(r"\s+", " ", line).strip(" -•\t") for line in text.splitlines()]

    for index, line in enumerate(lines):
        if not line:
            continue
        lower_line = line.lower()
        if not any(keyword in lower_line for keyword in CERT_KEYWORDS):
            continue

        window = [line]
        for offset in (1, 2):
            if index + offset < len(lines):
                nearby = lines[index + offset]
                if 5 <= len(nearby) <= 140 and not re.match(r"^[A-Z][A-Z\s]{3,}$", nearby):
                    window.append(nearby)
        candidate = " - ".join(window)
        candidate = candidate[:220].strip()
        key = candidate.lower()
        if len(candidate) > 5 and key not in seen:
            seen.add(key)
            certifications.append(candidate)

    return certifications


def _extract_education(text: str) -> list[str]:
    education_terms = r"bachelor|master|phd|doctorate|degree|university|college|institute|school|b\.?tech|m\.?tech|bsc|msc|mba"
    education: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if 6 <= len(cleaned) <= 160 and re.search(education_terms, cleaned, re.IGNORECASE):
            if cleaned not in education:
                education.append(cleaned)
    return education[:8]


def _extract_experience_years(text: str) -> float:
    patterns = (
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience",
        r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    )
    values: list[float] = []
    for pattern in patterns:
        values.extend(float(match) for match in re.findall(pattern, text, re.IGNORECASE))
    return max(values) if values else 0.0


def _extract_portfolio(text: str, email: str | None) -> str | None:
    email_domain = email.split("@", 1)[1].lower() if email and "@" in email else ""
    candidates = re.findall(
        r"(?:https?://)?(?:www\.)?(?!(?:linkedin|github)\.com\b)[A-Za-z0-9][A-Za-z0-9_-]*\.(?:com|net|org|io|dev|me|app)(?:/[^\s)\]]*)?",
        text,
        re.IGNORECASE,
    )
    for candidate in candidates:
        cleaned = candidate.strip(" .,:;)")
        domain = re.sub(r"^https?://", "", cleaned, flags=re.IGNORECASE).removeprefix("www.").split("/", 1)[0].lower()
        if domain == email_domain or domain in COMMON_EMAIL_DOMAINS:
            continue
        return cleaned
    return None


def parse_resume(pdf_bytes: bytes | BinaryIO) -> dict[str, Any]:
    text = _read_pdf_text(pdf_bytes)
    text_lower = text.lower()

    email = _first_match(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text)
    phone = _first_match(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,5}\d{2,4}", text)
    linkedin = _first_match(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_%\-./]+", text)
    github = _first_match(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.%-]+", text)
    portfolio = _extract_portfolio(text, email)

    sections = sum(
        1
        for section in ("experience", "education", "skills", "projects", "summary", "certifications")
        if section in text_lower
    )
    bullet_count = len(re.findall(r"(^|\n)\s*(?:[-*•·]|\d+[.)])\s+", text))
    action_keywords = sum(1 for keyword in ACTION_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b", text_lower))

    return {
        "text": text,
        "name": _extract_name(text),
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "portfolio": portfolio,
        "skills": [],
        "experience_years": _extract_experience_years(text),
        "education": _extract_education(text),
        "certifications": _extract_certifications(text),
        "sections": sections,
        "word_count": len(re.findall(r"\b\w+\b", text)),
        "bullet_count": bullet_count,
        "action_keywords": action_keywords,
    }
