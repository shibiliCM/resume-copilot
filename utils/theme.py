from __future__ import annotations

from pathlib import Path
from typing import Literal

import streamlit as st


ThemeName = Literal["dark", "light"]


def get_css() -> str:
    css_path = Path("styles/theme.css")
    if not css_path.exists():
        return ""
    return css_path.read_text(encoding="utf-8")


def inject_theme(theme: ThemeName) -> None:
    resolved_theme: ThemeName = "light" if theme == "light" else "dark"
    variable_override = (
        """
        :root {
          --bg: #f5f7fa;
          --card: #fff;
          --text: #1a1a2e;
          --accent: #00c9a7;
        }
        """
        if resolved_theme == "light"
        else ""
    )
    st.markdown(
        f"<style>{get_css()}{variable_override}</style>",
        unsafe_allow_html=True,
    )
