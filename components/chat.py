from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from utils.llm import stream_complete


STARTER_CHIPS = [
    "How can I improve my resume?",
    "What jobs match my skills?",
    "What skills am I missing?",
    "Review my ATS score",
]


def _system_prompt(resume_data: dict[str, Any]) -> str:
    resume_json = json.dumps(resume_data, indent=2, default=str)
    return (
        "You are CareerAI, a resume coach. The candidate's resume data: "
        f"{resume_json}\nAnswer ONLY career/resume questions. Be concise. Use bullet points."
    )


def _render_message(role: str, content: str) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="careerai-bubble-row {role}">
          <div class="careerai-bubble {role}">{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _append_user_message(content: str) -> None:
    st.session_state["chat_history"].append({"role": "user", "content": content})


def _assistant_response(system_prompt: str) -> str:
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state["chat_history"]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    chunks: list[str] = []
    placeholder = st.empty()
    typing_frames = (".", "..", "...")

    try:
        for index, chunk in enumerate(stream_complete(messages, system_prompt)):
            chunks.append(chunk)
            rendered = html.escape("".join(chunks)).replace("\n", "<br>")
            frame = typing_frames[index % len(typing_frames)]
            placeholder.markdown(
                f"""
                <div class="careerai-bubble-row assistant">
                  <div class="careerai-bubble assistant">{rendered}<span style="opacity:.55;margin-left:6px;">{frame}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as exc:
        chunks = [f"Could not reach the AI service: {exc}"]

    final_text = "".join(chunks).strip()
    placeholder.empty()
    return final_text


def render_chat(resume_data: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="fade-up" style="margin-bottom:1.5rem;">
          <h2 style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:var(--text);margin:0 0 6px;">
            💬 Resume Chat Assistant
          </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    cols = st.columns(len(STARTER_CHIPS))
    for col, chip in zip(cols, STARTER_CHIPS):
        if col.button(chip, use_container_width=True):
            _append_user_message(chip)
            st.rerun()

    st.markdown('<div class="careerai-chat-shell">', unsafe_allow_html=True)
    for message in st.session_state["chat_history"]:
        _render_message(message["role"], message["content"])

    if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
        answer = _assistant_response(_system_prompt(resume_data))
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        _render_message("assistant", answer)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("careerai_chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([6, 1])
        prompt = input_col.text_input("Message", label_visibility="collapsed", placeholder="Ask a question about your resume...")
        submitted = send_col.form_submit_button("Send", use_container_width=True)
        if submitted and prompt.strip():
            _append_user_message(prompt.strip())
            st.rerun()

    if st.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()
