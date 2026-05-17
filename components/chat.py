from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from utils.llm import active_provider, stream_complete


STARTER_CHIPS = [
    "How can I improve my resume?",
    "What jobs match my skills?",
    "What skills am I missing?",
    "Review my ATS score",
]


PROVIDER_LABELS = {
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
    "anthropic": "Anthropic",
    "none": "No key",
}


def _system_prompt(resume_data: dict[str, Any]) -> str:
    resume_json = json.dumps(resume_data, indent=2, default=str)
    return (
        "You are CareerAI, a resume coach. The candidate's resume data: "
        f"{resume_json}\nAnswer ONLY career/resume questions. Be concise. Use bullet points."
    )


def _render_message(role: str, content: str) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    avatar = "YOU" if role == "user" else "CA"
    st.markdown(
        f"""
        <div class="careerai-bubble-row {role}">
          <div class="careerai-avatar {role}">{avatar}</div>
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
                  <div class="careerai-avatar assistant">CA</div>
                  <div class="careerai-bubble assistant">{rendered}<span class="careerai-typing">{frame}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as exc:
        chunks = [f"{exc}"]

    final_text = "".join(chunks).strip()
    placeholder.empty()
    return final_text


def _metric(label: str, value: Any) -> str:
    return f"""
    <div class="careerai-chat-metric">
      <span>{html.escape(label)}</span>
      <strong>{html.escape(str(value))}</strong>
    </div>
    """


def render_chat(resume_data: dict[str, Any]) -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    provider = active_provider()
    provider_label = PROVIDER_LABELS.get(provider, provider.title())
    skill_count = len(resume_data.get("skills", []) or [])
    gap_count = len(resume_data.get("skill_gaps", []) or [])
    ats_score = resume_data.get("ats_score", "-")
    role = resume_data.get("predicted_role") or "Resume"

    st.markdown(
        f"""
        <section class="careerai-chat-hero">
          <div>
            <div class="careerai-kicker">Resume copilot</div>
            <h2>Chat with CareerAI</h2>
            <p>Ask for sharper bullets, missing skills, job-fit strategy, and ATS fixes based on this resume.</p>
          </div>
          <div class="careerai-provider-pill {provider}">
            <span></span>{html.escape(provider_label)}
          </div>
        </section>
        <div class="careerai-chat-metrics">
          {_metric("Role", role)}
          {_metric("ATS", f"{ats_score}/100" if isinstance(ats_score, (int, float)) else ats_score)}
          {_metric("Skills", skill_count)}
          {_metric("Gaps", gap_count)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="careerai-chip-row">', unsafe_allow_html=True)
    cols = st.columns(len(STARTER_CHIPS))
    for col, chip in zip(cols, STARTER_CHIPS):
        if col.button(chip, use_container_width=True):
            _append_user_message(chip)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="careerai-chat-shell">', unsafe_allow_html=True)
    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div class="careerai-empty-chat">
              <div class="careerai-empty-orbit">CA</div>
              <h3>Ready when you are</h3>
              <p>Start with a suggested prompt, or ask for a rewrite, role plan, skill roadmap, or ATS diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state["chat_history"]:
        _render_message(message["role"], message["content"])

    if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
        answer = _assistant_response(_system_prompt(resume_data))
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        _render_message("assistant", answer)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("careerai_chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([7, 1.2])
        prompt = input_col.text_input(
            "Message",
            label_visibility="collapsed",
            placeholder="Ask for resume rewrites, job matches, missing skills, or ATS fixes...",
        )
        submitted = send_col.form_submit_button("Send", use_container_width=True)
        if submitted and prompt.strip():
            _append_user_message(prompt.strip())
            st.rerun()

    if st.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()
