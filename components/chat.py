from __future__ import annotations

import html
import json
import re
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


def _markdown_to_html(text: str) -> str:
    # 1. Escape HTML first to prevent XSS
    escaped = html.escape(text)
    
    # 2. Parse bold: **text** -> <strong>text</strong>
    escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
    
    # 3. Parse code: `code` -> <code>code</code>
    escaped = re.sub(r"`(.*?)`", r"<code>\1</code>", escaped)
    
    # 4. Parse lists line by line
    lines = escaped.splitlines()
    in_list = False
    html_lines = []
    
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(?:\*|-|•)\s+(.*)", stripped)
        if match:
            if not in_list:
                html_lines.append('<ul style="margin: 4px 0; padding-left: 20px;">')
                in_list = True
            html_lines.append(f'<li style="margin: 4px 0;">{match.group(1)}</li>')
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(line)
            
    if in_list:
        html_lines.append("</ul>")
        
    result_lines = []
    for line in html_lines:
        if line.startswith("<ul") or line.startswith("<li") or line.endswith("</ul>") or line.endswith("</li>"):
            result_lines.append(line)
        else:
            result_lines.append(line + "<br>")
            
    result = "".join(result_lines)
    result = re.sub(r"(?:<br>)+$", "", result)
    return result


def _get_message_html(role: str, content: str) -> str:
    parsed_content = _markdown_to_html(content)
    avatar = "YOU" if role == "user" else "CA"
    return (
        f'<div class="careerai-bubble-row {role}">'
        f'<div class="careerai-avatar {role}">{avatar}</div>'
        f'<div class="careerai-bubble {role}">{parsed_content}</div>'
        f'</div>'
    )


def _get_history_html(history: list[dict[str, str]]) -> str:
    if not history:
        return (
            '<div class="careerai-empty-chat">'
            '<div class="careerai-empty-orbit">CA</div>'
            '<h3>Ready when you are</h3>'
            '<p>Start with a suggested prompt, or ask for a rewrite, role plan, skill roadmap, or ATS diagnosis.</p>'
            '</div>'
        )
    return "".join(_get_message_html(msg["role"], msg["content"]) for msg in history)


def _append_user_message(content: str) -> None:
    st.session_state["chat_history"].append({"role": "user", "content": content})


def _metric(label: str, value: Any) -> str:
    return f'<div class="careerai-chat-metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def render_chat(resume_data: dict[str, Any]) -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "chat_view_mode" not in st.session_state:
        st.session_state["chat_view_mode"] = "Full Screen"

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
        """,
        unsafe_allow_html=True,
    )

    metrics_html = "".join([
        _metric("Role", role),
        _metric("ATS", f"{ats_score}/100" if isinstance(ats_score, (int, float)) else ats_score),
        _metric("Skills", skill_count),
        _metric("Gaps", gap_count),
    ])
    st.markdown(
        f'<div class="careerai-chat-metrics">{metrics_html}</div>',
        unsafe_allow_html=True,
    )

    shell_height_style = 'style="height: 68vh; min-height: 520px; max-height: 800px;"'

    st.markdown('<div class="careerai-chip-row">', unsafe_allow_html=True)
    cols = st.columns(len(STARTER_CHIPS))
    for col, chip in zip(cols, STARTER_CHIPS):
        if col.button(chip, use_container_width=True):
            _append_user_message(chip)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Build history HTML
    history_html = _get_history_html(st.session_state["chat_history"])

    # If the last message is from the user, stream the assistant's response in place
    if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
        # Draw a beautiful loading bubble with animated typing dots inside the shell container
        loading_bubble = (
            f'<div class="careerai-bubble-row assistant">'
            f'<div class="careerai-avatar assistant">CA</div>'
            f'<div class="careerai-bubble assistant">'
            f'<div class="typing-dots"><i></i><i></i><i></i></div>'
            f'</div>'
            f'</div>'
        )
        
        # Place a markdown block placeholder where the shell usually is
        shell_placeholder = st.empty()
        shell_placeholder.markdown(
            f'<div class="careerai-chat-shell" {shell_height_style}>{history_html}{loading_bubble}</div>',
            unsafe_allow_html=True,
        )
        
        # Prepare system prompt and messages
        system_prompt = _system_prompt(resume_data)
        messages = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state["chat_history"]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        
        # Stream response
        chunks: list[str] = []
        typing_frames = (".", "..", "...")
        try:
            for index, chunk in enumerate(stream_complete(messages, system_prompt)):
                chunks.append(chunk)
                rendered = _markdown_to_html("".join(chunks))
                frame = typing_frames[index % len(typing_frames)]
                current_bubble = (
                    f'<div class="careerai-bubble-row assistant">'
                    f'<div class="careerai-avatar assistant">CA</div>'
                    f'<div class="careerai-bubble assistant">{rendered}<span class="careerai-typing">{frame}</span></div>'
                    f'</div>'
                )
                shell_placeholder.markdown(
                    f'<div class="careerai-chat-shell" {shell_height_style}>{history_html}{current_bubble}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            chunks = [f"{exc}"]
            
        answer = "".join(chunks).strip()
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()
    else:
        # Standard display of the shell container
        st.markdown(
            f'<div class="careerai-chat-shell" {shell_height_style}>{history_html}</div>',
            unsafe_allow_html=True,
        )

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
