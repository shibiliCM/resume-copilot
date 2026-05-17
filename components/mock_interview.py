from __future__ import annotations

import html
import json
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.llm import active_provider, chat_complete


DIMENSIONS = ["Clarity", "Relevance", "Depth", "Communication", "Role-Fit"]


def _resume_json(resume_data: dict[str, Any]) -> str:
    return json.dumps(resume_data, indent=2, default=str)


def _json_from_text(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_positions = [pos for pos in (cleaned.find("["), cleaned.find("{")) if pos != -1]
        if not start_positions:
            raise
        start = min(start_positions)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _fallback_questions(role: str, interview_type: str, resume_data: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
    skills = resume_data.get("skills", []) or []
    gaps = resume_data.get("skill_gaps", []) or []
    skill_text = ", ".join(skills[:5]) or "your current skills"
    gap_text = ", ".join(gaps[:3]) or "the role requirements"
    templates = {
        "Behavioral": [
            f"Tell me about a time you used {skill_text} to solve a difficult problem.",
            f"Describe a project where you had to learn something quickly for a {role} outcome.",
            "Tell me about a time you received critical feedback and changed your approach.",
            "Describe a situation where you had to communicate a technical idea to a non-technical audience.",
            f"Give an example of how you handled competing priorities while preparing for a {role} responsibility.",
        ],
        "Technical": [
            f"Walk me through one resume project and explain the architecture, tradeoffs, and measurable result.",
            f"Which of these skills are strongest for you: {skill_text}? Explain with a concrete example.",
            f"What would you do in the next 30 days to close these gaps: {gap_text}?",
            f"Explain how you would debug a production issue related to your {role} work.",
            "Choose one technical achievement from your resume and explain how you would improve it today.",
        ],
        "HR": [
            f"Why are you interested in this {role} role?",
            "Walk me through your resume in two minutes.",
            "What are your strongest professional strengths, and where is the evidence in your resume?",
            f"What compensation, work style, and growth expectations matter most to you for a {role} position?",
            "Why should we choose you over another candidate with similar experience?",
        ],
    }
    return [
        {"question": question, "ideal_answer_hints": "Use a concise STAR structure, include metrics, and connect the answer to the target role.", "difficulty": "Medium"}
        for question in templates.get(interview_type, templates["Behavioral"])[:n]
    ]


def _generate_questions(resume_data: dict[str, Any], role: str, interview_type: str, n: int = 5) -> tuple[list[dict[str, Any]], str | None]:
    system = (
        f"You are a strict but fair interviewer for {role}.\n"
        f"Candidate resume: {_resume_json(resume_data)}.\n"
        f"Generate {n} {interview_type} interview questions.\n"
        'Return strictly valid JSON only: [{"question": "...", "ideal_answer_hints": "...", "difficulty": "..."}]'
    )
    try:
        response = chat_complete([{"role": "user", "content": "Generate the interview questions now."}], system)
        parsed = _json_from_text(str(response))
        if not isinstance(parsed, list):
            raise ValueError("The AI response was not a JSON list.")
        questions = []
        for item in parsed[:n]:
            if isinstance(item, dict) and item.get("question"):
                questions.append(
                    {
                        "question": str(item.get("question", "")).strip(),
                        "ideal_answer_hints": str(item.get("ideal_answer_hints", "")).strip(),
                        "difficulty": str(item.get("difficulty", "Medium")).strip(),
                    }
                )
        if len(questions) < n:
            raise ValueError("The AI returned too few valid questions.")
        return questions, None
    except Exception as exc:
        return _fallback_questions(role, interview_type, resume_data, n), str(exc)


def _fallback_score(question: str, answer: str) -> dict[str, Any]:
    words = re.findall(r"\b\w+\b", answer)
    word_count = len(words)
    has_metric = bool(re.search(r"\d|%|\$|reduced|increased|improved|saved|grew", answer, re.IGNORECASE))
    has_structure = bool(re.search(r"situation|task|action|result|because|therefore|impact", answer, re.IGNORECASE))
    score = 4 + min(word_count // 35, 3) + (1 if has_metric else 0) + (1 if has_structure else 0)
    score = max(1, min(10, score))
    return {
        "score": score,
        "feedback": "Local scoring was used because the AI provider was unavailable. Your answer was assessed for detail, structure, relevance, and evidence.",
        "what_was_good": "You provided enough substance to evaluate the response." if word_count >= 40 else "You gave a starting answer that can be expanded.",
        "what_to_improve": f"Add a clear result, metric, and direct link back to the question: {question[:90]}...",
        "Clarity": max(1, min(10, score + (1 if word_count < 160 else 0))),
        "Relevance": score,
        "Depth": max(1, min(10, score - (0 if word_count > 80 else 1))),
        "Communication": max(1, min(10, score)),
        "Role-Fit": max(1, min(10, score + (1 if has_metric else 0))),
    }


def _score_answer(
    resume_data: dict[str, Any],
    role: str,
    interview_type: str,
    question: str,
    answer: str,
) -> tuple[dict[str, Any], str | None]:
    system = (
        f"Score this answer 1-10 for a {role} {interview_type} question.\n"
        f"Question: {question}. Answer: {answer}. Resume context: {_resume_json(resume_data)}\n"
        'Return strictly valid JSON only: {"score": 8, "feedback": "...", "what_was_good": "...", '
        '"what_to_improve": "...", "Clarity": 8, "Relevance": 8, "Depth": 8, '
        '"Communication": 8, "Role-Fit": 8}'
    )
    try:
        response = chat_complete([{"role": "user", "content": "Score my answer."}], system)
        parsed = _json_from_text(str(response))
        if not isinstance(parsed, dict):
            raise ValueError("The AI response was not a JSON object.")
        parsed["score"] = max(1, min(10, int(float(parsed.get("score", 1)))))
        for dimension in DIMENSIONS:
            parsed[dimension] = max(1, min(10, int(float(parsed.get(dimension, parsed["score"])))))
        return parsed, None
    except Exception as exc:
        return _fallback_score(question, answer), str(exc)


def _reset_interview() -> None:
    for key in (
        "interview_state",
        "interview_questions",
        "current_q_index",
        "interview_results",
        "interview_role",
        "interview_type",
        "interview_notice",
    ):
        st.session_state.pop(key, None)


def _provider_label() -> str:
    return {"chatgpt": "ChatGPT", "gemini": "Gemini", "anthropic": "Anthropic", "none": "Local fallback"}.get(active_provider(), "AI")


def render_mock_interview(resume_data: dict[str, Any]) -> None:
    st.session_state.setdefault("interview_state", "setup")
    st.session_state.setdefault("interview_questions", [])
    st.session_state.setdefault("current_q_index", 0)
    st.session_state.setdefault("interview_results", [])

    role_default = str(resume_data.get("predicted_role", "") or "")
    skill_count = len(resume_data.get("skills", []) or [])

    st.markdown(
        f"""
        <section class="careerai-interview-hero">
          <div>
            <div class="careerai-kicker">Interview simulator</div>
            <h2>AI Mock Interview</h2>
            <p>Practice targeted questions, submit answers, and get a scored report card across clarity, depth, communication, relevance, and role fit.</p>
          </div>
          <div class="careerai-provider-pill {active_provider()}"><span></span>{html.escape(_provider_label())}</div>
        </section>
        <div class="careerai-interview-stats">
          <div><span>Detected role</span><strong>{html.escape(role_default or "Not predicted")}</strong></div>
          <div><span>Skills found</span><strong>{skill_count}</strong></div>
          <div><span>Questions</span><strong>5</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    notice = st.session_state.get("interview_notice")
    if notice:
        st.info(notice)

    if st.session_state.interview_state == "setup":
        st.markdown('<div class="careerai-panel careerai-setup-panel">', unsafe_allow_html=True)
        target_role = st.text_input("Target Role", value=role_default)
        interview_type = st.selectbox("Interview Type", ["Behavioral", "Technical", "HR"])
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Generate Questions", type="primary"):
            if not target_role.strip():
                st.warning("Enter a target role first.")
            else:
                with st.spinner("Preparing interview questions..."):
                    questions, warning = _generate_questions(resume_data, target_role.strip(), interview_type)
                    st.session_state.interview_questions = questions
                    st.session_state.interview_role = target_role.strip()
                    st.session_state.interview_type = interview_type
                    st.session_state.current_q_index = 0
                    st.session_state.interview_results = []
                    st.session_state.interview_notice = (
                        f"AI provider unavailable, so CareerAI generated local practice questions. Reason: {warning}"
                        if warning else None
                    )
                    st.session_state.interview_state = "asking"
                    st.rerun()

    elif st.session_state.interview_state == "asking":
        questions = st.session_state.interview_questions
        q_index = st.session_state.current_q_index
        total = len(questions)
        question_data = questions[q_index]

        st.progress(q_index / total, text=f"Question {q_index + 1} of {total}")
        st.markdown(
            f"""
            <div class="careerai-question-card">
              <div class="careerai-question-meta">
                <span>Q{q_index + 1}</span>
                <strong>{html.escape(question_data.get('difficulty', 'Medium'))}</strong>
              </div>
              <h3>{html.escape(question_data['question'])}</h3>
              <p>{html.escape(question_data.get('ideal_answer_hints', 'Use a concise, evidence-backed answer.'))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        answer = st.text_area("Your Answer", height=190, key=f"interview_answer_{q_index}")

        if st.button("Submit Answer", type="primary"):
            if not answer.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                with st.spinner("Scoring your answer..."):
                    score_data, warning = _score_answer(
                        resume_data,
                        st.session_state.interview_role,
                        st.session_state.interview_type,
                        question_data["question"],
                        answer.strip(),
                    )
                    if warning:
                        score_data["provider_warning"] = warning
                    st.session_state.interview_results.append(
                        {"question": question_data["question"], "answer": answer.strip(), "score_data": score_data}
                    )
                    st.session_state.current_q_index += 1
                    st.session_state.interview_state = "report" if st.session_state.current_q_index >= total else "asking"
                    st.rerun()

    elif st.session_state.interview_state == "report":
        results = st.session_state.interview_results
        if not results:
            st.warning("No interview results found.")
            _reset_interview()
            st.rerun()

        averages = {
            dimension: sum(item["score_data"].get(dimension, item["score_data"].get("score", 0)) for item in results) / len(results)
            for dimension in DIMENSIONS
        }
        overall = sum(item["score_data"].get("score", 0) for item in results) / len(results)
        st.markdown(f"<div class='careerai-report-score'><span>Final report card</span><strong>{overall:.1f}/10</strong></div>", unsafe_allow_html=True)

        radar_df = pd.DataFrame({"dimension": list(averages.keys()), "score": list(averages.values())})
        fig = px.line_polar(radar_df, r="score", theta="dimension", line_close=True, range_r=[0, 10])
        fig.update_traces(fill="toself")
        fig.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        for index, result in enumerate(results, 1):
            score_data = result["score_data"]
            with st.expander(f"Q{index}: {result['question']} - {score_data.get('score', 0)}/10"):
                st.write(f"**Your answer:** {result['answer']}")
                st.write(f"**Feedback:** {score_data.get('feedback', '')}")
                st.write(f"**What was good:** {score_data.get('what_was_good', '')}")
                st.write(f"**What to improve:** {score_data.get('what_to_improve', '')}")
                if score_data.get("provider_warning"):
                    st.caption(f"Local scoring fallback used: {score_data['provider_warning']}")

        if st.button("Start New Interview"):
            _reset_interview()
            st.rerun()
