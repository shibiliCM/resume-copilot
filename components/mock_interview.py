from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.llm import chat_complete


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


def _generate_questions(resume_data: dict[str, Any], role: str, interview_type: str, n: int = 5) -> list[dict[str, Any]]:
    system = (
        f"You are a strict but fair interviewer for {role}.\n"
        f"Candidate resume: {_resume_json(resume_data)}.\n"
        f"Generate {n} {interview_type} interview questions.\n"
        'Return JSON: [{"question": "...", "ideal_answer_hints": "...", "difficulty": "..."}]'
    )
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
    return questions


def _score_answer(
    resume_data: dict[str, Any],
    role: str,
    interview_type: str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    system = (
        f"Score this answer 1-10 for a {role} {interview_type} question.\n"
        f"Question: {question}. Answer: {answer}. Resume context: {_resume_json(resume_data)}\n"
        'Return JSON: {"score": 8, "feedback": "...", "what_was_good": "...", '
        '"what_to_improve": "...", "Clarity": 8, "Relevance": 8, "Depth": 8, '
        '"Communication": 8, "Role-Fit": 8}'
    )
    response = chat_complete([{"role": "user", "content": "Score my answer."}], system)
    parsed = _json_from_text(str(response))
    if not isinstance(parsed, dict):
        raise ValueError("The AI response was not a JSON object.")
    parsed["score"] = max(1, min(10, int(float(parsed.get("score", 1)))))
    for dimension in DIMENSIONS:
        parsed[dimension] = max(1, min(10, int(float(parsed.get(dimension, parsed["score"])))))
    return parsed


def _reset_interview() -> None:
    for key in (
        "interview_state",
        "interview_questions",
        "current_q_index",
        "interview_results",
        "interview_role",
        "interview_type",
    ):
        st.session_state.pop(key, None)


def render_mock_interview(resume_data: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="fade-up" style="margin-bottom:1.5rem;">
          <h2 style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:var(--text);margin:0 0 6px;">
            🎤 AI Mock Interview
          </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("interview_state", "setup")
    st.session_state.setdefault("interview_questions", [])
    st.session_state.setdefault("current_q_index", 0)
    st.session_state.setdefault("interview_results", [])

    if st.session_state.interview_state == "setup":
        target_role = st.text_input("Target Role", value=str(resume_data.get("predicted_role", "") or ""))
        interview_type = st.selectbox("Interview Type", ["Behavioral", "Technical", "HR"])

        if st.button("Generate Questions", type="primary"):
            if not target_role.strip():
                st.warning("Enter a target role first.")
            else:
                with st.spinner("Generating customized questions..."):
                    try:
                        st.session_state.interview_questions = _generate_questions(resume_data, target_role.strip(), interview_type)
                        st.session_state.interview_role = target_role.strip()
                        st.session_state.interview_type = interview_type
                        st.session_state.current_q_index = 0
                        st.session_state.interview_results = []
                        st.session_state.interview_state = "asking"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to generate questions: {exc}")

    elif st.session_state.interview_state == "asking":
        questions = st.session_state.interview_questions
        q_index = st.session_state.current_q_index
        total = len(questions)
        question_data = questions[q_index]

        st.progress(q_index / total, text=f"Question {q_index + 1} of {total}")
        st.markdown(
            f"""
            <div class="careerai-card" style="padding:1rem;border-radius:14px;border:1px solid rgba(128,128,128,.18);margin-bottom:1rem;">
              <div style="font-weight:800;margin-bottom:.35rem;">Q{q_index + 1}: {question_data['question']}</div>
              <div style="opacity:.72;font-size:13px;">Difficulty: {question_data.get('difficulty', 'Medium')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        answer = st.text_area("Your Answer", height=170, key=f"interview_answer_{q_index}")

        if st.button("Submit Answer", type="primary"):
            if not answer.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                with st.spinner("Scoring your answer..."):
                    try:
                        score_data = _score_answer(
                            resume_data,
                            st.session_state.interview_role,
                            st.session_state.interview_type,
                            question_data["question"],
                            answer.strip(),
                        )
                        st.session_state.interview_results.append(
                            {"question": question_data["question"], "answer": answer.strip(), "score_data": score_data}
                        )
                        st.session_state.current_q_index += 1
                        st.session_state.interview_state = (
                            "report" if st.session_state.current_q_index >= total else "asking"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to score answer: {exc}")

        if st.session_state.interview_results:
            latest = st.session_state.interview_results[-1]["score_data"]
            with st.expander("Latest feedback", expanded=False):
                st.markdown(f"<span class='careerai-score-badge'>{latest.get('score', 0)}/10</span>", unsafe_allow_html=True)
                st.write(f"**Feedback:** {latest.get('feedback', '')}")
                st.write(f"**What was good:** {latest.get('what_was_good', '')}")
                st.write(f"**What to improve:** {latest.get('what_to_improve', '')}")

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
        st.markdown(f"### Final Report Card: **{overall:.1f}/10**")

        radar_df = pd.DataFrame({"dimension": list(averages.keys()), "score": list(averages.values())})
        fig = px.line_polar(radar_df, r="score", theta="dimension", line_close=True, range_r=[0, 10])
        fig.update_traces(fill="toself")
        fig.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=420)
        st.plotly_chart(fig, use_container_width=True)

        strengths: list[str] = []
        improvements: list[str] = []
        for index, result in enumerate(results, 1):
            score_data = result["score_data"]
            strengths.append(str(score_data.get("what_was_good", "")).strip())
            improvements.append(str(score_data.get("what_to_improve", "")).strip())
            with st.expander(f"Q{index}: {result['question']} - {score_data.get('score', 0)}/10"):
                st.write(f"**Your answer:** {result['answer']}")
                st.write(f"**Feedback:** {score_data.get('feedback', '')}")
                st.write(f"**What was good:** {score_data.get('what_was_good', '')}")
                st.write(f"**What to improve:** {score_data.get('what_to_improve', '')}")

        st.markdown("### Strengths")
        for item in [value for value in strengths if value][:3]:
            st.write(f"- {item}")

        st.markdown("### Improvements")
        for item in [value for value in improvements if value][:3]:
            st.write(f"- {item}")

        if st.button("Start New Interview"):
            _reset_interview()
            st.rerun()
