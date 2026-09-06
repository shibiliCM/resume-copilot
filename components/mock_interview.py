from __future__ import annotations

import html
import json
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.llm import active_provider, chat_complete, _check_offline_option_score


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
            {
                "question": f"Tell me about a time you used {skill_text} to solve a difficult problem.",
                "options": {
                    "A": "I faced a critical bug in production. Using my skills, I diagnosed the issue using logs, refactored the code, and deployed a fix that reduced latency by 30%.",
                    "B": "I had a problem with our application. I checked my skills, asked a senior developer, and we resolved it after a few hours.",
                    "C": "I usually ask my team lead to assign difficult problems to someone else so I can focus on simpler tasks."
                }
            },
            {
                "question": f"Describe a project where you had to learn something quickly for a {role} outcome.",
                "options": {
                    "A": "Our project required a new technology. I spent my weekend studying documentation, built a small proof of concept, and integrated it successfully into the sprint.",
                    "B": "We had to learn a new tool. I read some tutorials online and started using it on the job.",
                    "C": "I told my manager that I don't know the tool, so I waited for the training program next quarter."
                }
            },
            {
                "question": "Tell me about a time you received critical feedback and changed your approach.",
                "options": {
                    "A": "My peer review highlighted that my code documentation was lacking. I took it seriously, created wiki pages, and received positive feedback in the next review.",
                    "B": "My manager told me to write more comments, so I started adding comments in my code.",
                    "C": "I argued that my code was self-documenting and did not need comments."
                }
            },
            {
                "question": "Describe a situation where you had to communicate a technical idea to a non-technical audience.",
                "options": {
                    "A": "I used a simple analogy comparing our database storage to a library, which helped the product managers understand the need for our refactoring work.",
                    "B": "I explained the schemas and query execution plans directly, trying to make it as simple as possible.",
                    "C": "I told them it is too technical for them to understand and just asked for their approval."
                }
            },
            {
                "question": f"Give an example of how you handled competing priorities while preparing for a {role} responsibility.",
                "options": {
                    "A": "I listed all tasks, estimated the business value of each, aligned with my manager on high-priority items, and delegated the remaining ones.",
                    "B": "I worked overtime to complete all tasks, prioritizing whatever came first in my email.",
                    "C": "I became stressed and requested to delay some deadlines until things quieted down."
                }
            }
        ],
        "Technical": [
            {
                "question": f"Walk me through one resume project and explain the architecture, tradeoffs, and measurable result.",
                "options": {
                    "A": "I built a project with a Microservices architecture. The tradeoff was higher setup complexity, but the result was a 40% improvement in scalability.",
                    "B": "I worked on a web project using MVC architecture. It worked well and had a nice UI.",
                    "C": "I don't recall the specific architecture, but it was a database-driven app."
                }
            },
            {
                "question": f"Which of these skills are strongest for you: {skill_text}? Explain with a concrete example.",
                "options": {
                    "A": f"My strongest skill is working with {skill_text}. I recently applied this skill to optimize our codebase, saving 15 hours of manual work weekly.",
                    "B": f"I am comfortable with {skill_text} because I use them daily in my current role.",
                    "C": f"I list {skill_text} on my resume, but I would need guidance to use them in a real project."
                }
            },
            {
                "question": f"What would you do in the next 30 days to close these gaps: {gap_text}?",
                "options": {
                    "A": "I have already bookmarked a structured certification course and plan to build two hands-on projects to master these gaps within the month.",
                    "B": "I will search for tutorials on these topics and try to read about them when I have time.",
                    "C": "I think these gaps are minor and I can learn them on the job as they come up."
                }
            },
            {
                "question": f"Explain how you would debug a production issue related to your {role} work.",
                "options": {
                    "A": "I would check the error logs, look at the recent commits, replicate the issue in staging, apply a fix, and verify it with automated tests.",
                    "B": "I would look at the code where I think the error is, make some modifications, and deploy to see if it fixes it.",
                    "C": "I would restart the production server and hope that clears the issue."
                }
            },
            {
                "question": "Choose one technical achievement from your resume and explain how you would improve it today.",
                "options": {
                    "A": "I would improve the security and test coverage by implementing JWT authentication and writing end-to-end integration tests.",
                    "B": "I would rewrite some parts of the code to make it look cleaner and run slightly faster.",
                    "C": "I think it is already perfect and does not need any improvements."
                }
            }
        ],
        "HR": [
            {
                "question": f"Why are you interested in this {role} role?",
                "options": {
                    "A": "I want to apply my technical background to solve your scale challenges, and I see a strong alignment between my skills and your company culture.",
                    "B": "It seems like a good career opportunity that fits my qualifications well.",
                    "C": "I am looking for a job change and your company was hiring for this role."
                }
            },
            {
                "question": "Walk me through your resume in two minutes.",
                "options": {
                    "A": "I have X years of experience, specializing in key skills. I led a major project that achieved a 30% performance boost, and I'm looking to apply this expertise here.",
                    "B": "I started as a junior dev, then got promoted, and now I'm looking for my next role to continue my career path.",
                    "C": "You can read the details in the resume; it lists all my education and previous companies."
                }
            },
            {
                "question": "What are your strongest professional strengths, and where is the evidence in your resume?",
                "options": {
                    "A": "My strongest strength is problem-solving. In my resume, this is demonstrated where I successfully resolved a long-standing performance bottleneck.",
                    "B": "I am a hard worker and a quick learner, which you can see from my progression across my jobs.",
                    "C": "I am very cooperative and punctual. I always complete my tasks on time."
                }
            },
            {
                "question": f"What compensation, work style, and growth expectations matter most to you for a {role} position?",
                "options": {
                    "A": "I look for a competitive salary aligned with the market, a collaborative environment, and opportunities to take ownership of projects.",
                    "B": "I prefer a hybrid work schedule and standard industry benefits.",
                    "C": "I am primarily interested in a high salary and minimal overtime."
                }
            },
            {
                "question": "Why should we choose you over another candidate with similar experience?",
                "options": {
                    "A": "Beyond my technical capabilities, I bring a proactive problem-solving mindset and a track record of driving measurable business impact.",
                    "B": "I am very dedicated, and I will work hard to ensure all my tasks are completed correctly.",
                    "C": "I think I am the best fit, and my previous managers have always liked my work."
                }
            }
        ]
    }
    
    selected = templates.get(interview_type, templates["Behavioral"])[:n]
    results = []
    for item in selected:
        results.append({
            "question": item["question"],
            "difficulty": "Medium",
            "ideal_answer_hints": "Use a concise STAR structure, include metrics, and connect the answer to the target role.",
            "options": item["options"]
        })
    return results


def _generate_questions(resume_data: dict[str, Any], role: str, interview_type: str, n: int = 5) -> tuple[list[dict[str, Any]], str | None]:
    system = (
        f"You are a strict but fair interviewer for {role}.\n"
        f"Candidate resume: {_resume_json(resume_data)}.\n"
        f"Generate {n} {interview_type} interview questions.\n"
        "For each question, provide 3 multiple choice options (A, B, C) where:\n"
        "Option A is a strong, STAR-structured answer.\n"
        "Option B is a generic, average answer.\n"
        "Option C is a weak or incorrect answer.\n"
        'Return strictly valid JSON only: [{"question": "...", "ideal_answer_hints": "...", "difficulty": "...", "options": {"A": "...", "B": "...", "C": "..."}}]'
    )
    try:
        response = chat_complete([{"role": "user", "content": "Generate the interview questions now."}], system)
        parsed = _json_from_text(str(response))
        if not isinstance(parsed, list):
            raise ValueError("The AI response was not a JSON list.")
        questions = []
        for item in parsed[:n]:
            if isinstance(item, dict) and item.get("question"):
                options = item.get("options")
                if not isinstance(options, dict) or not all(k in options for k in ("A", "B", "C")):
                    options = {
                        "A": "A strong, STAR-structured response detailing action and measurable results.",
                        "B": "An average, textbook response explaining the concept without specific metrics.",
                        "C": "A weak or incorrect response."
                    }
                questions.append(
                    {
                        "question": str(item.get("question", "")).strip(),
                        "ideal_answer_hints": str(item.get("ideal_answer_hints", "")).strip(),
                        "difficulty": str(item.get("difficulty", "Medium")).strip(),
                        "options": options
                    }
                )
        if len(questions) < n:
            raise ValueError("The AI returned too few valid questions.")
        return questions, None
    except Exception as exc:
        return _fallback_questions(role, interview_type, resume_data, n), str(exc)


def _fallback_score(question: str, answer: str) -> dict[str, Any]:
    match_res = _check_offline_option_score(answer)
    if match_res:
        score, feedback, what_good, what_improve = match_res
        return {
            "score": score,
            "feedback": feedback,
            "what_was_good": what_good,
            "what_to_improve": what_improve,
            "Clarity": score,
            "Relevance": score,
            "Depth": score,
            "Communication": score,
            "Role-Fit": score,
        }

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
        
        # Refer online link search query
        import urllib.parse
        search_query = urllib.parse.quote(f"how to answer {question_data['question']} interview question")
        refer_online_html = (
            f'<div style="margin-top: 12px;">'
            f'<a href="https://www.google.com/search?q={search_query}" target="_blank" '
            f'style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; '
            f'background: rgba(77, 166, 255, 0.12); border: 1px solid rgba(77, 166, 255, 0.3); '
            f'border-radius: 8px; color: #4da6ff; font-size: 13px; font-weight: 600; text-decoration: none; '
            f'transition: background 0.2s;">'
            f'🔍 Refer Online (Search Google)</a>'
            f'</div>'
        )

        st.markdown(
            f"""
            <div class="careerai-question-card">
              <div class="careerai-question-meta">
                <span>Q{q_index + 1}</span>
                <strong>{html.escape(question_data.get('difficulty', 'Medium'))}</strong>
              </div>
              <h3>{html.escape(question_data['question'])}</h3>
              <p>{html.escape(question_data.get('ideal_answer_hints', 'Use a concise, evidence-backed answer.'))}</p>
              {refer_online_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        options_dict = question_data.get("options", {})
        opt_a = options_dict.get("A", "Strong STAR-structured answer")
        opt_b = options_dict.get("B", "Average generic answer")
        opt_c = options_dict.get("C", "Weak/incorrect answer")

        if "shuffled_options" not in question_data:
            import random
            keys = ["A", "B", "C"]
            random.shuffle(keys)
            shuffled_options = []
            for i, key in enumerate(keys):
                letter = chr(65 + i)  # 'A', 'B', 'C'
                opt_text = options_dict.get(key, "")
                shuffled_options.append({
                    "display": f"Option {letter}: {opt_text}",
                    "original_key": key,
                    "text": opt_text
                })
            question_data["shuffled_options"] = shuffled_options

        shuffled = question_data["shuffled_options"]
        radio_choices = [item["display"] for item in shuffled] + ["Write a custom answer..."]

        choice = st.radio(
            "Select an answer option or write a custom one:",
            radio_choices,
            key=f"interview_choice_{q_index}"
        )

        final_answer = ""
        matched_item = None
        for item in shuffled:
            if item["display"] == choice:
                matched_item = item
                break

        if matched_item:
            final_answer = matched_item["text"]
            st.info(f"Selected: {final_answer}")
        else:
            final_answer = st.text_area("Your Custom Answer", height=150, key=f"interview_answer_{q_index}")

        if st.button("Submit Answer", type="primary"):
            if not final_answer.strip():
                st.warning("Please provide or select an answer before submitting.")
            else:
                with st.spinner("Scoring your answer..."):
                    score_data, warning = _score_answer(
                        resume_data,
                        st.session_state.interview_role,
                        st.session_state.interview_type,
                        question_data["question"],
                        final_answer.strip(),
                    )
                    if warning:
                        score_data["provider_warning"] = warning
                    st.session_state.interview_results.append(
                        {
                            "question": question_data["question"],
                            "answer": final_answer.strip(),
                            "ideal_answer": opt_a,
                            "score_data": score_data
                        }
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
                # Side-by-side comparison cards
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f'<div style="background: rgba(255, 255, 255, 0.02); padding: 16px; '
                        f'border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); '
                        f'min-height: 140px; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">'
                        f'<span style="color: #4da6ff; font-weight: 700; font-size: 14px; display: block; margin-bottom: 8px;">👤 YOUR ANSWER</span>'
                        f'<p style="margin: 0; font-size: 13.5px; line-height: 1.5; color: #e1e3e6;">{html.escape(result["answer"])}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    ideal_ans = result.get("ideal_answer", "Option A")
                    st.markdown(
                        f'<div style="background: rgba(0, 201, 167, 0.04); padding: 16px; '
                        f'border-radius: 12px; border: 1px solid rgba(0, 201, 167, 0.2); '
                        f'min-height: 140px; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">'
                        f'<span style="color: #00c9a7; font-weight: 700; font-size: 14px; display: block; margin-bottom: 8px;">🌟 CORRECT / IDEAL ANSWER</span>'
                        f'<p style="margin: 0; font-size: 13.5px; line-height: 1.5; color: #f1f3f4;">{html.escape(ideal_ans)}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Feedback:** {score_data.get('feedback', '')}")
                st.write(f"**What was good:** {score_data.get('what_was_good', '')}")
                st.write(f"**What to improve:** {score_data.get('what_to_improve', '')}")
                if score_data.get("provider_warning"):
                    st.caption(f"Local scoring fallback used: {score_data['provider_warning']}")

        if st.button("Start New Interview"):
            _reset_interview()
            st.rerun()
