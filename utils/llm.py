from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import Any

import anthropic
import streamlit as st


ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def _config(name: str, default: str = "") -> str:
    session_value = st.session_state.get(name, "")
    if session_value:
        value = str(session_value).strip()
        if not _is_placeholder(value):
            return value
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    value = str(secret_value or os.getenv(name, default)).strip()
    return "" if _is_placeholder(value) else value


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().strip('"').strip("'").lower()
    return lowered.startswith("your_") and lowered.endswith("_api_key_here")


def active_provider() -> str:
    openai_base_url = _config("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    local_openai = openai_base_url.startswith(("http://localhost", "http://127.0.0.1"))
    if _config("OPENAI_API_KEY") or local_openai:
        return "chatgpt"
    if _config("GEMINI_API_KEY"):
        return "gemini"
    if _config("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def get_client() -> anthropic.Anthropic:
    api_key = _config("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("No Anthropic key found.")
    return anthropic.Anthropic(api_key=api_key)


def _openai_payload(messages: list[dict[str, str]], system: str, stream: bool) -> bytes:
    return json.dumps(
        {
            "model": _config("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.7,
            "max_tokens": 1200,
            "stream": stream,
        }
    ).encode("utf-8")


def _openai_request(messages: list[dict[str, str]], system: str, stream: bool = False) -> urllib.request.Request:
    base_url = _config("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = _config("OPENAI_API_KEY") or "local"
    return urllib.request.Request(
        f"{base_url}/chat/completions",
        data=_openai_payload(messages, system, stream),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )


def _chatgpt_complete(messages: list[dict[str, str]], system: str) -> str:
    try:
        with urllib.request.urlopen(_openai_request(messages, system), timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_friendly_api_error("ChatGPT/OpenAI", exc.code, details or exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach ChatGPT/OpenAI endpoint: {exc.reason}") from exc


def _chatgpt_stream(messages: list[dict[str, str]], system: str) -> Generator[str, None, None]:
    try:
        with urllib.request.urlopen(_openai_request(messages, system, stream=True), timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                delta = data["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_friendly_api_error("ChatGPT/OpenAI", exc.code, details or exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach ChatGPT/OpenAI endpoint: {exc.reason}") from exc


def _gemini_prompt(messages: list[dict[str, str]], system: str) -> str:
    turns = [f"System: {system}"]
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        turns.append(f"{role}: {message['content']}")
    return "\n\n".join(turns)


def _gemini_complete(messages: list[dict[str, str]], system: str) -> str:
    api_key = _config("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Gemini key found.")
    model = _config("GEMINI_MODEL", "gemini-3.5-flash")
    payload = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": _gemini_prompt(messages, system)}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1200},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts).strip()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_friendly_api_error("Gemini", exc.code, details or exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc


def _anthropic_complete(messages: list[dict[str, str]], system: str) -> str:
    response = get_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def _anthropic_stream(messages: list[dict[str, str]], system: str) -> Generator[str, None, None]:
    with get_client().messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _check_offline_option_score(user_ans: str) -> tuple[int, str, str, str] | None:
    import re
    # Clean text helper
    def clean(t: str) -> str:
        return re.sub(r"\s+", "", str(t)).lower()
        
    cleaned_ans = clean(user_ans)
    if not cleaned_ans:
        return None
        
    # Option A (Strong) candidates:
    strong_texts = [
        "I faced a critical bug in production. Using my skills, I diagnosed the issue using logs, refactored the code, and deployed a fix that reduced latency by 30%.",
        "Our project required a new technology. I spent my weekend studying documentation, built a small proof of concept, and integrated it successfully into the sprint.",
        "My peer review highlighted that my code documentation was lacking. I took it seriously, created wiki pages, and received positive feedback in the next review.",
        "I used a simple analogy comparing our database storage to a library, which helped the product managers understand the need for our refactoring work.",
        "I listed all tasks, estimated the business value of each, aligned with my manager on high-priority items, and delegated the remaining ones.",
        "I built a project with a Microservices architecture. The tradeoff was higher setup complexity, but the result was a 40% improvement in scalability.",
        "I have already bookmarked a structured certification course and plan to build two hands-on projects to master these gaps within the month.",
        "I would check the error logs, look at the recent commits, replicate the issue in staging, apply a fix, and verify it with automated tests.",
        "I designed and deployed a key feature using our core stack, reducing user drop-off rate by 15% and optimizing database loads.",
        "I have set up a local development environment to build a microservices project, specifically practicing container orchestration and networking.",
        "I immediately aligned with the product manager to understand the priority, adjusted our sprint plan, and communicated impact to the team.",
        "Your product scales to millions of users, and my experience in backend optimization directly solves the performance needs of this role.",
        "I analyzed log patterns, isolated the issue to a connection pool leak, replicated it in staging, and deployed a pooled connection patch."
    ]
    
    # Option B (Average) candidates:
    average_texts = [
        "I had a problem with our application. I checked my skills, asked a senior developer, and we resolved it after a few hours.",
        "We had to learn a new tool. I read some tutorials online and started using it on the job.",
        "My manager told me to write more comments, so I started adding comments in my code.",
        "I explained the schemas and query execution plans directly, trying to make it as simple as possible.",
        "I worked overtime to complete all tasks, prioritizing whatever came first in my email.",
        "I worked on a web project using MVC architecture. It worked well and had a nice UI.",
        "I will search for tutorials on these topics and try to read about them when I have time.",
        "I would look at the code where I think the error is, make some modifications, and deploy to see if it fixes it.",
        "I worked on the project for a few months, building standard screens and backend APIs based on specifications.",
        "I plan to search for tutorials online and read about these topics when my schedule allows.",
        "We had to rewrite some components. I worked overtime to finish the new tasks before the deadline.",
        "It seems like a great company with a good culture and good benefits, which fits my search.",
        "I searched the error on StackOverflow, tried a few potential solutions, and restarted the server."
    ]
    
    # Option C (Weak) candidates:
    weak_texts = [
        "I usually ask my team lead to assign difficult problems to someone else so I can focus on simpler tasks.",
        "I told my manager that I don't know the tool, so I waited for the training program next quarter.",
        "I argued that my code was self-documenting and did not need comments.",
        "I told them it is too technical for them to understand and just asked for their approval.",
        "I became stressed and requested to delay some deadlines until things quieted down.",
        "I don't recall the specific architecture, but it was a database-driven app.",
        "I think these gaps are minor and I can learn them on the job as they come up.",
        "I would restart the production server and hope that clears the issue.",
        "I don't recall the specific details, but it was a team effort and we completed it on time.",
        "I believe these skills can be picked up easily on the job, so I haven't started studying them yet.",
        "I complained to the manager that changing requirements mid-sprint would delay our project delivery.",
        "I am looking for a new job opportunity where I can apply my skills and grow my career.",
        "I passed the ticket to the infrastructure team since production access was restricted."
    ]

    for t in strong_texts:
        if clean(t) == cleaned_ans:
            return 10, "Excellent STAR-structured answer! Your response perfectly demonstrates key accomplishments, actions, and clear outcomes.", "Clear STAR structure, quantified results, and highly relevant details.", "None. This is an ideal answer."
            
    for t in average_texts:
        if clean(t) == cleaned_ans:
            return 7, "Good, average response. You described the tasks and responsibilities, but it lacks specific business impact or quantified metrics.", "Good high-level overview of the work done.", "Quantify your achievements with concrete metrics and structure it using the STAR format."
            
    for t in weak_texts:
        if clean(t) == cleaned_ans:
            return 4, "Weak response. The answer lacks detail, structure, and positive outcomes.", "Acknowledged the topic.", "Be more proactive, focus on positive results, and detail your actions."

    # Dynamic checks for skills-based templates
    if cleaned_ans.startswith("mystrongestskillisworkingwith"):
        return 10, "Excellent STAR-structured answer! Your response perfectly demonstrates key accomplishments, actions, and clear outcomes.", "Clear STAR structure, quantified results, and highly relevant details.", "None. This is an ideal answer."
        
    if cleaned_ans.endswith("becauseiusethemdailyinmycurrentrole"):
        return 7, "Good, average response. You described the tasks and responsibilities, but it lacks specific business impact or quantified metrics.", "Good high-level overview of the work done.", "Quantify your achievements with concrete metrics and structure it using the STAR format."
        
    if cleaned_ans.endswith("butiwouldneedguidancetouse theminarealproject") or cleaned_ans.endswith("butiwouldneedguidancetouse theminarealproject."):
        return 4, "Weak response. The answer lacks detail, structure, and positive outcomes.", "Acknowledged the topic.", "Be more proactive, focus on positive results, and detail your actions."

    return None


def _offline_local_response(messages: list[dict[str, str]], system: str) -> str:
    import re
    resume = {}
    try:
        # Try to parse any JSON from system prompt
        start = system.find("{")
        end = system.rfind("}")
        if start != -1 and end != -1:
            resume = json.loads(system[start:end+1])
    except Exception:
        pass

    # Extract resume fields with default fallbacks
    role = resume.get("predicted_role") or "Software Engineer"
    skills = resume.get("skills", []) or ["Python", "SQL", "Git", "Docker"]
    gaps = resume.get("skill_gaps", []) or ["System Design", "Cloud Deployments", "Kubernetes"]
    ats_score = resume.get("ats_score") or "72"

    skills_str = ", ".join(skills[:5])
    gaps_str = ", ".join(gaps[:3]) if gaps else "none detected"

    # Check if system prompt is asking for JSON lists (Interview Questions/Scoring)
    if "strictly valid JSON only" in system or "options" in system:
        if "score" in system:
            # Extract user answer from system prompt if present, else fallback
            ans_match = re.search(r"Answer:\s*(.*?)\.\s*Resume context:", system, re.DOTALL)
            if ans_match:
                user_ans = ans_match.group(1).strip()
            else:
                user_ans = messages[-1]["content"] if messages else ""
                
            match_res = _check_offline_option_score(user_ans)
            if match_res:
                score, feedback, what_good, what_improve = match_res
                fallback_score_obj = {
                    "score": score,
                    "feedback": feedback,
                    "what_was_good": what_good,
                    "what_to_improve": what_improve,
                    "Clarity": score,
                    "Relevance": score,
                    "Depth": score,
                    "Communication": score,
                    "Role-Fit": score
                }
            else:
                words = re.findall(r"\b\w+\b", user_ans)
                word_count = len(words)
                has_metric = bool(re.search(r"\d|%|\$|reduced|increased|improved|saved|grew", user_ans, re.IGNORECASE))
                has_structure = bool(re.search(r"situation|task|action|result|because|therefore|impact", user_ans, re.IGNORECASE))
                score = 4 + min(word_count // 35, 3) + (1 if has_metric else 0) + (1 if has_structure else 0)
                score = max(1, min(10, score))
                
                fallback_score_obj = {
                    "score": score,
                    "feedback": "Offline analysis completed. Your answer was assessed for clarity, relevance, and evidence.",
                    "what_was_good": "You provided enough substance to evaluate the response." if word_count >= 40 else "You gave a starting answer.",
                    "what_to_improve": "Add a clear STAR structure, connecting your actions directly to the business result.",
                    "Clarity": max(1, min(10, score + (1 if word_count < 160 else 0))),
                    "Relevance": score,
                    "Depth": max(1, min(10, score - (0 if word_count > 80 else 1))),
                    "Communication": max(1, min(10, score)),
                    "Role-Fit": max(1, min(10, score + (1 if has_metric else 0)))
                }
            return json.dumps(fallback_score_obj)
        else:
            mock_questions = [
                {
                    "question": f"Walk me through one project on your resume where you used {skills_str}.",
                    "ideal_answer_hints": "Use STAR structure: specify what you built, your contribution, and the metric.",
                    "difficulty": "Medium",
                    "options": {
                        "A": "I designed and deployed a key feature using our core stack, reducing user drop-off rate by 15% and optimizing database loads.",
                        "B": "I worked on the project for a few months, building standard screens and backend APIs based on specifications.",
                        "C": "I don't recall the specific details, but it was a team effort and we completed it on time."
                    }
                },
                {
                    "question": f"How do you plan to close the visible skill gaps in your profile, specifically {gaps_str}?",
                    "ideal_answer_hints": "Show proactivity: mention self-study, a planned project, or specific resources.",
                    "difficulty": "Medium",
                    "options": {
                        "A": "I have set up a local development environment to build a microservices project, specifically practicing container orchestration and networking.",
                        "B": "I plan to search for tutorials online and read about these topics when my schedule allows.",
                        "C": "I believe these skills can be picked up easily on the job, so I haven't started studying them yet."
                    }
                },
                {
                    "question": "Tell me about a time you had to deal with a sudden change in requirements or project scope.",
                    "ideal_answer_hints": "Focus on adaptability: communication, reassessing priorities, and alignment.",
                    "difficulty": "Medium",
                    "options": {
                        "A": "I immediately aligned with the product manager to understand the priority, adjusted our sprint plan, and communicated impact to the team.",
                        "B": "We had to rewrite some components. I worked overtime to finish the new tasks before the deadline.",
                        "C": "I complained to the manager that changing requirements mid-sprint would delay our project delivery."
                    }
                },
                {
                    "question": f"Why do you want to join us as a {role}?",
                    "ideal_answer_hints": "Alignment: connect your skills to their challenges and show interest in their engineering culture.",
                    "difficulty": "Medium",
                    "options": {
                        "A": "Your product scales to millions of users, and my experience in backend optimization directly solves the performance needs of this role.",
                        "B": "It seems like a great company with a good culture and good benefits, which fits my search.",
                        "C": "I am looking for a new job opportunity where I can apply my skills and grow my career."
                    }
                },
                {
                    "question": "Describe a scenario where you had to debug a complex issue in a production environment.",
                    "ideal_answer_hints": "Analytical thinking: checking logs, replicating the bug, adding tests, and building a post-mortem.",
                    "difficulty": "Hard",
                    "options": {
                        "A": "I analyzed log patterns, isolated the issue to a connection pool leak, replicated it in staging, and deployed a pooled connection patch.",
                        "B": "I searched the error on StackOverflow, tried a few potential solutions, and restarted the server.",
                        "C": "I passed the ticket to the infrastructure team since production access was restricted."
                    }
                }
            ]
            return json.dumps(mock_questions)

    query = messages[-1]["content"].lower() if messages else ""

    if any(k in query for k in ("improve", "score", "ats", "review", "grade", "points")):
        return (
            f"Here is my direct analysis of your **{role}** resume to optimize for ATS systems:\n\n"
            f"* **Current ATS Score: {ats_score}/100** - A solid baseline, but we can target 85+ by making specific updates.\n"
            f"* **Keyword Alignment**: Ensure high-priority keywords like **{gaps_str}** are represented. Do not just list them; integrate them into your work achievements.\n"
            f"* **Quantify Achievements**: Ensure at least 60% of your bullet points contain measurable metrics (e.g., 'reduced API latency by 15%', 'boosted team throughput by 20%').\n"
            f"* **Formatting Cleanliness**: Avoid non-standard headers, side columns, graphics, or tables. ATS parsers read left-to-right, top-to-bottom."
        )

    if any(k in query for k in ("skill", "gap", "missing", "learn", "course", "cert")):
        return (
            f"Looking closely at your current skillset, here is your skill gap analysis for a **{role}** profile:\n\n"
            f"* **Core Missing Keywords**: **{gaps_str}**. These are highly sought after by recruiters for {role} roles.\n"
            f"* **Action Plan**: Create 1-2 small projects that showcase these skills. For instance, build an API service utilizing **{gaps[0] if gaps else 'System Design'}** to demonstrate capability.\n"
            f"* **Online Learning**: Focus on self-study certifications or documentation walks to build immediate fluency."
        )

    if any(k in query for k in ("job", "match", "hire", "recruit", "role", "position")):
        return (
            f"Based on your background and your detected skills (**{skills_str}**), you are a strong match for the following positions:\n\n"
            f"* **{role}** (Primary Match): High alignment with your project history.\n"
            f"* **Technical Developer**: Fits your hands-on coding and debugging experience.\n"
            f"* **Systems Analyst / Engineer**: Leverages your structural knowledge of application data flows."
        )

    if any(k in query for k in ("project", "portfolio", "build", "recommend")):
        return (
            f"To strengthen your portfolio for **{role}** roles, I recommend adding projects that use **{gaps_str}**:\n\n"
            f"* **Scalable Backend API**: Build a high-concurrency microservice with automated pipelines.\n"
            f"* **Cloud Deploy Service**: Package an application with Docker, set up configurations, and host it on AWS or GCP.\n"
            f"* **System Optimization Tool**: Identify a slow process, refactor the database schema, and document the latency improvements."
        )

    # General coaching fallback
    return (
        f"As your CareerAI Coach, here is my general advice to polish your profile for **{role}**:\n\n"
        f"* **Showcase Impact**: Focus on *how* you solved problems. Use action verbs (Designed, Built, Saved) and add metrics.\n"
        f"* **Bridge Gaps**: Target your key missing skills: **{gaps_str}** in your next project.\n"
        f"* **Practice Mock Interviews**: Go to the Simulator tab to run custom behavioral and technical scenarios for {role}."
    )


def chat_complete(messages: list[dict[str, str]], system: str, stream: bool = False) -> str | Any:
    if stream:
        return stream_complete(messages, system)

    provider = active_provider()
    providers_to_try = []
    if provider != "none":
        providers_to_try.append(provider)

    for p in ["gemini", "chatgpt", "anthropic"]:
        if p not in providers_to_try:
            if p == "gemini" and _config("GEMINI_API_KEY"):
                providers_to_try.append(p)
            elif p == "chatgpt":
                openai_base_url = _config("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                local_openai = openai_base_url.startswith(("http://localhost", "http://127.0.0.1"))
                if _config("OPENAI_API_KEY") or local_openai:
                    providers_to_try.append(p)
            elif p == "anthropic" and _config("ANTHROPIC_API_KEY"):
                providers_to_try.append(p)

    for p in providers_to_try:
        try:
            if p == "chatgpt":
                return _chatgpt_complete(messages, system)
            if p == "gemini":
                return _gemini_complete(messages, system)
            if p == "anthropic":
                return _anthropic_complete(messages, system)
        except Exception:
            continue

    return _offline_local_response(messages, system)


def stream_complete(messages: list[dict[str, str]], system: str) -> Generator[str, None, None]:
    provider = active_provider()
    providers_to_try = []
    if provider != "none":
        providers_to_try.append(provider)

    for p in ["gemini", "chatgpt", "anthropic"]:
        if p not in providers_to_try:
            if p == "gemini" and _config("GEMINI_API_KEY"):
                providers_to_try.append(p)
            elif p == "chatgpt":
                openai_base_url = _config("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                local_openai = openai_base_url.startswith(("http://localhost", "http://127.0.0.1"))
                if _config("OPENAI_API_KEY") or local_openai:
                    providers_to_try.append(p)
            elif p == "anthropic" and _config("ANTHROPIC_API_KEY"):
                providers_to_try.append(p)

    success = False
    for p in providers_to_try:
        try:
            if p == "chatgpt":
                yield from _chatgpt_stream(messages, system)
            elif p == "gemini":
                yield _gemini_complete(messages, system)
            elif p == "anthropic":
                yield from _anthropic_stream(messages, system)
            success = True
            break
        except Exception:
            continue

    if not success:
        yield _offline_local_response(messages, system)


def _friendly_api_error(provider: str, code: int, details: str) -> str:
    message = details
    try:
        data = json.loads(details)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and "message" in err:
                message = str(err["message"]).strip()
            elif "message" in data:
                message = str(data["message"]).strip()
    except Exception:
        pass

    lowered = message.lower()
    if "reported as leaked" in lowered or "api key was reported as leaked" in lowered:
        return (
            f"{provider} rejected this API key because it has been reported as leaked. "
            "Revoke it, create a new key, put the new key in .streamlit/secrets.toml, then restart Streamlit."
        )
    if code in {401, 403}:
        return (
            f"{provider} rejected the configured key with HTTP {code}. "
            "Check that the key is active, unrestricted for this API, and saved in .streamlit/secrets.toml."
        )
    if code == 429:
        if "quota" in lowered or "exhausted" in lowered:
            if provider.lower() == "gemini":
                return (
                    f"{provider} API Error 429 (Resource Exhausted): You have exceeded your API quota. "
                    "If you are on the Gemini Free Tier (limited to 20 requests/day for gemini-3.5-flash), "
                    "please enable pay-as-you-go billing in Google AI Studio to increase limits, or switch to another provider."
                )
            else:
                return (
                    f"{provider} API Error 429 (Resource Exhausted): You have exceeded your API quota. "
                    "Please check your usage limits, billing details, and remaining credits in your OpenAI account dashboard."
                )
        return f"{provider} API Error 429: Rate limit exceeded. Please wait and try again."

    return f"{provider} API error {code}: {message}"
