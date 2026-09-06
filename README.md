# CareerAI Resume Copilot 🚀

CareerAI is a premium, AI-powered resume scanner, coach, and mock interview simulator. It leverages state-of-the-art LLMs (Gemini, Claude, or ChatGPT) to evaluate your resume, analyze skill gaps, calculate ATS score alignment, and run interactive mock interviews with real-time feedback.

---

## 🛠️ Tech Stack & Features

- **Frontend/Backend**: Streamlit (Python)
- **Styling**: Tailored Dark-Mode Glassmorphism (CSS)
- **AI Integrations**: Google Gemini API, OpenAI API, Anthropic API
- **Key Modules**:
  - 📊 **Dashboard & ATS Evaluator**: Extract text and grade formatting, structure, and keyword relevance.
  - 💬 **Interactive Chat Assistant**: ChatGPT/Gemini-styled chat bubble interface with rich markdown lists, bold formatting, and typing animations.
  - 🎤 **AI Mock Interview Simulator**: Multiple-choice structured question flows, Google Search reference triggers, and real-time scored reports across 5 core criteria.

---

## 💻 Local Setup & Development

### 1. Prerequisites
- Python 3.10 or 3.11 installed.

### 2. Installation
Clone the repository and install dependencies:
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration (Secrets)
Create or edit `.streamlit/secrets.toml` in your project root:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
GEMINI_MODEL = "gemini-3.5-flash"

# Optional:
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"
```

### 4. Running the App
```bash
python -m streamlit run app.py
```
The app will launch locally at `http://localhost:8501`.

---

## 🚀 Production Deployment

### 1. Streamlit Community Cloud (Recommended & Free)
Streamlit Cloud is the industry standard for deploying stateful Streamlit applications. It is completely free and connects directly to your GitHub repository.

1. Push your code to a **GitHub repository**.
2. Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**, select your repository, branch, and set the main file path to `app.py`.
4. Under **Advanced Settings**, paste the content of your `.streamlit/secrets.toml` into the **Secrets** text area.
5. Click **Deploy**. Any future commits you push to GitHub will automatically deploy!

### 2. Render / Railway / Docker Deployments
For custom hosting, a production-ready `Dockerfile` is provided in the repository root.

- **To build the Docker image**:
  ```bash
  docker build -t careerai-copilot .
  ```
- **To run the container locally**:
  ```bash
  docker run -p 8501:8501 --env-file .env careerai-copilot
  ```
- **Cloud Hosting**: You can deploy this directory to **Render**, **Railway**, or **Fly.io** directly. Select "Web Service" (Docker environment), and it will build and expose the app automatically.

---

