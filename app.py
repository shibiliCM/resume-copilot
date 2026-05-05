import streamlit as st
import pickle
import pdfplumber
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Career Copilot")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
jobs_df = pd.read_csv("jobs.csv")

# Extract text from PDF
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Clean text (same style as training)
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    return text

# skill detection
skills_list = [
    "python", "machine learning", "data analysis",
    "sql", "tensorflow", "pandas", "numpy",
    "deep learning", "nlp", "excel", "tableau",
    "power bi", "java", "c++", "html", "css"
]

def extract_skills(text):
    found = []
    text = text.lower()
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found



# Upload
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

def calculate_score(skills, max_match):
    # weight settings
    skill_weight = 0.5
    match_weight = 0.5

    # skill score (normalize by expected skill count)
    expected_skills = 10  # you can tweak this
    skill_score = min(len(skills) / expected_skills, 1.0) * 100

    # match score already in %
    match_score = max_match  # e.g., 82.3

    # final weighted score
    final = (skill_score * skill_weight) + (match_score * match_weight)
    return round(final, 2)

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

    st.subheader("Resume Preview")
    st.write(resume_text[:500])
 
    # Extract skills
    skills = extract_skills(resume_text)

    st.subheader("Detected Skills")
    st.write(skills)

    # Prediction
    cleaned = clean_text(resume_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    st.subheader("Predicted Job Role")
    st.success(prediction)
    
    # Job Matching
    job_descriptions = jobs_df["Description"].fillna("")
    job_vectors = vectorizer.transform(job_descriptions)
    resume_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(resume_vector, job_vectors)[0]
    jobs_df["Match %"] = (scores * 100).round(2)
    top_jobs = jobs_df.sort_values(by="Match %", ascending=False).head(3)

    st.subheader("Top Job Matches")
    st.write(top_jobs[["Job_Title", "Match %"]])
    
    # compute score
    max_match = top_jobs["Match %"].iloc[0] if len(top_jobs) > 0 else 0
    score = calculate_score(skills, max_match)

    st.subheader("Resume Score")
    st.metric(label="Score", value=f"{score}/100")
    
    if score < 50:
        st.warning("Improve your resume by adding more relevant skills.")
    elif score < 75:
        st.info("Good resume. Add a few more strong keywords.")
    else:
        st.success("Strong resume for this role!")