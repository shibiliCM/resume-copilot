# CareerAI Presentation Documentation

## 1. Project Objective

CareerAI is a resume intelligence platform. It analyzes a resume PDF and gives:

- predicted job role
- detected technical and soft skills
- job matching score
- explainable ATS score
- skill gaps for a selected target role
- project recommendations for improving the resume
- model performance comparison

## 2. Methodology

```text
PDF Upload
-> Text Extraction
-> Cleaning
-> Skill Detection
-> Role Prediction
-> Model Comparison
-> Job Matching
-> ATS Scoring
-> Recommendations
```

### Step-by-step method

| Step | Process | Purpose |
|---|---|---|
| 1 | User uploads a resume PDF | Start analysis from a real resume |
| 2 | Extract text using pdfplumber and tagged-PDF fallback | Convert PDF content into readable text |
| 3 | Clean text by lowercasing and removing noisy symbols | Prepare consistent text for ML |
| 4 | Detect technical and soft skills | Understand the candidate profile |
| 5 | Convert text into TF-IDF vectors | Represent resume text numerically |
| 6 | Predict role using a trained classifier | Estimate the best role category |
| 7 | Compare Logistic Regression and Naive Bayes | Evaluate which model performs better |
| 8 | Match resume with job descriptions using cosine similarity | Recommend closest jobs |
| 9 | Calculate ATS score | Explain resume strength for ATS systems |
| 10 | Generate recommendations | Suggest skill gaps and projects |

## 3. EDA

EDA means Exploratory Data Analysis. It helps understand the dataset before model training.

### Dataset summary

- Resumes: total resume records in the dataset.
- Job Roles: number of different role categories.
- Top Role: most common role category in the dataset.

### EDA graphs added

- Top skills frequency
- Job role distribution
- Word cloud

### Word cloud meaning

The bigger the word, the more often it appears in the resume dataset.

Presentation line:

> The word cloud provides a quick visual summary of common terms in the resume dataset. Larger words appear more frequently.

## 4. Model Comparison

The app trains and compares two models:

1. Logistic Regression
2. Naive Bayes

Both models use:

- same cleaned resume text
- same TF-IDF features
- same train/test split
- same evaluation metrics

### Metrics used

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted roles were actually correct |
| Recall | How many actual roles were successfully found |
| F1 Score | Balance between precision and recall |
| Confusion Matrix | Shows correct and incorrect class predictions |

Presentation line:

> Logistic Regression and Naive Bayes are trained on the same resume dataset and compared using accuracy, precision, recall, F1 score, and confusion matrix. This helps choose the better model for resume role classification.

## 5. ATS Score Definition

The ATS score is transparent and explainable.

```text
ATS Score = Keyword Match x 40%
          + Skill Coverage x 30%
          + Formatting Rules x 30%
```

### Component explanation

| Component | Weight | What it checks |
|---|---:|---|
| Keyword Match | 40% | Action verbs and ATS-friendly keywords |
| Skill Coverage | 30% | Detected technical skills in the resume |
| Formatting Rules | 30% | Email, phone, LinkedIn, GitHub, sections, bullets, length, certifications |

Presentation line:

> The ATS score is not a black box. It is calculated using a weighted formula: 40% keyword match, 30% skill coverage, and 30% formatting rules. The user can see exactly why they received the score.

## 6. Recommendation Logic

Recommendations are based on:

- predicted role
- selected target role
- detected skills
- missing skills
- job matching results
- role-specific project templates

Example:

```text
If target role = Data Scientist
and missing skills = machine learning, statistics, visualization
then recommend projects involving prediction models, EDA dashboards, and model evaluation.
```

## 7. Full Presentation Script

### Opening

CareerAI is a resume intelligence platform that analyzes a PDF resume and gives career-focused insights.

### Methodology

The system follows a clear pipeline: PDF upload, text extraction, cleaning, skill detection, role prediction, model comparison, job matching, ATS scoring, and recommendations.

### EDA

Before model evaluation, the system shows EDA charts such as top skills frequency, job role distribution, and word cloud.

### Model performance

The system compares Logistic Regression and Naive Bayes. Both models are trained on the same dataset split and evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

### ATS score

The ATS score is transparent. It is calculated as a weighted sum of keyword match, skill coverage, and formatting rules.

### Conclusion

CareerAI does not only predict a job role. It also explains skill gaps, gives ATS feedback, and recommends real-world projects to improve the resume.

## 8. Limitations

| Limitation | Future improvement |
|---|---|
| Scanned PDFs may not extract correctly | Add OCR |
| Word cloud may include common non-skill words | Add stronger stop-word filtering |
| Keyword extraction may miss unusual wording | Add NLP phrase extraction |
| ATS weights are custom | Validate weights using recruiter feedback |

## 9. Files to Present

- App: `app.py`
- Dataset: `data/Resume.csv`
- Jobs: `data/jobs.csv`
- Models: `models/model.pkl`, `models/vectorizer.pkl`
- Presentation document: `presentation_docs/CareerAI_Presentation_Documentation.html`

