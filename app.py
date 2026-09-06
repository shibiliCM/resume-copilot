"""
CareerAI - Resume Intelligence Platform
Features:
  - Animated aurora + particle canvas background
  - Gemini + ChatGPT career coaching
  - Project and soft-skill recommendations per target role
  - ATS full report, skill gap, job match engine
  - Score ring, radar chart, skill chips
"""

import streamlit as st
import streamlit.components.v1 as components
import pickle, re, json, math, os, html
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pdfplumber

from utils.resume_parser import parse_resume
from utils.theme import inject_theme
from components.ats_report import render_ats_report
from components.chat import render_chat
from components.mock_interview import render_mock_interview
from utils.llm import _friendly_api_error


# ------------------------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="CareerAI",
    layout="wide",
    page_icon="CA",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------------------------
# ANIMATED BACKGROUND  (canvas + aurora layers)
# ------------------------------------------------------------------------------------------------
PARTICLES_HTML = Path("components/particles.html").read_text(encoding="utf-8")
components.html(PARTICLES_HTML, height=0, scrolling=False)

# ------------------------------------------------------------------------------------------------
# MASTER CSS
# ------------------------------------------------------------------------------------------------
MAIN_CSS = Path("styles/main.css").read_text(encoding="utf-8")
st.markdown(f"<style>{MAIN_CSS}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------------------------
SKILLS_TECH = [
    "python","java","sql","javascript","typescript","c++","c#","go","rust",
    "html","css","react","angular","vue","node.js","django","flask","fastapi",
    "aws","azure","gcp","docker","kubernetes","terraform","ci/cd","linux","git",
    "tensorflow","pytorch","scikit-learn","keras","hugging face","langchain",
    "machine learning","deep learning","nlp","natural language processing",
    "computer vision","mlops","data analysis","data visualization","statistics",
    "excel","power bi","tableau","spark","hadoop","airflow","kafka",
    "postgresql","mongodb","redis","mysql","rest api","graphql",
    "agile","scrum","devops","cybersecurity","cloud computing","microservices",
    "monitoring","etl",
    "business analysis","market research","financial analysis","accounting",
    "crm","salesforce","figma","ux research","wireframing","product strategy",
    "roadmapping","user research","seo","content strategy",
    "ms office","word","powerpoint","outlook","google sheets","google analytics",
    "jira","trello","asana","notion","slack","zoom","postman","swagger",
    "sap","quickbooks","tally","oracle","opera pms","micros","fidelio",
    "onq","holidex","reservation system","property management system",
    "hubspot","wordpress","shopify",
    "pandas","numpy","scipy","matplotlib","seaborn","plotly","jupyter",
    "r programming","matlab","sas","spss","stata",
    "snowflake","bigquery","redshift","databricks","dbt","looker","qlik",
    "github","github actions","gitlab","jenkins","azure devops",
    "bash","shell scripting","powershell","windows server","nginx","apache",
    "express.js","spring boot","laravel","ruby on rails",".net","asp.net",
    "next.js","tailwind css","bootstrap","sass","redux","vite","webpack",
    "flutter","react native","android","ios","swift","kotlin",
]
TECH_SKILL_ALIASES = {
    "python": ["python","python3","py"],
    "java": ["java","core java","java se","java ee"],
    "sql": ["sql","structured query language","sql queries","querying","queries"],
    "javascript": ["javascript","java script","js","ecmascript"],
    "typescript": ["typescript","type script","ts"],
    "c++": ["c++","cpp","c plus plus"],
    "c#": ["c#","c sharp","csharp"],
    "go": ["golang","go language","go programming"],
    "html": ["html","html5"],
    "css": ["css","css3"],
    "react": ["react","react.js","reactjs","react js"],
    "angular": ["angular","angular.js","angularjs"],
    "vue": ["vue","vue.js","vuejs"],
    "node.js": ["node.js","nodejs","node js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi","fast api"],
    "express.js": ["express.js","express js","expressjs","express"],
    "spring boot": ["spring boot","springboot"],
    "laravel": ["laravel"],
    "ruby on rails": ["ruby on rails","rails","ror"],
    ".net": [".net","dotnet","dot net"],
    "asp.net": ["asp.net","asp net"],
    "next.js": ["next.js","nextjs","next js"],
    "tailwind css": ["tailwind","tailwind css"],
    "bootstrap": ["bootstrap"],
    "sass": ["sass","scss"],
    "redux": ["redux","redux toolkit"],
    "vite": ["vite"],
    "webpack": ["webpack"],
    "aws": ["aws","amazon web services","ec2","s3","lambda","rds","cloudfront"],
    "azure": ["azure","microsoft azure"],
    "gcp": ["gcp","google cloud","google cloud platform"],
    "docker": ["docker","docker compose","docker-compose","containerization","containerisation"],
    "kubernetes": ["kubernetes","k8s"],
    "terraform": ["terraform","infrastructure as code","iac"],
    "ci/cd": ["ci/cd","cicd","ci cd","continuous integration","continuous deployment","continuous delivery"],
    "github actions": ["github actions"],
    "gitlab": ["gitlab","gitlab ci"],
    "jenkins": ["jenkins"],
    "azure devops": ["azure devops","ado"],
    "linux": ["linux","unix","ubuntu","centos","red hat","redhat"],
    "bash": ["bash","bash scripting"],
    "shell scripting": ["shell scripting","shell script","shell scripts"],
    "powershell": ["powershell","power shell"],
    "windows server": ["windows server"],
    "git": ["git","version control"],
    "github": ["github","git hub"],
    "tensorflow": ["tensorflow","tensor flow"],
    "pytorch": ["pytorch","py torch"],
    "scikit-learn": ["scikit-learn","scikit learn","sklearn"],
    "keras": ["keras"],
    "hugging face": ["hugging face","huggingface"],
    "langchain": ["langchain","lang chain"],
    "machine learning": [
        "machine learning","machine-learning","ml","predictive modeling","predictive modelling",
        "predictive model","predictive models","model building","model training",
    ],
    "deep learning": ["deep learning","neural networks","neural network"],
    "nlp": ["nlp","natural language processing"],
    "computer vision": ["computer vision","image processing"],
    "mlops": ["mlops","ml ops","model deployment","model monitoring"],
    "data analysis": ["data analysis","data analytics","analytics","eda","exploratory data analysis","data manipulation"],
    "data visualization": ["data visualization","data visualisation","visualization","visualisation","dashboarding"],
    "statistics": ["statistics","statistical analysis","statistical modelling","statistical modeling","hypothesis testing"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],
    "jupyter": ["jupyter","jupyter notebook","jupyter notebooks"],
    "r programming": ["r programming","r studio","rstudio"],
    "matlab": ["matlab"],
    "sas": ["sas"],
    "spss": ["spss"],
    "stata": ["stata"],
    "excel": ["excel","microsoft excel","ms excel","advanced excel","spreadsheets","spreadsheet"],
    "word": ["word","microsoft word","ms word"],
    "powerpoint": ["powerpoint","power point","microsoft powerpoint","ms powerpoint"],
    "outlook": ["outlook","microsoft outlook","ms outlook"],
    "ms office": ["ms office","microsoft office","office suite","office 365","microsoft 365"],
    "google sheets": ["google sheets","g sheets","sheets"],
    "power bi": ["power bi","powerbi","microsoft power bi"],
    "tableau": ["tableau"],
    "looker": ["looker","looker studio","google data studio"],
    "qlik": ["qlik","qlikview","qlik sense"],
    "spark": ["spark","apache spark","pyspark"],
    "hadoop": ["hadoop","apache hadoop"],
    "airflow": ["airflow","apache airflow"],
    "kafka": ["kafka","apache kafka"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery","google bigquery"],
    "redshift": ["redshift","amazon redshift"],
    "databricks": ["databricks"],
    "dbt": ["dbt","data build tool"],
    "postgresql": ["postgresql","postgres","postgre sql"],
    "mongodb": ["mongodb","mongo db"],
    "redis": ["redis"],
    "mysql": ["mysql","my sql"],
    "oracle": ["oracle","oracle database"],
    "rest api": ["rest api","rest apis","restful api","restful apis","api development","api integration","apis"],
    "graphql": ["graphql","graph ql"],
    "postman": ["postman"],
    "swagger": ["swagger","openapi","open api"],
    "agile": ["agile","agile methodology"],
    "scrum": ["scrum","scrum master"],
    "devops": ["devops","dev ops"],
    "cybersecurity": ["cybersecurity","cyber security","information security","infosec"],
    "cloud computing": ["cloud computing","cloud platforms","cloud services"],
    "microservices": ["microservices","micro services"],
    "monitoring": ["monitoring","observability","prometheus","grafana"],
    "etl": ["etl","elt","data pipeline","data pipelines"],
    "business analysis": ["business analysis","requirements analysis","gap analysis","process analysis"],
    "market research": ["market research","competitive research","competitor analysis"],
    "financial analysis": ["financial analysis","financial modeling","financial modelling","valuation","forecasting"],
    "accounting": ["accounting","bookkeeping","general ledger","accounts payable","accounts receivable"],
    "crm": ["crm","customer relationship management"],
    "salesforce": ["salesforce","sales force"],
    "hubspot": ["hubspot","hub spot"],
    "quickbooks": ["quickbooks","quick books"],
    "tally": ["tally","tally erp","tally prime"],
    "sap": ["sap","sap erp"],
    "figma": ["figma"],
    "ux research": ["ux research","user experience research","usability research"],
    "wireframing": ["wireframing","wireframe","wireframes"],
    "product strategy": ["product strategy","product planning"],
    "roadmapping": ["roadmapping","roadmap","product roadmap"],
    "user research": ["user research","user interviews","customer interviews"],
    "seo": ["seo","search engine optimization","search engine optimisation"],
    "content strategy": ["content strategy","content marketing"],
    "jira": ["jira","atlassian jira"],
    "trello": ["trello"],
    "asana": ["asana"],
    "notion": ["notion"],
    "slack": ["slack"],
    "zoom": ["zoom"],
    "google analytics": ["google analytics","ga4","universal analytics"],
    "wordpress": ["wordpress","word press"],
    "shopify": ["shopify"],
    "opera pms": ["opera pms","opera property management system","opera reservation system","opera"],
    "micros": ["micros","oracle micros"],
    "fidelio": ["fidelio"],
    "onq": ["onq","hilton onq"],
    "holidex": ["holidex"],
    "reservation system": ["reservation system","reservation systems","hotel reservation system","ors"],
    "property management system": ["property management system","property management systems","pms"],
    "nginx": ["nginx"],
    "apache": ["apache","apache http server"],
    "flutter": ["flutter"],
    "react native": ["react native"],
    "android": ["android","android development"],
    "ios": ["ios","ios development"],
    "swift": ["swift"],
    "kotlin": ["kotlin"],
}
SKILLS_SOFT = [
    "communication","leadership","teamwork","problem-solving","adaptability",
    "critical thinking","creativity","emotional intelligence","conflict resolution",
    "decision making","collaboration","negotiation","public speaking",
    "active listening","project management","mentoring","strategic thinking",
    "time management","stakeholder management","change management",
]
SOFT_SKILL_ALIASES = {
    "communication": [
        "communication","communications","communicated","communicate","presented","presentation",
        "presenting","written","verbal","documentation","report writing","client relations",
        "customer service","customer satisfaction","customer focused","customer-focused",
        "client-facing","explained","briefed","articulated","storytelling",
    ],
    "leadership": [
        "leadership","led","lead","leading","managed team","supervised","owned","ownership",
        "drove","directed","headed","team lead","leader","team leader","team management",
        "initiative owner","guided","coordinated team",
    ],
    "teamwork": [
        "teamwork","team player","worked with","cross functional","cross-functional",
        "team environment","group project","partnered with","worked closely","team collaboration",
    ],
    "problem-solving": [
        "problem solving","problem-solving","troubleshot","troubleshooting","debugged",
        "resolved issue","resolved issues","root cause","analysed problem","analyzed problem",
        "solved","solution-oriented","issue resolution",
    ],
    "adaptability": [
        "adaptability","adapted","adaptable","flexible","learned quickly","quick learner",
        "fast-paced","changing requirements","new tools","pivoted","handled ambiguity",
    ],
    "critical thinking": [
        "critical thinking","analytical thinking","root cause analysis","evaluated",
        "assessed","prioritized","tradeoff","trade-off","decision rationale","diagnosed",
    ],
    "creativity": [
        "creativity","creative","designed","brainstormed","innovative","innovation",
        "ideated","conceptualized","new approach","original",
    ],
    "emotional intelligence": [
        "emotional intelligence","empathy","empathetic","relationship building",
        "interpersonal","handled sensitive","customer empathy","people skills",
    ],
    "conflict resolution": [
        "conflict resolution","resolved conflict","resolved conflicts","mediated",
        "de-escalated","deescalated","handled complaints","negotiated resolution",
    ],
    "decision making": [
        "decision making","decision-making","made decisions","prioritized","selected",
        "chose","evaluated options","tradeoff","trade-off","judgement","judgment",
    ],
    "collaboration": [
        "collaboration","collaborated","collaborating","partnered","cross functional",
        "cross-functional","stakeholders","worked with","coordinated with","liaised",
    ],
    "negotiation": [
        "negotiation","negotiated","bargained","vendor management","contract discussion",
        "pricing discussion","agreement","aligned expectations",
    ],
    "public speaking": [
        "public speaking","speaker","spoke at","presented to","presentation","demoed",
        "workshop","training session","facilitated session",
    ],
    "active listening": [
        "active listening","listened","gathered requirements","requirements gathering",
        "user interviews","customer feedback","stakeholder feedback","discovery calls",
    ],
    "project management": [
        "project management","managed project","project planning","timeline","milestone",
        "roadmap","delivery","sprint","scrum","kanban","risk management","status reporting",
    ],
    "mentoring": [
        "mentoring","mentor","mentored","coached","trained","onboarded","guided junior",
        "training and development","knowledge sharing","peer support",
    ],
    "strategic thinking": [
        "strategic thinking","strategy","strategic","roadmap","long-term","market analysis",
        "business case","prioritization","competitive analysis","planning",
    ],
    "time management": [
        "time management","deadline","deadlines","prioritized tasks","multitasking",
        "multi-tasking","multi tasker","multi-tasker","skilled multi-tasker",
        "managed workload","on time","time-sensitive","scheduled",
    ],
    "stakeholder management": [
        "stakeholder management","stakeholders","client management","vendor management",
        "executive updates","cross-functional alignment","aligned stakeholders",
        "business partners","client-facing",
    ],
    "change management": [
        "change management","process change","transformation","migration","adoption",
        "rollout","change initiative","trained users","training and development","transitioned",
    ],
}
HIGH_SIGNAL = {
    "python","machine learning","deep learning","pytorch","tensorflow","kubernetes",
    "nlp","natural language processing","computer vision","mlops","aws","docker",
    "sql","react","typescript","fastapi","langchain","hugging face",
}
BASIC_STOP_WORDS = {
    "the","and","for","with","from","that","this","you","your","are","was","were","has","have","had",
    "not","but","all","can","will","our","their","his","her","she","him","they","them","its","into",
    "over","under","about","than","then","also","more","most","such","any","each","other","using",
    "use","used","work","worked","working","experience","experienced","skills","skill","summary",
    "responsibilities","responsibility","education","project","projects","company","team","resume",
    "professional","business","management","service","services","customer","clients","client",
    "including","various","within","across","based","related","years","month","months","year",
}
INSIGHTS_EDA_SAMPLE_SIZE = 350
INSIGHTS_MODEL_SAMPLE_SIZE = 800
ATS_KW = [
    "developed","designed","managed","implemented","improved","optimized",
    "built","led","launched","delivered","reduced","increased","scaled",
    "collaborated","architected","deployed","automated","created","achieved",
]
CERT_KW = ["certified","certification","aws certified","gcp","azure","pmp","cissp",
           "coursera","udemy","udacity","edx","deep learning specialization"]

ROLE_REQUIRED = {
    "data scientist":    ["python","sql","machine learning","statistics","pytorch","tableau","data visualization"],
    "ml engineer":       ["python","docker","kubernetes","mlops","pytorch","aws","git","ci/cd"],
    "software engineer": ["python","git","rest api","sql","docker","agile","javascript"],
    "data analyst":      ["sql","excel","tableau","python","statistics","power bi","data visualization"],
    "devops engineer":   ["docker","kubernetes","terraform","ci/cd","linux","aws","git","monitoring"],
    "frontend developer":["react","javascript","typescript","css","html","git","rest api"],
    "backend developer": ["python","sql","rest api","docker","git","postgresql","redis"],
    "data engineer":     ["python","sql","spark","airflow","kafka","aws","docker","etl"],
    "product manager":   ["product strategy","user research","roadmapping","data analysis","communication","stakeholder management"],
    "business analyst":  ["business analysis","sql","excel","data visualization","communication","stakeholder management"],
    "consultant":        ["business analysis","market research","data analysis","communication","strategic thinking","stakeholder management"],
    "finance analyst":   ["financial analysis","excel","sql","data visualization","accounting","communication"],
    "ux designer":       ["figma","ux research","wireframing","user research","communication","creativity"],
}

ROLE_PROJECTS = {
    "data scientist": [
        "End-to-end ML pipeline (data ingestion â†’ model deployment) on AWS/GCP",
        "NLP sentiment analysis app with real-time dashboard (Streamlit/Gradio)",
        "Kaggle competition writeup with top 10% finish (public notebook)",
        "Time-series forecasting for a real business problem with measurable impact",
        "LLM fine-tuning project using HuggingFace + LoRA with evaluation metrics",
    ],
    "ml engineer": [
        "Production MLOps pipeline: CI/CD, model registry, drift monitoring",
        "Model serving API with FastAPI + Docker + Kubernetes auto-scaling",
        "Feature store implementation using Feast or Tecton",
        "A/B testing framework for model experiments with statistical significance",
        "GPU-optimised training pipeline with mixed precision & distributed training",
    ],
    "software engineer": [
        "Full-stack web app with auth, REST API, database (deployed to cloud)",
        "Open-source CLI tool with 50+ GitHub stars and comprehensive docs",
        "System design project: design and implement a scalable microservice",
        "Real-time chat or notification system using WebSockets",
        "Contribute meaningful PRs to a popular open-source repository",
    ],
    "data analyst": [
        "Interactive Tableau / Power BI dashboard connected to live data source",
        "A/B test analysis with statistical significance write-up",
        "SQL analytics project: build a full data model + business insight report",
        "Python-based EDA on a public dataset with published findings",
        "Cohort analysis / funnel analysis for a SaaS-style dataset",
    ],
    "frontend developer": [
        "Pixel-perfect, accessible design system component library (React + Storybook)",
        "Performance-optimised SPA with Lighthouse score 95+",
        "Real-time collaborative tool (like Figma Jam) using WebSockets",
        "Micro-frontend architecture demo with Module Federation",
        "Web animation showcase using Framer Motion / GSAP",
    ],
    "devops engineer": [
        "GitOps workflow: full IaC with Terraform + ArgoCD on Kubernetes",
        "Observability stack: Prometheus + Grafana + Loki from scratch",
        "Zero-downtime deployment pipeline with blue-green strategy",
        "Chaos engineering experiment with documented RCA",
        "Security hardening runbook + automated CIS benchmark scanning",
    ],
    "backend developer": [
        "Production REST API with authentication, PostgreSQL, Redis caching, and tests",
        "Event-driven order processing service using queues and idempotent workers",
        "API observability project with structured logs, tracing, metrics, and alerts",
        "Database performance project: indexing, query plans, migrations, and load tests",
        "Secure file-processing backend with background jobs and role-based access",
    ],
    "data engineer": [
        "Batch ETL pipeline from raw CSV/API data to warehouse-ready tables",
        "Streaming analytics pipeline with Kafka, Spark, and dashboard outputs",
        "Airflow orchestration project with data quality checks and lineage notes",
        "Lakehouse-style project with partitioning, schema evolution, and validation",
        "Cost-optimised cloud data pipeline with monitoring and retry strategy",
    ],
    "product manager": [
        "Product discovery case study with user research, personas, and prioritised roadmap",
        "Experiment plan for onboarding conversion with metrics, A/B design, and launch plan",
        "PRD for a real feature including tradeoffs, success metrics, and rollout risks",
        "Competitive analysis plus positioning strategy for a SaaS product",
        "Analytics dashboard translating product usage data into roadmap decisions",
    ],
    "business analyst": [
        "Requirements-to-dashboard project for a real business workflow",
        "SQL-backed KPI analysis with stakeholder-ready insight report",
        "Process improvement case study with current-state map and ROI estimate",
        "Customer segmentation analysis with recommendations for sales or marketing",
        "Executive dashboard with drilldowns, assumptions, and decision notes",
    ],
    "consultant": [
        "Market-entry strategy case study with research, sizing, risks, and recommendation",
        "Operations improvement project with process map, bottleneck analysis, and ROI model",
        "Client-ready diagnostic dashboard with insights, options, and implementation roadmap",
        "Competitive benchmarking report with pricing, positioning, and strategic actions",
        "Change-management plan for a real team or business transformation scenario",
    ],
    "finance analyst": [
        "Three-statement financial model with scenario analysis and dashboard summary",
        "Company valuation project using comparable companies and DCF assumptions",
        "Budget variance analysis with automated Excel/Python reporting",
        "Portfolio risk dashboard with allocation, drawdown, and performance metrics",
        "Unit economics model for a SaaS or marketplace business",
    ],
    "ux designer": [
        "End-to-end UX case study from user research to clickable Figma prototype",
        "Accessibility redesign of a real product screen with before/after usability notes",
        "Design system component library with usage rules and responsive variants",
        "Usability testing report with findings, severity, and redesign decisions",
        "Mobile onboarding prototype tied to activation metrics and user feedback",
    ],
}

SOFT_SKILL_TIPS = {
    "communication": "Practice writing technical blog posts - they show depth AND communication.",
    "leadership": "Mention any time you mentored someone or drove a decision. Even small counts.",
    "teamwork": "Quantify team impact: 'collaborated with 5-person team to ship X in 3 weeks'.",
    "problem-solving": "Add a 'Challenges' section in project descriptions - shows real problem-solving.",
    "adaptability": "Highlight pivots or times you learned a new tool fast under pressure.",
    "critical thinking": "Include your decision rationale in project READMEs - interviewers love it.",
    "project management": "Use STAR format in experience bullets to show ownership end-to-end.",
    "creativity": "Show a new idea, redesign, campaign, prototype, or process improvement you created.",
    "emotional intelligence": "Add examples of customer empathy, stakeholder trust, or sensitive communication.",
    "conflict resolution": "Mention a complaint, disagreement, or blocker you resolved with a clear outcome.",
    "decision making": "Show how you compared options and chose a direction under constraints.",
    "collaboration": "Name the teams or stakeholders you worked with and what shipped because of it.",
    "negotiation": "Include vendor, client, pricing, scope, or expectation-setting examples.",
    "public speaking": "Add demos, presentations, workshops, training sessions, or stakeholder briefings.",
    "active listening": "Show requirements gathering, interviews, feedback loops, or discovery calls.",
    "mentoring": "Mention onboarding, training, peer support, or guiding junior teammates.",
    "strategic thinking": "Tie your work to business goals, market insight, roadmap, or long-term planning.",
    "time management": "Show deadline ownership, prioritization, workload management, or on-time delivery.",
    "stakeholder management": "Add examples of aligning business partners, clients, vendors, or executives.",
    "change management": "Mention rollout, adoption, migration, training users, or process transformation.",
}

ROLE_ICONS = {
    "data scientist":"DS","ml engineer":"ML","data analyst":"DA","software engineer":"SE",
    "devops engineer":"DO","product manager":"PM","frontend developer":"FE",
    "backend developer":"BE","fullstack developer":"FS","data engineer":"DE",
    "business analyst":"BA","consultant":"CO","finance analyst":"FA","ux designer":"UX",
}

ROLE_ALIASES = {
    "data science": "data scientist",
    "machine learning engineer": "ml engineer",
    "frontend engineer": "frontend developer",
    "front end developer": "frontend developer",
    "front-end developer": "frontend developer",
    "backend engineer": "backend developer",
    "back end developer": "backend developer",
    "back-end developer": "backend developer",
    "software developer": "software engineer",
    "business consultant": "consultant",
    "management consultant": "consultant",
    "financial analyst": "finance analyst",
    "ui ux designer": "ux designer",
    "ui/ux designer": "ux designer",
    "product owner": "product manager",
}

# ------------------------------------------------------------------------------------------------
# UTILITIES — pure HTML components
# ------------------------------------------------------------------------------------------------
def score_color(v):
    return "#5effd8" if v>=75 else "#ffd166" if v>=50 else "#ff6b7a"

def svg_ring(value, size=88, stroke=7, color="#5effd8"):
    r=(size-stroke*2)/2; c=size/2
    circ=2*math.pi*r; dash=circ*value/100; gap=circ-dash
    return f"""
    <div style="position:relative;width:{size}px;height:{size}px;display:inline-block;">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg)">
        <circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="{stroke}"/>
        <circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
          stroke-dasharray="{dash:.2f} {gap:.2f}" stroke-linecap="round"/>
      </svg>
      <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                  font-family:'Syne',sans-serif;font-size:{size//4}px;font-weight:800;color:{color};">
        {int(value)}
      </div>
    </div>"""

def xcard(body, glow="rgba(94,255,216,0.06)", extra=""):
    return f"""<div class="xcard fade-up" style="box-shadow:0 8px 40px rgba(0,0,0,0.35),0 0 40px {glow};{extra}">{body}</div>"""

def chip(text, tier="tech"):
    configs={
        "tech":  ("#5effd8","rgba(94,255,216,0.10)","rgba(94,255,216,0.22)"),
        "hi":    ("#5effd8","rgba(94,255,216,0.18)","rgba(94,255,216,0.4)"),
        "soft":  ("#ffd166","rgba(255,209,102,0.10)","rgba(255,209,102,0.22)"),
        "gap":   ("#ff6b7a","rgba(255,107,122,0.10)","rgba(255,107,122,0.25)"),
        "violet":("#a78bfa","rgba(167,139,250,0.10)","rgba(167,139,250,0.25)"),
        "blue":  ("#4da6ff","rgba(77,166,255,0.10)","rgba(77,166,255,0.22)"),
    }
    fg,bg,border=configs.get(tier,configs["tech"])
    return (f'<span style="display:inline-block;margin:3px 2px;padding:5px 12px;'
            f'border-radius:99px;font-size:11.5px;background:{bg};color:{fg};'
            f'border:1px solid {border};transition:all .2s;">{text}</span>')

def mini_bar(label, value, color="#5effd8"):
    return f"""
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;color:rgba(255,255,255,0.5);">{label}</span>
        <span style="font-size:12px;font-weight:600;color:{color};">{int(value)}%</span>
      </div>
      <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:4px;overflow:hidden;">
        <div style="width:{value}%;height:100%;background:{color};border-radius:99px;"></div>
      </div>
    </div>"""

def radar_svg(data, size=200):
    labels=list(data.keys()); vals=[v/100 for v in data.values()]
    n=len(labels); cx=cy=size/2; r=size*.36
    angles=[math.pi*2*i/n-math.pi/2 for i in range(n)]
    def pt(a,rad): return cx+rad*math.cos(a),cy+rad*math.sin(a)
    rings="".join(
        f'<polygon class="radar-ring ring-{int(f*100)}" points="{" ".join(f"{pt(a,r*f)[0]:.1f},{pt(a,r*f)[1]:.1f}" for a in angles)}" '
        f'fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="0.5"/>' for f in [.25,.5,.75,1])
    spokes="".join(f'<line class="radar-spoke" x1="{cx}" y1="{cy}" x2="{pt(a,r)[0]:.1f}" y2="{pt(a,r)[1]:.1f}" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>' for a in angles)
    pts=" ".join(f"{pt(angles[i],r*vals[i])[0]:.1f},{pt(angles[i],r*vals[i])[1]:.1f}" for i in range(n))
    dots="".join(f'<circle class="radar-dot" cx="{pt(angles[i],r*vals[i])[0]:.1f}" cy="{pt(angles[i],r*vals[i])[1]:.1f}" r="3.5" fill="#5effd8"/>' for i in range(n))
    lbls=""
    for i,lbl in enumerate(labels):
        lx,ly=pt(angles[i],r+24); anc="middle"
        if lx<cx-8: anc="end"
        elif lx>cx+8: anc="start"
        lbls+=f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anc}" dominant-baseline="middle" fill="rgba(255,255,255,0.38)" font-size="10" font-family="DM Sans,sans-serif">{lbl}</text>'
    score=sum(data.values())/max(len(data),1)
    return (f'<svg class="advanced-radar" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'{rings}{spokes}<polygon class="radar-shape" points="{pts}" fill="rgba(94,255,216,0.09)" stroke="#5effd8" stroke-width="1.5"/>'
            f'<circle class="radar-core" cx="{cx}" cy="{cy}" r="5" fill="#4da6ff"/>{dots}{lbls}'
            f'<text x="{cx}" y="{cy+25}" text-anchor="middle" fill="rgba(255,255,255,0.55)" font-size="9" font-family="DM Sans,sans-serif">{score:.0f}% avg</text></svg>')

def ats_row(label, ok):
    icon,color,bg=("OK","#5effd8","rgba(94,255,216,0.07)") if ok else ("NO","#ff6b7a","rgba(255,107,122,0.07)")
    return f"""<div style="display:flex;align-items:center;gap:10px;padding:7px 10px;
    border-radius:9px;background:{bg};margin-bottom:5px;">
    <span style="font-weight:700;color:{color};width:16px;">{icon}</span>
    <span style="font-size:12px;color:rgba(255,255,255,0.65);">{label}</span></div>"""

def job_row_html(rank, title, company, pct):
    colors=["#5effd8","#4da6ff","rgba(255,255,255,0.4)"]
    c=colors[min(rank,2)]
    company_html=f'<div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:1px;">{company}</div>' if company else ""
    return f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:12px;padding:10px 12px;margin-bottom:7px;border-left:3px solid {c};">
    <div style="display:flex;align-items:center;gap:9px;">
      {job_badge(title, rank)}
      <div style="flex:1;"><div style="font-size:13px;font-weight:500;color:rgba(255,255,255,0.9);font-family:'Syne',sans-serif;">{title}</div>
      {company_html}</div>
      <div style="text-align:right;">
        <div style="font-size:14px;font-weight:700;color:{c};font-family:'Syne',sans-serif;">{pct:.1f}%</div>
        <div style="width:62px;height:3px;background:rgba(255,255,255,0.08);border-radius:99px;margin-top:4px;overflow:hidden;">
          <div style="width:{int(pct)}%;height:100%;background:{c};border-radius:99px;"></div></div>
      </div>
    </div></div>"""

def job_badge(title, rank=0):
    tl=str(title).lower()
    if any(k in tl for k in ["data","analyst","scientist","bioinformatics"]):
        label, cls = "DATA", "data"
    elif any(k in tl for k in ["engineer","developer","architect","software"]):
        label, cls = "CODE", "code"
    elif any(k in tl for k in ["cloud","devops","security","network"]):
        label, cls = "OPS", "ops"
    elif any(k in tl for k in ["manager","product","lead","business"]):
        label, cls = "PM", "pm"
    else:
        label, cls = "JOB", "job"
    return f'<span class="job-badge job-badge-{cls}" style="animation-delay:{rank*.08}s;">{label}</span>'

# ------------------------------------------------------------------------------------------------
# AI MULTI-PANEL HELPER
# ------------------------------------------------------------------------------------------------
def get_config_value(name: str, default: str = "") -> str:
    """Read config from Streamlit session, secrets, then environment."""
    session_value = st.session_state.get(name, "")
    if session_value:
        return session_value
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    return secret_value or os.getenv(name, default)

def set_config_value(name: str, value: str):
    value = str(value or "").strip()
    if value:
        st.session_state[name] = value

def provider_status(provider: str) -> tuple:
    if provider == "gemini":
        return (
            bool(get_config_value("GEMINI_API_KEY")),
            "Gemini key loaded" if get_config_value("GEMINI_API_KEY") else "Add Gemini API key",
        )
    if provider == "chatgpt":
        base_url = get_config_value("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        is_local = base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1")
        ready = is_local or bool(get_config_value("OPENAI_API_KEY"))
        return (
            ready,
            "OpenAI/local endpoint ready" if ready else "Add OpenAI API key",
        )
    return False, "Provider not configured"


def call_gemini_api(prompt: str) -> str:
    """Call the official Gemini REST API."""
    import urllib.request, urllib.error
    api_key = get_config_value("GEMINI_API_KEY")
    if not api_key:
        return "Add a GEMINI_API_KEY from Google AI Studio, then try again."

    model = get_config_value("GEMINI_MODEL", "gemini-3.5-flash")
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
            parts = data["candidates"][0]["content"]["parts"]
            return "\n".join(part.get("text", "") for part in parts).strip()
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="ignore")
        return _friendly_api_error("Gemini", e.code, details or e.reason)
    except urllib.error.URLError as e:
        return f"Could not reach Gemini API: {e.reason}"
    except Exception as e:
        return f"Gemini coaching error: {e}"


def call_chatgpt_api(prompt: str) -> str:
    """Call OpenAI Chat Completions or an OpenAI-compatible local server."""
    import urllib.request, urllib.error
    base_url = get_config_value("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = get_config_value("OPENAI_API_KEY")
    is_local = base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1")
    if not api_key and not is_local:
        return "Add OPENAI_API_KEY, or set OPENAI_BASE_URL to an OpenAI-compatible local server."

    model = get_config_value("OPENAI_MODEL", "gpt-4o-mini")
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a practical resume and career coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'local'}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="ignore")
        return _friendly_api_error("ChatGPT", e.code, details or e.reason)
    except urllib.error.URLError as e:
        return f"Could not reach ChatGPT/OpenAI endpoint: {e.reason}"
    except Exception as e:
        return f"ChatGPT coaching error: {e}"


def call_ai_provider(provider: str, prompt: str) -> str:
    if provider == "gemini":
        return call_gemini_api(prompt)
    if provider == "chatgpt":
        return call_chatgpt_api(prompt)
    return "Unknown AI provider."


def provider_ready(provider: str) -> bool:
    if provider == "gemini":
        return bool(get_config_value("GEMINI_API_KEY"))
    if provider == "chatgpt":
        base_url = get_config_value("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        is_local = base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1")
        return is_local or bool(get_config_value("OPENAI_API_KEY"))
    return False


def provider_setup_message(provider: str) -> str:
    if provider == "gemini":
        return "Using built-in recommendations because live Gemini coaching is not configured."
    if provider == "chatgpt":
        return "Using built-in recommendations because live ChatGPT coaching is not configured."
    return "Provider is not configured."

def local_coaching_response(persona: str, role: str, tech: list, soft: list,
                            score: float, gap: list, tone_note: str, projects=None) -> str:
    projects = projects or recommend_projects_for_role(normalize_role(role), role, tech, soft, gap, [], "", pd.DataFrame())
    gaps = gap[:4]
    gap_text = ", ".join(gaps) if gaps else "your core role skills are already well covered"
    tech_text = ", ".join(tech[:6]) if tech else "no clear technical skills were detected yet"
    soft_missing = [s for s in SOFT_SKILL_TIPS if s not in soft][:3]
    soft_text = ", ".join(soft_missing) if soft_missing else "communication, leadership, and ownership"
    project_a = projects[0]["title"] if projects else "A role-focused portfolio project with measurable outcomes"
    project_b = projects[1]["title"] if len(projects) > 1 else "A deployed project that mirrors the target job description"

    if persona == "Gemini":
        return (
            f"The candidate is aiming at {role}, and the resume is close enough to shape into a sharper story.\n\n"
            f"Current score: {score}/100. The strongest visible technical signals are {tech_text}. "
            f"The main gap area is {gap_text}. Add those missing keywords naturally inside project bullets, not only in a skills list.\n\n"
            f"Recommended projects: {project_a}; and {project_b}. These help because they turn the missing skills into proof, which is much stronger than simply naming tools.\n\n"
            f"Soft-skill suggestion: show evidence for {soft_text}. Use short bullets with ownership, collaboration, and measurable outcomes."
        )

    return (
        f"For a {role} resume, focus on clarity, keyword coverage, and proof of impact.\n\n"
        f"Detected technical skills: {tech_text}. Priority gaps: {gap_text}. Add 2-3 bullets that connect these skills to results, metrics, or shipped work.\n\n"
        f"Project recommendations: {project_a}; and {project_b}. Put the best one near the top of the resume with tools used, problem solved, and measurable result.\n\n"
        f"Soft skills to strengthen: {soft_text}. The resume should show these through examples, not generic claims."
    )


def build_prompt(persona: str, name: str, role: str, tech: list, soft: list,
                 score: float, gap: list, tone_note: str, projects=None) -> str:
    tech_str  = ", ".join(tech[:12]) or "none detected"
    soft_str  = ", ".join(soft[:6])  or "none detected"
    gap_str   = ", ".join(gap[:5])   or "none - great coverage!"
    projects  = projects or recommend_projects_for_role(normalize_role(role), role, tech, soft, gap, [], "", pd.DataFrame())
    proj_str  = "\n".join(
        f"  {i+1}. {p['title']} - {p['why']}" for i,p in enumerate(projects[:3])
    )

    return f"""You are {name}, an AI career coach. {tone_note}

A candidate's resume has been analysed. Here is their data:
- Target / predicted role: {role}
- Overall resume score: {score}/100
- Technical skills found: {tech_str}
- Soft skills found: {soft_str}
- Key skill gaps: {gap_str}

Suggested projects for this role:
{proj_str}

Your task:
1. Start with a warm, personalised 1-sentence greeting addressed to "the candidate" (not by name).
2. Give 2â€“3 specific, actionable improvement tips for their resume, referencing their actual skill gaps.
3. Recommend 2 projects from the list above (explain WHY each one fits their role + gaps).
4. Suggest 2â€“3 soft skills they should develop and how (with concrete examples / resources).
5. End with one motivating sentence.

Keep the tone {tone_note}. Use short paragraphs. Total response: 220â€“280 words. Do NOT use markdown headers."""


def persona_card_header(cls, emoji, name, tagline, color):
    if cls == "gemini":
        avatar = """
        <div class="ai-avatar gemini-runner" aria-hidden="true">
          <span class="runner-head"></span>
          <span class="runner-body"></span>
          <span class="runner-arm arm-a"></span>
          <span class="runner-arm arm-b"></span>
          <span class="runner-leg leg-a"></span>
          <span class="runner-leg leg-b"></span>
          <span class="runner-trail"></span>
        </div>"""
    elif cls == "chatgpt":
        avatar = """
        <div class="ai-avatar chatgpt-typing" aria-hidden="true">
          <span class="bot-face">C</span>
          <span class="typing-dots"><i></i><i></i><i></i></span>
        </div>"""
    else:
        avatar = f"""
        <div class="ai-avatar" aria-hidden="true">
          <span class="bot-face">{emoji}</span>
        </div>"""
    return f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
      {avatar}
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:{color};">{name}</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.35);">{tagline}</div>
      </div>
    </div>"""


# ------------------------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        model      = pickle.load(open("models/model.pkl","rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))
        jobs_df    = pd.read_csv("data/jobs.csv")
        return model, vectorizer, jobs_df
    except Exception as e:
        return None, None, pd.DataFrame()

def pdf_literal_unescape(value):
    replacements = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
        "(": "(",
        ")": ")",
        "\\": "\\",
    }
    out = []
    escaped = False
    for ch in value:
        if escaped:
            out.append(replacements.get(ch, ch))
            escaped = False
        elif ch == "\\":
            escaped = True
        else:
            out.append(ch)
    if escaped:
        out.append("\\")
    return "".join(out)

def extract_pdf_tagged_text(file):
    """Fallback for tagged/Canva-style PDFs where useful text lives in /E or /T strings."""
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        if hasattr(file, "getvalue"):
            raw = file.getvalue()
        else:
            raw = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
    except Exception:
        return ""

    data = raw.decode("latin-1", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    chunks = []
    for marker in re.finditer(r"/(?:E|T|ActualText)\s*\(", data):
        i = marker.end()
        depth = 1
        escaped = False
        chars = []
        while i < len(data) and depth > 0:
            ch = data[i]
            if escaped:
                chars.append("\\" + ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "(":
                depth += 1
                chars.append(ch)
            elif ch == ")":
                depth -= 1
                if depth > 0:
                    chars.append(ch)
            else:
                chars.append(ch)
            i += 1
        chunk = pdf_literal_unescape("".join(chars)).strip()
        if chunk and len(chunk) > 1:
            chunks.append(chunk)
    return "\n".join(dict.fromkeys(chunks))

def extract_text(file):
    text=""
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t=page.extract_text()
                if t: text+=t+"\n"
    except:
        text=""
    tagged_text = extract_pdf_tagged_text(file)
    combined = "\n".join(part for part in [text, tagged_text] if part.strip())
    return combined

def clean_text(text): return re.sub(r'[^a-zA-Z ]',' ',str(text)).lower()

def normalize_role(role):
    role = re.sub(r"\s+", " ", str(role or "")).strip().lower()
    return ROLE_ALIASES.get(role, role)

def display_role(role):
    role = str(role or "").strip()
    return role.title() if role else ""

def text_has_phrase(text, phrase):
    phrase = str(phrase or "").lower().strip()
    if not phrase:
        return False
    escaped = re.escape(phrase).replace(r"\ ", r"[\s\-]+").replace(r"\-", r"[\s\-]+")
    if re.search(r"[^a-z0-9\s-]", phrase):
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
    else:
        pattern = r"\b" + escaped + r"\b"
    return bool(re.search(pattern, text))

def extract_soft_skills(text):
    tl = str(text or "").lower()
    normalized = re.sub(r"[^a-z0-9+#.\-/]+", " ", tl)
    found = []
    for skill in SKILLS_SOFT:
        aliases = SOFT_SKILL_ALIASES.get(skill, [skill])
        if any(text_has_phrase(normalized, alias) for alias in aliases):
            found.append(skill)
    return found

def extract_tech_skills(text):
    tl = str(text or "").lower()
    normalized = re.sub(r"[^a-z0-9+#.\-/]+", " ", tl)
    skills_to_check = list(dict.fromkeys(SKILLS_TECH + list(TECH_SKILL_ALIASES.keys())))
    found = []
    for skill in skills_to_check:
        aliases = TECH_SKILL_ALIASES.get(skill, [skill])
        if any(text_has_phrase(normalized, alias) for alias in aliases):
            found.append(skill)
    return found

def extract_skills(text):
    tech=extract_tech_skills(text)
    soft=extract_soft_skills(text)
    return tech,soft

def ats_full(text):
    tl=text.lower()
    return {
        "word_count":len(tl.split()),
        "kw_hits":sum(1 for k in ATS_KW if k in tl),
        "cert_hits":sum(1 for c in CERT_KW if c in tl),
        "has_email":bool(re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}',text)),
        "has_phone":bool(re.search(r'\+?\d[\d\s\-().]{7,}\d',text)),
        "has_linkedin":"linkedin" in tl,
        "has_github":"github" in tl,
        "bullets":text.count("â€¢")+text.count("Â·")+text.count("-"),
        "sections":sum(1 for s in ["experience","education","skills","projects","summary"] if s in tl),
        "length_ok":300<len(tl.split())<1000,
    }

def ats_score_breakdown(tech, ats):
    keywords_found = int(ats.get("kw_hits", ats.get("action_keywords", 0)) or 0)
    skills_found = len(tech)
    certs_found = int(ats.get("cert_hits", len(ats.get("certifications", []))) or 0)
    format_checks = [
        {"label": "Email present", "passed": ats.get("has_email", bool(ats.get("email")))},
        {"label": "Phone present", "passed": ats.get("has_phone", bool(ats.get("phone")))},
        {"label": "LinkedIn present", "passed": ats.get("has_linkedin", bool(ats.get("linkedin")))},
        {"label": "GitHub / portfolio present", "passed": ats.get("has_github", bool(ats.get("github") or ats.get("portfolio")))},
        {"label": "Resume length 300-1000 words", "passed": ats.get("length_ok", 300 <= ats.get("word_count", 0) <= 1000)},
        {"label": "At least 3 resume sections", "passed": ats.get("sections", 0) >= 3},
        {"label": "At least 5 bullet points", "passed": ats.get("bullets", ats.get("bullet_count", 0)) >= 5},
    ]
    checks_passed = sum(1 for item in format_checks if item["passed"])
    total_checks = len(format_checks)
    keyword_match = min(keywords_found / 10, 1) * 40
    skill_coverage = min(skills_found / 15, 1) * 30
    formatting = (checks_passed / total_checks) * 20 if total_checks else 0
    certs_bonus = min(certs_found * 5, 15)
    cert_deduction = 0
    components = [
        {"name": "Keyword match", "weight": 40, "raw": min(keywords_found / 10, 1) * 100, "weighted": keyword_match,
         "evidence": f"{keywords_found} action/ATS keywords found"},
        {"name": "Skill coverage", "weight": 30, "raw": min(skills_found / 15, 1) * 100, "weighted": skill_coverage,
         "evidence": f"{skills_found} technical skills detected"},
        {"name": "Formatting rules", "weight": 20, "raw": (checks_passed / total_checks) * 100 if total_checks else 0, "weighted": formatting,
         "evidence": f"{checks_passed}/{total_checks} formatting checks passed"},
        {"name": "Certificate bonus", "weight": 15, "raw": min(certs_found / 3, 1) * 100, "weighted": certs_bonus,
         "evidence": f"{certs_found} certificate signals found"},
    ]
    return {
        "score": round(min(keyword_match + skill_coverage + formatting - cert_deduction + certs_bonus, 100), 1),
        "components": components,
        "format_checks": format_checks,
    }

def compute_scores(tech, soft, max_match, ats, w=(35,35,20,10)):
    sw,mw,aw,sow=w
    sk=min(len(tech)/12,1.0)*100; sf=min(len(soft)/6,1.0)*100
    at=ats_score_breakdown(tech, ats)["score"]
    weight_sum=max(sw+mw+aw+sow,1)
    total=(sk*sw+max_match*mw+at*aw+sf*sow)/weight_sum
    total=max(0,min(total,100))
    return {"overall":round(total,1),"skills":round(sk,1),"soft":round(sf,1),"ats":round(at,1),"match":round(max_match,1)}

def role_fit_recommendations(all_found, prediction, target_role_key="", limit=5):
    found = set(all_found)
    recs = []
    predicted_key = normalize_role(prediction)
    for role, required in ROLE_REQUIRED.items():
        present = [s for s in required if s in found]
        missing = [s for s in required if s not in found]
        score = (len(present) / max(len(required), 1)) * 100
        if role == predicted_key:
            score += 12
        if target_role_key and role == target_role_key:
            score += 8
        recs.append({
            "role": role,
            "score": min(round(score, 1), 100),
            "present": present,
            "missing": missing,
        })
    return sorted(recs, key=lambda r: r["score"], reverse=True)[:limit]

def project_deliverable_for_role(role_key):
    if role_key in ["data scientist", "ml engineer"]:
        return "GitHub repo, clean README, deployed demo, model metrics, and short error analysis"
    if role_key in ["data analyst", "business analyst", "consultant", "finance analyst"]:
        return "dashboard or spreadsheet model, insight memo, assumptions, and executive summary"
    if role_key in ["frontend developer", "ux designer"]:
        return "case study, live prototype, accessibility notes, screenshots, and design decisions"
    if role_key in ["backend developer", "software engineer", "data engineer", "devops engineer"]:
        return "working service, tests, architecture diagram, deployment notes, and monitoring evidence"
    if role_key == "product manager":
        return "PRD, research notes, prioritised roadmap, metrics, and launch-risk checklist"
    return "portfolio write-up, measurable outcome, screenshots, and implementation notes"

def resume_domain_hint(resume_text):
    text = str(resume_text or "").lower()
    domains = [
        ("healthcare", ["healthcare","hospital","patient","clinical","medical"]),
        ("finance", ["finance","banking","accounting","investment","portfolio","loan"]),
        ("education", ["teacher","student","education","curriculum","training"]),
        ("sales/business", ["sales","customer","crm","revenue","business development"]),
        ("marketing", ["marketing","campaign","seo","content","brand"]),
        ("operations", ["operations","supply chain","inventory","logistics","process"]),
        ("technology", ["software","developer","cloud","api","database","machine learning"]),
    ]
    for label, keywords in domains:
        if any(k in text for k in keywords):
            return label
    return "real-world"

def recommend_projects_for_role(role_key, active_role, tech, soft, gap, present, resume_text, top_jobs, limit=5):
    templates = ROLE_PROJECTS.get(role_key, ROLE_PROJECTS.get("software engineer", []))
    domain = resume_domain_hint(resume_text)
    best_job = ""
    if top_jobs is not None and not top_jobs.empty:
        best_job = str(top_jobs.iloc[0].get("Job_Title", "") or "")

    recommendations = []
    for i, title in enumerate(templates):
        title_l = title.lower()
        gap_hits = [s for s in gap if s in title_l][:3]
        if not gap_hits:
            gap_hits = gap[:2]
        anchor_hits = [s for s in present if s in title_l][:3]
        if not anchor_hits:
            anchor_hits = tech[:2] or soft[:2]
        focus_skills = []
        for skill in gap_hits + anchor_hits:
            if skill and skill not in focus_skills:
                focus_skills.append(skill)

        if gap_hits and anchor_hits:
            why = f"Builds on your existing {', '.join(anchor_hits[:2])} while closing {', '.join(gap_hits[:2])} for {active_role}."
        elif gap_hits:
            why = f"Targets missing {active_role} skills: {', '.join(gap_hits[:3])}."
        elif anchor_hits:
            why = f"Turns your current {', '.join(anchor_hits[:2])} experience into stronger portfolio proof for {active_role}."
        else:
            why = f"Creates practical proof for {active_role} using a realistic business problem."

        if best_job:
            why += f" It also lines up with job-market signals like {best_job}."
        if domain != "real-world":
            why += f" Use a {domain} problem or dataset so the project connects back to the uploaded CV."

        recommendations.append({
            "title": title,
            "why": why,
            "build": project_deliverable_for_role(role_key),
            "skills": focus_skills[:5],
            "priority": i + 1,
        })

    if not recommendations:
        recommendations = [{
            "title": f"{active_role} portfolio proof project",
            "why": f"Creates evidence for the role using the skills and gaps detected in the uploaded CV.",
            "build": project_deliverable_for_role(role_key),
            "skills": (gap[:3] + present[:2])[:5],
            "priority": 1,
        }]
    return recommendations[:limit]

def project_card_html(project, index, compact=False):
    colors = ["#5effd8","#4da6ff","#a78bfa","#ffd166","#ff9f6b"]
    color = colors[index % len(colors)]
    skills_html = "".join(chip(html.escape(s), "gap" if i < 2 else "blue") for i, s in enumerate(project.get("skills", [])))
    body_extra = "" if compact else f"""
      <div style="font-size:11px;color:rgba(255,255,255,0.42);line-height:1.55;margin-top:8px;">Deliverable: {html.escape(project.get('build',''))}</div>
      <div style="line-height:2;margin-top:8px;">{skills_html}</div>
    """
    return xcard(f"""
    <div style="font-size:22px;margin-bottom:8px;color:{color};font-family:'Syne',sans-serif;">{index+1}</div>
    <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:{color};margin-bottom:6px;">{html.escape(project.get('title','Project'))}</div>
    <div style="font-size:12px;color:rgba(255,255,255,0.65);line-height:1.65;">{html.escape(project.get('why',''))}</div>
    {body_extra}
    """, glow=f"rgba(94,255,216,{0.04+index*0.01})")

def role_fit_cards_html(recs):
    rows = []
    for rec in recs[:3]:
        color = score_color(rec["score"])
        present = ", ".join(rec["present"][:3]) or "resume signal is still light"
        missing = ", ".join(rec["missing"][:3]) or "few obvious gaps"
        rows.append(f"""
        <div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:10px 12px;margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
            <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#fff;">{html.escape(display_role(rec['role']))}</div>
            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:{color};">{int(rec['score'])}%</div>
          </div>
          <div style="font-size:11px;color:rgba(255,255,255,0.42);line-height:1.55;">Matches: {html.escape(present)}</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.32);line-height:1.55;">Next gaps: {html.escape(missing)}</div>
        </div>
        """)
    return "".join(rows)

def radar_focus_details(focus, scores, tech, soft, gap, ats, top_jobs, role):
    value = scores[focus.lower()]
    color = score_color(value)
    status = "Strong" if value >= 75 else "Improving" if value >= 50 else "Needs attention"

    if focus == "Skills":
        missing = ", ".join(gap[:4]) if gap else "No major role gaps found"
        detail = f"{len(tech)} technical skills detected. Missing for {role}: {missing}."
        action = "Add 2-3 role-specific skills in your skills section and back each one with a project or experience bullet."
    elif focus == "ATS":
        missing_checks = []
        if not ats["has_email"]: missing_checks.append("email")
        if not ats["has_phone"]: missing_checks.append("phone")
        if not ats["has_linkedin"]: missing_checks.append("LinkedIn")
        if not ats["has_github"]: missing_checks.append("GitHub/portfolio")
        if not ats["length_ok"]: missing_checks.append("300-1000 words")
        if ats["sections"] < 3: missing_checks.append("clear sections")
        if ats["kw_hits"] < 5: missing_checks.append("action keywords")
        if ats["bullets"] < 5: missing_checks.append("bullet points")
        detail = f"{ats['word_count']} words, {ats['sections']} sections, {ats['kw_hits']} action keywords."
        action = "Fix: " + (", ".join(missing_checks[:4]) if missing_checks else "ATS basics look good; improve impact metrics next.")
    elif focus == "Match":
        best = top_jobs.iloc[0]["Job_Title"] if not top_jobs.empty else "No job data loaded"
        detail = f"Best current job match: {best}."
        action = f"Mirror the keywords from strong {role} job descriptions in your summary, skills, and top project bullets."
    else:
        missing_soft = [s for s in SOFT_SKILL_TIPS if s not in soft][:4]
        detail = f"{len(soft)} soft skills detected."
        action = "Add evidence for: " + (", ".join(missing_soft) if missing_soft else "leadership, communication, and ownership with measurable examples.")

    return {
        "value": value,
        "color": color,
        "status": status,
        "detail": html.escape(str(detail)),
        "action": html.escape(str(action)),
    }

def radar_score_graph(radar_data):
    colors = {
        "Skills": "#5effd8",
        "ATS": "#4da6ff",
        "Match": "#ffd166",
        "Soft": "#ff9f6b",
    }
    rows = []
    for label, value in radar_data.items():
        color = colors.get(label, score_color(value))
        rows.append(f"""
        <div style="display:grid;grid-template-columns:52px 1fr 36px;align-items:center;gap:8px;margin-bottom:9px;">
          <div style="font-size:11px;color:rgba(255,255,255,0.48);">{label}</div>
          <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
            <div style="width:{value}%;height:100%;background:{color};border-radius:99px;box-shadow:0 0 14px {color};"></div>
          </div>
          <div style="font-size:12px;font-weight:800;color:{color};text-align:right;">{int(value)}</div>
        </div>
        """)
    return "".join(rows)

@st.cache_data(show_spinner=False)
def load_resume_dataset():
    try:
        df = pd.read_csv("data/Resume.csv", usecols=["Resume_str", "Category"])
        if "Resume_str" not in df.columns or "Category" not in df.columns:
            return pd.DataFrame()
        df = df[["Resume_str", "Category"]].dropna()
        df["Resume_str"] = df["Resume_str"].astype(str)
        df["Category"] = df["Category"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def build_eda_summary(_df):
    if _df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    df = _df.sample(n=min(len(_df), INSIGHTS_EDA_SAMPLE_SIZE), random_state=42) if len(_df) > INSIGHTS_EDA_SAMPLE_SIZE else _df
    texts = df["Resume_str"].astype(str).str.lower().tolist()
    skill_names = list(dict.fromkeys(
        HIGH_SIGNAL
        | {
            "python","sql","machine learning","data analysis","data visualization","statistics",
            "excel","power bi","tableau","pandas","numpy","matplotlib","seaborn","jupyter",
            "java","javascript","html","css","react","node.js","git","docker","aws",
            "communication","leadership","teamwork","problem-solving","project management",
            "collaboration","time management","stakeholder management",
        }
    ))
    skill_counts = {}
    for skill in skill_names:
        aliases = TECH_SKILL_ALIASES.get(skill, SOFT_SKILL_ALIASES.get(skill, [skill]))
        pattern = "|".join(
            r"(?<![a-z0-9])" + re.escape(alias.lower()).replace(r"\ ", r"[\s\-]+").replace(r"\-", r"[\s\-]+") + r"(?![a-z0-9])"
            for alias in aliases[:8]
            if alias
        )
        if pattern:
            count = sum(1 for text in texts if re.search(pattern, text))
            if count:
                skill_counts[skill] = count

    clean = re.sub(r"[^a-zA-Z+#.]+", " ", " ".join(texts))
    word_counts = {}
    for word in clean.split():
        if len(word) < 3 or word in BASIC_STOP_WORDS or word.isdigit():
            continue
        word_counts[word] = word_counts.get(word, 0) + 1

    skill_df = (
        pd.DataFrame(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True), columns=["Skill", "Frequency"])
        if skill_counts else pd.DataFrame(columns=["Skill", "Frequency"])
    )
    role_df = (
        _df["Category"].value_counts().rename_axis("Role").reset_index(name="Count")
        if "Category" in _df.columns else pd.DataFrame(columns=["Role", "Count"])
    )
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:70]
    return skill_df, role_df, top_words

def model_metrics_payload(y_true, y_pred, train_size, test_size, source):
    labels = sorted(pd.Series(list(y_true) + list(y_pred)).astype(str).unique())
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics_df = pd.DataFrame({
        "Role": labels,
        "Precision": per_class_p,
        "Recall": per_class_r,
        "F1": per_class_f1,
        "Support": support,
    })
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "metrics_df": metrics_df,
        "confusion_matrix": cm_df,
        "train_size": train_size,
        "test_size": test_size,
        "source": source,
    }

@st.cache_data(show_spinner=False)
def evaluate_resume_model(_df):
    if _df.empty or _df["Category"].nunique() < 2:
        return None

    df = _df.sample(n=min(len(_df), INSIGHTS_MODEL_SAMPLE_SIZE), random_state=42) if len(_df) > INSIGHTS_MODEL_SAMPLE_SIZE else _df
    X = df["Resume_str"].astype(str).map(clean_text)
    y = df["Category"].astype(str)
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=800, solver="lbfgs"),
            "Naive Bayes": MultinomialNB(alpha=0.5),
        }
        model_results = {}
        comparison_rows = []
        for model_name, candidate in candidates.items():
            candidate.fit(X_train_vec, y_train)
            y_pred = candidate.predict(X_test_vec)
            result = model_metrics_payload(
                y_test, y_pred,
                train_size=len(X_train),
                test_size=len(X_test),
                source="Fresh 80/20 train/test split",
            )
            result["model_name"] = model_name
            model_results[model_name] = result
            comparison_rows.append({
                "Model": model_name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False).reset_index(drop=True)
        best_model = str(comparison_df.iloc[0]["Model"])
        best = model_results[best_model].copy()
        best["models"] = model_results
        best["comparison_df"] = comparison_df
        best["best_model"] = best_model
        best["source"] = "Logistic Regression vs Naive Bayes on the same 80/20 split"
        return best
    except Exception as e:
        return {"error": str(e)}

def bar_chart_html(df, label_col, value_col, color="#5effd8", limit=18):
    if df is None or df.empty:
        return "<p style='color:rgba(255,255,255,0.3);font-size:13px;'>No chart data available.</p>"
    rows_df = df.head(limit).copy()
    max_value = max(float(rows_df[value_col].max()), 1)
    rows = []
    for _, row in rows_df.iterrows():
        label = html.escape(str(row[label_col]))
        value = float(row[value_col])
        width = max(3, (value / max_value) * 100)
        rows.append(f"""
        <div style="display:grid;grid-template-columns:minmax(96px,160px) 1fr 44px;gap:10px;align-items:center;margin:9px 0;">
          <div style="font-size:11px;color:rgba(255,255,255,0.56);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>
          <div style="height:10px;background:rgba(255,255,255,0.055);border-radius:99px;overflow:hidden;">
            <div style="height:100%;width:{width:.1f}%;background:{color};border-radius:99px;box-shadow:0 0 16px {color};"></div>
          </div>
          <div style="font-size:11px;color:{color};font-weight:800;text-align:right;">{int(value)}</div>
        </div>
        """)
    return "<div style='padding:2px 4px 4px;'>" + "".join(rows) + "</div>"

def model_comparison_html(comparison_df):
    if comparison_df is None or comparison_df.empty:
        return "<p style='color:rgba(255,255,255,0.3);font-size:13px;'>No model comparison available.</p>"
    metric_colors = {
        "Accuracy": "#5effd8",
        "Precision": "#4da6ff",
        "Recall": "#ffd166",
        "F1": "#a78bfa",
    }
    cards = []
    for _, row in comparison_df.iterrows():
        model_name = str(row["Model"])
        is_best = model_name == str(comparison_df.iloc[0]["Model"])
        rows = []
        for metric, color in metric_colors.items():
            value = float(row[metric]) * 100
            rows.append(f"""
            <div style="display:grid;grid-template-columns:74px 1fr 48px;gap:9px;align-items:center;margin:8px 0;">
              <div style="font-size:10px;color:rgba(255,255,255,0.46);">{metric}</div>
              <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{value:.1f}%;background:{color};border-radius:99px;box-shadow:0 0 12px {color};"></div>
              </div>
              <div style="font-size:11px;color:{color};font-weight:800;text-align:right;">{value:.1f}%</div>
            </div>
            """)
        badge = "<span style='font-size:10px;color:#06101f;background:#5effd8;border-radius:999px;padding:3px 8px;font-weight:900;'>BEST</span>" if is_best else ""
        cards.append(f"""
        <div style="padding:14px;border:1px solid {'rgba(94,255,216,0.28)' if is_best else 'rgba(255,255,255,0.07)'};border-radius:14px;background:{'rgba(94,255,216,0.045)' if is_best else 'rgba(255,255,255,0.025)'};">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px;">
            <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:#fff;">{html.escape(model_name)}</div>
            {badge}
          </div>
          {''.join(rows)}
        </div>
        """)
    return f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;'>{''.join(cards)}</div>"

def word_cloud_html(top_words):
    if not top_words:
        return "<p style='color:rgba(255,255,255,0.3);font-size:13px;'>No word cloud data available.</p>"
    max_count = max(count for _, count in top_words) or 1
    colors = ["#5effd8","#4da6ff","#a78bfa","#ffd166","#ff9f6b","#ff6b7a"]
    spans = []
    for i, (word, count) in enumerate(top_words):
        size = 12 + int((count / max_count) * 26)
        color = colors[i % len(colors)]
        spans.append(
            f'<span title="{count}" style="display:inline-block;margin:5px 7px;font-size:{size}px;'
            f'font-family:\'Syne\',sans-serif;font-weight:800;color:{color};opacity:.9;">{html.escape(word)}</span>'
        )
    return "<div style='line-height:1.45;text-align:center;padding:8px 4px;'>" + "".join(spans) + "</div>"

def confusion_matrix_html(cm_df, max_labels=14):
    if cm_df is None or cm_df.empty:
        return "<p style='color:rgba(255,255,255,0.3);font-size:13px;'>No confusion matrix available.</p>"
    display_df = cm_df.iloc[:max_labels, :max_labels]
    max_value = max(int(display_df.values.max()), 1)
    header = "".join(
        f"<th style='padding:6px;color:rgba(255,255,255,0.42);font-size:10px;max-width:78px;overflow:hidden;text-overflow:ellipsis;'>{html.escape(str(c))}</th>"
        for c in display_df.columns
    )
    rows = []
    for idx, row in display_df.iterrows():
        cells = []
        for val in row:
            alpha = 0.08 + 0.58 * (int(val) / max_value)
            cells.append(
                f"<td style='padding:7px;text-align:center;background:rgba(94,255,216,{alpha:.2f});"
                f"color:#fff;border:1px solid rgba(255,255,255,0.04);font-size:11px;'>{int(val)}</td>"
            )
        rows.append(
            f"<tr><th style='padding:6px;color:rgba(255,255,255,0.5);font-size:10px;text-align:right;max-width:90px;overflow:hidden;text-overflow:ellipsis;'>{html.escape(str(idx))}</th>{''.join(cells)}</tr>"
        )
    note = ""
    if len(cm_df) > max_labels:
        note = f"<div style='font-size:11px;color:rgba(255,255,255,0.28);margin-top:8px;'>Showing first {max_labels} classes for readability.</div>"
    return (
        "<div style='overflow:auto;max-width:100%;'>"
        "<table style='border-collapse:collapse;min-width:680px;'>"
        f"<tr><th></th>{header}</tr>{''.join(rows)}</table></div>{note}"
    )

def ats_breakdown_html(breakdown):
    rows = []
    for item in breakdown["components"]:
        name = item["name"]
        raw = item["raw"]
        weighted = item["weighted"]
        evidence = item["evidence"]
        color = score_color(raw)
        rows.append(f"""
        <div style="display:grid;grid-template-columns:150px 1fr 74px 84px;gap:12px;align-items:center;margin:10px 0;">
          <div style="font-size:12px;font-weight:800;color:rgba(255,255,255,0.70);">{html.escape(name)}</div>
          <div>
            <div style="height:9px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
              <div style="height:100%;width:{raw:.1f}%;background:{color};border-radius:99px;box-shadow:0 0 14px {color};"></div>
            </div>
            <div style="font-size:10px;color:rgba(255,255,255,0.32);margin-top:4px;">{html.escape(evidence)}</div>
          </div>
          <div style="font-size:12px;color:{color};font-weight:800;text-align:right;">{raw:.1f}%</div>
          <div style="font-size:12px;color:#5effd8;font-weight:800;text-align:right;">{weighted:.1f} pts</div>
        </div>
        """)
    score = breakdown["score"]
    return f"""
    <div style="font-size:12px;color:rgba(255,255,255,0.44);line-height:1.7;margin-bottom:12px;">
      <strong style="color:#fff;">ATS Score =</strong>
      Keyword match x 40% + Skill coverage x 30% + Formatting rules x 30%.
      Final score: <strong style="color:#5effd8;">{score}/100</strong>.
    </div>
    {''.join(rows)}
    <div style="font-size:11px;color:rgba(255,255,255,0.30);line-height:1.6;margin-top:12px;">
      This makes the score transparent: each component shows its raw score, weight, evidence, and point contribution.
    </div>
    """

def methodology_html():
    steps = [
        ("01", "PDF upload", "The user uploads a resume PDF into the app."),
        ("02", "Text extraction", "The app extracts readable text using pdfplumber and a tagged-PDF fallback for difficult PDFs."),
        ("03", "Cleaning", "Text is normalized by lowercasing and removing noisy symbols so the model sees consistent input."),
        ("04", "Skill detection", "Technical and soft skills are detected with curated keyword and alias dictionaries."),
        ("05", "Role prediction", "The cleaned text is transformed with TF-IDF and passed to the trained classifier to predict the job role."),
        ("06", "Model comparison", "Logistic Regression and Naive Bayes are trained on the same split and compared using accuracy, precision, recall, F1, and confusion matrix."),
        ("07", "Job matching", "The resume and job descriptions are vectorized and compared using cosine similarity."),
        ("08", "ATS scoring", "The ATS score is calculated from keyword match, skill coverage, and formatting/completeness rules."),
        ("09", "Recommendations", "The app uses gaps, predicted role, target role, and job matches to suggest projects and improvements."),
    ]
    cards = []
    for num, title, body in steps:
        cards.append(f"""
        <div style="display:grid;grid-template-columns:38px 1fr;gap:12px;padding:11px 0;border-bottom:1px solid rgba(255,255,255,0.055);">
          <div style="width:32px;height:32px;border-radius:10px;background:rgba(94,255,216,0.08);border:1px solid rgba(94,255,216,0.18);display:flex;align-items:center;justify-content:center;color:#5effd8;font-size:11px;font-weight:900;">{num}</div>
          <div>
            <div style="font-size:13px;font-weight:800;color:rgba(255,255,255,0.78);">{html.escape(title)}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.38);line-height:1.55;margin-top:2px;">{html.escape(body)}</div>
          </div>
        </div>
        """)
    return "<div>" + "".join(cards) + "</div>"

def render_insights_page():
    df = load_resume_dataset()
    st.markdown("""
    <div class="fade-up" style="margin-bottom:1.25rem;">
      <h2 style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#fff;margin:0 0 6px;">
        Dataset & Model Insights
      </h2>
      <p style="font-size:13px;color:rgba(255,255,255,0.35);">
        Explore training-data patterns and check whether the resume classifier is trustworthy.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No training data found. Place `data/Resume.csv` with `Resume_str` and `Category` columns.")
        return

    tab_eda, tab_model, tab_method = st.tabs(["EDA Graphs", "Model Performance", "Methodology"])

    with tab_eda:
        skill_df, role_df, top_words = build_eda_summary(df)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.html(xcard(f"<div style='font-size:11px;color:rgba(255,255,255,.35);'>Resumes</div><div style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:#5effd8;'>{len(df)}</div>"))
        with m2:
            st.html(xcard(f"<div style='font-size:11px;color:rgba(255,255,255,.35);'>Job Roles</div><div style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:#4da6ff;'>{df['Category'].nunique()}</div>"))
        with m3:
            top_role = html.escape(str(role_df.iloc[0]['Role'])) if not role_df.empty else "-"
            st.html(xcard(f"<div style='font-size:11px;color:rgba(255,255,255,.35);'>Top Role</div><div style='font-family:Syne,sans-serif;font-size:18px;font-weight:800;color:#ffd166;'>{top_role}</div>"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top Skills Frequency")
            st.html(xcard(bar_chart_html(skill_df, "Skill", "Frequency", "#5effd8", limit=18), glow="rgba(94,255,216,0.04)"))
        with c2:
            st.markdown("#### Job Role Distribution")
            st.html(xcard(bar_chart_html(role_df, "Role", "Count", "#4da6ff", limit=18), glow="rgba(77,166,255,0.04)"))

        st.markdown("#### Word Cloud")
        st.html(xcard(word_cloud_html(top_words), glow="rgba(167,139,250,0.05)"))
        st.caption(f"EDA uses a fast sample of up to {INSIGHTS_EDA_SAMPLE_SIZE} resumes for skill and word-frequency charts; role distribution uses the full dataset.")

    with tab_model:
        with st.spinner("Calculating model performance..."):
            perf = evaluate_resume_model(df)
        if not perf:
            st.warning("Not enough data to evaluate the model.")
        elif "error" in perf:
            st.error(f"Model evaluation failed: {perf['error']}")
        else:
            st.markdown("#### Model Comparison")
            st.caption("Both models use the same cleaned resume text, TF-IDF features, and 80/20 train-test split.")
            comparison_df = perf.get("comparison_df", pd.DataFrame())
            st.html(xcard(model_comparison_html(comparison_df), glow="rgba(167,139,250,0.05)"))
            if not comparison_df.empty:
                display_compare = comparison_df.copy()
                for col in ["Accuracy", "Precision", "Recall", "F1"]:
                    display_compare[col] = (display_compare[col] * 100).round(1)
                st.dataframe(display_compare, use_container_width=True)

            st.markdown(f"#### Best Model: {perf.get('best_model', perf.get('model_name', 'Classifier'))}")
            p1, p2, p3, p4 = st.columns(4)
            for col, label, value, color in [
                (p1, "Accuracy", perf["accuracy"], "#5effd8"),
                (p2, "Precision", perf["precision"], "#4da6ff"),
                (p3, "Recall", perf["recall"], "#ffd166"),
                (p4, "F1 Score", perf["f1"], "#a78bfa"),
            ]:
                with col:
                    st.html(xcard(f"<div style='font-size:10px;color:rgba(255,255,255,.35);letter-spacing:.08em;text-transform:uppercase;'>{label}</div><div style='font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:{color};'>{value*100:.1f}%</div>"))

            st.caption(f"{perf.get('source', 'Model evaluation')} | Reference rows: {perf['train_size']} | Evaluated rows: {perf['test_size']}")
            st.markdown("#### Precision / Recall by Role")
            display_metrics = perf["metrics_df"].copy()
            for col in ["Precision", "Recall", "F1"]:
                display_metrics[col] = (display_metrics[col] * 100).round(1)
            st.dataframe(display_metrics, use_container_width=True)

            st.markdown("#### Confusion Matrix")
            st.html(xcard(confusion_matrix_html(perf["confusion_matrix"]), glow="rgba(94,255,216,0.04)"))

    with tab_method:
        st.markdown("#### Project Methodology")
        st.html(xcard(methodology_html(), glow="rgba(167,139,250,0.05)"))
        st.info("Presentation pipeline: PDF -> text extraction -> cleaning -> skill detection -> role prediction -> model comparison -> job matching -> ATS scoring -> recommendations.")

# ------------------------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1.25rem;border-bottom:1px solid rgba(94,255,216,0.08);margin-bottom:1rem;">
      <div class="brand-lockup">
        <div class="brand-orbit" aria-hidden="true">
          <span class="orbit-ring"></span>
          <span class="orbit-dot dot-one"></span>
          <span class="orbit-dot dot-two"></span>
          <span class="brand-core">CA</span>
        </div>
        <span style="font-family:'Syne',sans-serif;font-size:19px;font-weight:800;color:#fff;">CareerAI</span>
      </div>
      <div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:.12em;">RESUME INTELLIGENCE</div>
    </div>""", unsafe_allow_html=True)

    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    page = st.radio("Navigation", ["Dashboard","AI Coaches","Job Matches","Skill Analysis","ATS Report", "💬 Chat Assistant", "🎤 Mock Interview", "Insights"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Score weights"):
        sw  = st.slider("Skills",    0,100,35)
        mw  = st.slider("Job match", 0,100,35)
        aw  = st.slider("ATS",       0,100,20)
        sow = st.slider("Soft",      0,100,10)

    role_options = [
        "Use predicted role",
        "Data Scientist",
        "ML Engineer",
        "Software Engineer",
        "Data Analyst",
        "DevOps Engineer",
        "Frontend Developer",
        "Backend Developer",
        "Data Engineer",
        "Product Manager",
        "Business Analyst",
        "Consultant",
        "Finance Analyst",
        "UX Designer",
    ]
    selected_role = st.selectbox("Target role", role_options)
    target_role = "" if selected_role == "Use predicted role" else selected_role

    st.markdown("""
    <div style="margin-top:1rem;padding:12px;background:rgba(94,255,216,0.04);
                border:1px solid rgba(94,255,216,0.10);border-radius:12px;">
      <div style="font-size:10px;color:#5effd8;font-weight:700;letter-spacing:.08em;margin-bottom:6px;">QUICK TIPS</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.3);line-height:1.75;">
        - 1 page, 400-700 words<br>
        - Action verbs + metrics<br>
        - Mirror job description keywords<br>
        - Quantify every achievement
      </div>
    </div>""", unsafe_allow_html=True)

inject_theme(st.session_state.theme)


# ------------------------------------------------------------------------------------------------
# HERO
# ------------------------------------------------------------------------------------------------
st.markdown("""
<div class="fade-up" style="
  padding:2rem 2.2rem;border-radius:24px;margin-bottom:1.5rem;
  background:linear-gradient(135deg,rgba(94,255,216,0.07),rgba(77,166,255,0.05),rgba(167,139,250,0.04));
  border:1px solid rgba(94,255,216,0.12);position:relative;overflow:hidden;">
  <div style="position:absolute;top:-30px;right:-30px;width:200px;height:200px;
              border-radius:50%;background:radial-gradient(circle,rgba(94,255,216,0.08),transparent);pointer-events:none;"></div>
  <div style="font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:#fff;margin-bottom:8px;">
    CareerAI <span style="color:#5effd8;">Intelligence</span>
  </div>
  <div style="font-size:14px;color:rgba(255,255,255,0.42);max-width:640px;line-height:1.75;">
    AI-powered resume analysis | Gemini + ChatGPT coaching | ATS scoring | Semantic job matching | Skill gap intelligence
  </div>
</div>
""", unsafe_allow_html=True)

if "Insights" in page:
    render_insights_page()
    st.stop()

# ------------------------------------------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if uploaded_file is None:
    st.markdown("""
    <div class="fade-up-1" style="text-align:center;padding:3.5rem 1rem;">
      <div style="font-size:42px;margin-bottom:1rem;color:#5effd8;font-family:'Syne',sans-serif;">PDF</div>
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                  color:rgba(255,255,255,0.5);margin-bottom:8px;">Upload your resume to begin</div>
      <div style="font-size:13px;color:rgba(255,255,255,0.2);">
        PDF format | Analysis runs locally | AI coaching uses Gemini and ChatGPT APIs
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------------------------------------------
model, vectorizer, jobs_df = load_models()

with st.spinner("Analysing resumeâ€¦"):
    resume_data = parse_resume(uploaded_file)
    resume_text = resume_data["text"]
    if not resume_text.strip():
        st.error("Could not extract text from PDF. Try a text-based PDF.")
        st.stop()

    cleaned = clean_text(resume_text)
    tech_skills, soft_skills = extract_skills(resume_text)
    resume_data["skills"] = tech_skills + soft_skills
    all_found = tech_skills + soft_skills
    ats = {
        **resume_data,
        "kw_hits": resume_data.get("action_keywords", 0),
        "has_email": bool(resume_data.get("email")),
        "has_phone": bool(resume_data.get("phone")),
        "has_linkedin": bool(resume_data.get("linkedin")),
        "has_github": bool(resume_data.get("github")) or bool(resume_data.get("portfolio")),
        "length_ok": 300 < resume_data.get("word_count", 0) < 1000,
        "sections": resume_data.get("sections", 0),
        "bullets": resume_data.get("bullet_count", 0),
        "bullet_count": resume_data.get("bullet_count", 0),
        "cert_hits": len(resume_data.get("certifications", []))
    }

    prediction="Data Scientist"; confidence=82.0
    top_jobs=pd.DataFrame(); max_match=68.0

    target_role_key = normalize_role(target_role)

    if model and vectorizer and not jobs_df.empty:
        vec=vectorizer.transform([cleaned])
        prediction=model.predict(vec)[0]
        try: confidence=float(max(model.predict_proba(vec)[0]))*100
        except: confidence=78.0
        match_query = cleaned if not target_role_key else f"{cleaned} {target_role_key} {target_role_key} {target_role_key}"
        job_text = (
            jobs_df.get("Job_Title", pd.Series("", index=jobs_df.index)).fillna("").astype(str)
            + " "
            + jobs_df["Description"].fillna("").astype(str)
        )
        jdv=vectorizer.transform(job_text)
        sims=cosine_similarity(vectorizer.transform([match_query]),jdv)[0]
        jobs_df["Match %"]=(sims*100).round(1)
        top_jobs=jobs_df.sort_values("Match %",ascending=False).head(10).reset_index(drop=True)
        max_match=float(top_jobs["Match %"].iloc[0]) if len(top_jobs) else 68.0

    scores=compute_scores(tech_skills,soft_skills,max_match,ats,w=(sw,mw,aw,sow))
    role_key=target_role_key or normalize_role(prediction)
    active_role=display_role(role_key) if target_role else display_role(prediction)
    active_role_html=html.escape(active_role)
    prediction_html=html.escape(str(prediction))
    required=ROLE_REQUIRED.get(role_key,["python","sql","git","docker","communication"])
    gap=[s for s in required if s not in all_found]
    present=[s for s in required if s in all_found]
    role_recommendations=role_fit_recommendations(all_found, prediction, target_role_key)
    project_recommendations=recommend_projects_for_role(
        role_key, active_role, tech_skills, soft_skills, gap, present, resume_text, top_jobs
    )
    radar_data={"Skills":scores["skills"],"ATS":scores["ats"],"Match":scores["match"],"Soft":scores["soft"]}
    weakest_radar_area=min(radar_data, key=radar_data.get)
    ai_context_key=(
        uploaded_file.name, active_role, tuple(tech_skills), tuple(soft_skills), tuple(gap),
        tuple(p["title"] for p in project_recommendations)
    )
    if st.session_state.get("ai_context_key") != ai_context_key:
        st.session_state.pop("ai_responses", None)
        st.session_state.pop("ai_status", None)
        st.session_state["ai_context_key"] = ai_context_key

# ------------------------------------------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------------------------------------------
if "Dashboard" in page:

    # Score strip
    c1,c2,c3,c4,c5=st.columns([1.8,1,1,1,1])
    with c1:
        oc=score_color(scores["overall"])
        st.html(xcard(f"""
        <div style="display:flex;align-items:center;gap:16px;">
          <div style="animation:pulseGlow 3s infinite;">{svg_ring(scores['overall'],90,8,oc)}</div>
          <div>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px;">Overall score</div>
            <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:{oc};">{scores['overall']}<span style="font-size:14px;color:rgba(255,255,255,0.25);">/100</span></div>
            <div style="font-size:12px;color:rgba(255,255,255,0.35);margin-top:3px;">{"Excellent" if scores['overall']>=80 else "Good" if scores['overall']>=60 else "Needs work"}</div>
          </div>
        </div>""", glow="rgba(94,255,216,0.10)"))

    for col,(lbl,val,ico,clr) in zip([c2,c3,c4,c5],[
        ("Tech Skills",scores["skills"],"SK","#5effd8"),
        ("ATS Score",scores["ats"],"ATS","#4da6ff"),
        ("Job Match",scores["match"],"JOB","#ffd166"),
        ("Soft Skills",scores["soft"],"SOFT","#ff9f6b"),
    ]):
        with col:
            st.html(xcard(f"""
            <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;">{ico} {lbl}</div>
            <div style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:{clr};margin:4px 0;">{int(val)}</div>
            <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:3px;overflow:hidden;">
              <div style="width:{val}%;height:100%;background:{clr};border-radius:99px;"></div>
            </div>"""))

    st.markdown("<br>", unsafe_allow_html=True)

    # Middle row
    ca,cb,cc=st.columns([1.1,1,1.2])
    with ca:
        role_icon=ROLE_ICONS.get(role_key,"ROLE")
        cc_col=score_color(confidence)
        st.html(xcard(f"""
        <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">{"Target role" if target_role else "Role prediction"}</div>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
          <div style="width:46px;height:46px;border-radius:12px;background:rgba(94,255,216,0.09);
                      border:1px solid rgba(94,255,216,0.15);display:flex;align-items:center;justify-content:center;font-size:22px;">{role_icon}</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#fff;">{active_role_html}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:2px;">Predicted <span style="color:{cc_col};font-weight:600;">{prediction_html}</span> ({confidence:.0f}%)</div>
          </div>
        </div>
        {f'<div style="background:rgba(255,209,102,0.08);border:1px solid rgba(255,209,102,0.15);border-radius:9px;padding:7px 10px;font-size:12px;color:#ffd166;margin-bottom:12px;">Analyzing against <strong>{active_role_html}</strong></div>' if target_role else ''}
        {mini_bar("Skill coverage", scores['skills'], "#5effd8")}
        {mini_bar("ATS readiness",  scores['ats'],    "#4da6ff")}
        {mini_bar("Job market fit", scores['match'],  "#ffd166")}
        """, glow="rgba(94,255,216,0.07)"))

    with cb:
        radar_focus = weakest_radar_area
        radar_detail = radar_focus_details(radar_focus, scores, tech_skills, soft_skills, gap, ats, top_jobs, active_role)
        st.html(xcard(f"""
        <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">Performance radar</div>
        <div style="display:flex;justify-content:center;">{radar_svg(radar_data,200)}</div>
        <div style="margin-top:10px;border-top:1px solid rgba(255,255,255,0.06);padding-top:12px;">
          {radar_score_graph(radar_data)}
        </div>
        <div style="font-size:11px;color:rgba(255,255,255,0.28);margin-top:8px;">Lowest metric is highlighted below.</div>
        """))
        st.html(xcard(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;">
          <div>
            <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;">Weakest metric</div>
            <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:800;color:{radar_detail['color']};">{radar_focus}</div>
          </div>
          <div style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:{radar_detail['color']};">{int(radar_detail['value'])}</div>
        </div>
        <div style="font-size:12px;color:{radar_detail['color']};font-weight:700;margin-bottom:7px;">{radar_detail['status']}</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.52);line-height:1.6;margin-bottom:8px;">{radar_detail['detail']}</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.68);line-height:1.6;border-top:1px solid rgba(255,255,255,0.06);padding-top:8px;">{radar_detail['action']}</div>
        """, glow="rgba(167,139,250,0.05)"))

    with cc:
        m_html="".join(job_row_html(i,r.get("Job_Title","Role"),"" if pd.isna(r.get("Company","")) else str(r.get("Company","")),r["Match %"])
                       for i,(_,r) in enumerate(top_jobs.head(3).iterrows())) if not top_jobs.empty else \
               '<p style="color:rgba(255,255,255,0.2);font-size:13px;">Load data/jobs.csv to see matches.</p>'
        st.html(xcard(f"""
        <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">Top job matches</div>
        {m_html}""", glow="rgba(77,166,255,0.07)"))

    st.markdown("<br>", unsafe_allow_html=True)

    # Bottom row â€” skills + ATS + quick recs
    cd,ce,cf=st.columns(3)
    with cd:
        hi_h ="".join(chip(s,"hi")   for s in tech_skills if s in HIGH_SIGNAL)
        rest_h="".join(chip(s,"tech") for s in tech_skills if s not in HIGH_SIGNAL)
        sf_h ="".join(chip(s,"soft") for s in soft_skills[:8])
        st.html(xcard(f"""
        <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">Skills snapshot</div>
        <div style="font-size:10px;color:#5effd8;margin-bottom:4px;">HIGH-SIGNAL</div><div style="line-height:2.3;">{hi_h or '<span style="font-size:12px;color:rgba(255,255,255,0.2);">None detected</span>'}</div>
        <div style="font-size:10px;color:#4da6ff;margin:8px 0 4px;">TECHNICAL</div><div style="line-height:2.3;">{rest_h or '<span style="font-size:12px;color:rgba(255,255,255,0.2);">None</span>'}</div>
        <div style="font-size:10px;color:#ffd166;margin:8px 0 4px;">SOFT SKILLS</div><div style="line-height:2.3;">{sf_h or '<span style="font-size:12px;color:rgba(255,255,255,0.2);">None</span>'}</div>
        <div style="margin-top:10px;font-size:11px;color:rgba(255,255,255,0.2);border-top:1px solid rgba(255,255,255,0.05);padding-top:8px;">{len(tech_skills)} technical | {len(soft_skills)} soft | {sum(1 for s in tech_skills if s in HIGH_SIGNAL)} high-signal</div>
        """))

    with ce:
        checks=[("Email",ats["has_email"]),("Phone",ats["has_phone"]),("LinkedIn",ats["has_linkedin"]),
                ("GitHub",ats["has_github"]),("Good length",ats["length_ok"]),
                ("3+ sections",ats["sections"]>=3),("Action keywords",ats["kw_hits"]>=5),
                ("Bullet points",ats["bullets"]>=5)]
        st.html(xcard(f"""
        <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">ATS checklist</div>
        {"".join(ats_row(l,o) for l,o in checks)}
        <div style="margin-top:8px;font-size:11px;color:rgba(255,255,255,0.2);border-top:1px solid rgba(255,255,255,0.05);padding-top:8px;">{ats['word_count']} words | {ats['sections']} sections | {ats['kw_hits']} keywords</div>
        """, glow="rgba(77,166,255,0.06)"))

    with cf:
        gap_h="".join(chip("+"+s,"gap") for s in gap) or chip("No gaps!","hi")
        projects=project_recommendations[:2]
        proj_h="".join(
            f'<div style="padding:7px 10px;background:rgba(94,255,216,0.05);border-radius:9px;border-left:2px solid #5effd8;margin-bottom:6px;font-size:12px;color:rgba(255,255,255,0.7);line-height:1.5;"><strong style="color:#5effd8;">{html.escape(p["title"])}</strong><br>{html.escape(p["why"])}</div>'
            for p in projects
        )
        st.html(xcard(f"""
        <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">Quick intel</div>
        <div style="font-size:11px;color:#ff6b7a;margin-bottom:5px;">Skill gaps for <em style="color:rgba(255,255,255,0.5);">{active_role_html}</em></div>
        <div style="line-height:2.2;margin-bottom:12px;">{gap_h}</div>
        <div style="font-size:11px;color:#5effd8;margin-bottom:6px;">Suggested projects</div>
        {proj_h}
        """, glow="rgba(255,209,102,0.04)"))


# ------------------------------------------------------------------------------------------------
# AI COACHES  (multi-AI panel)
# ------------------------------------------------------------------------------------------------
elif "AI Coaches" in page:

    st.markdown("""
    <div class="fade-up" style="margin-bottom:1.5rem;">
      <h2 style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#fff;margin:0 0 6px;">
        AI Career Coaches
      </h2>
      <p style="font-size:13px;color:rgba(255,255,255,0.35);">
        Review recommendations, project ideas, and soft-skill suggestions for your target role.
      </p>
    </div>
    """, unsafe_allow_html=True)

    PERSONAS = [
        {
            "id": "gemini",
            "name": "Gemini",
            "provider": "gemini",
            "tagline": "Creative / Lateral / Big-picture",
            "emoji": "G",
            "color": "#4da6ff",
            "tone": "energetic and creative, like a startup founder who thinks laterally and challenges assumptions",
        },
        {
            "id": "chatgpt",
            "name": "ChatGPT",
            "provider": "chatgpt",
            "tagline": "Structured / Direct / Actionable",
            "emoji": "C",
            "color": "#a78bfa",
            "tone": "direct and structured, like a no-nonsense executive coach who gives clear next steps",
        },
    ]

    run_all = st.button(
        "Generate Recommendations",
        type="primary",
        use_container_width=True,
    )

    if run_all or "ai_responses" in st.session_state:
        if run_all:
            st.session_state["ai_responses"] = {}
            st.session_state["ai_status"] = {}
            with st.spinner("Generating recommendations..."):
                for p in PERSONAS:
                    if provider_ready(p["provider"]):
                        prompt = build_prompt(
                            persona=p["name"], name=p["name"],
                            role=active_role,
                            tech=tech_skills, soft=soft_skills,
                            score=scores["overall"], gap=gap,
                            tone_note=p["tone"],
                            projects=project_recommendations,
                        )
                        response_text = call_ai_provider(p["provider"], prompt)
                        response_lower = response_text.lower()
                        error_markers = ("api error", "coaching error", "could not", "add ")
                        is_error = response_lower.startswith(error_markers) or any(marker in response_lower for marker in error_markers[:2])
                        
                        if is_error:
                            fallback_p = "chatgpt" if p["provider"] == "gemini" else "gemini"
                            if provider_ready(fallback_p):
                                fallback_res = call_ai_provider(fallback_p, prompt)
                                fallback_res_lower = fallback_res.lower()
                                if not (fallback_res_lower.startswith(error_markers) or any(marker in fallback_res_lower for marker in error_markers[:2])):
                                    response_text = fallback_res
                                    is_error = False
                                    
                        if is_error:
                            st.session_state["ai_responses"][p["id"]] = local_coaching_response(
                                persona=p["name"],
                                role=active_role,
                                tech=tech_skills,
                                soft=soft_skills,
                                score=scores["overall"],
                                gap=gap,
                                tone_note=p["tone"],
                                projects=project_recommendations,
                            )
                            st.session_state["ai_status"][p["id"]] = "local"
                        else:
                            st.session_state["ai_responses"][p["id"]] = response_text
                            st.session_state["ai_status"][p["id"]] = "ready"
                    else:
                        st.session_state["ai_responses"][p["id"]] = local_coaching_response(
                            persona=p["name"],
                            role=active_role,
                            tech=tech_skills,
                            soft=soft_skills,
                            score=scores["overall"],
                            gap=gap,
                            tone_note=p["tone"],
                            projects=project_recommendations,
                        )
                        st.session_state["ai_status"][p["id"]] = "local"

        cols = st.columns(2)
        for col, p in zip(cols, PERSONAS):
            with col:
                resp = st.session_state["ai_responses"].get(p["id"], "")
                status = st.session_state.get("ai_status", {}).get(p["id"], "waiting")
                header_html = persona_card_header(p["id"],p["emoji"],p["name"],p["tagline"],p["color"])
                status_color = "#5effd8" if status in ["ready","local"] else "#ff6b7a" if status == "error" else "#ffd166"
                status_label = "Generated" if status == "ready" else "Recommendation" if status == "local" else "Needs attention" if status == "error" else "Ready to generate"

                # render response text â€” wrap newlines into paragraph tags
                paras = [x.strip() for x in resp.split("\n") if x.strip()]
                body_html = "".join(
                    f'<p style="font-size:13px;color:rgba(255,255,255,0.65);line-height:1.75;margin-bottom:10px;">{html.escape(para)}</p>'
                    for para in paras
                )

                st.html(f"""
                <div class="ai-panel {p['id']} fade-up">
                  {header_html}
                  <div style="font-size:11px;color:{status_color};font-weight:700;margin:-4px 0 12px;">{status_label}</div>
                  {body_html if body_html else '<div class="loading-shimmer" style="height:200px;"></div>'}
                </div>""")

    else:
        # placeholder cards
        for p, col in zip(PERSONAS, st.columns(2)):
            with col:
                st.html(f"""
                <div class="ai-panel {p['id']} fade-up">
                  {persona_card_header(p['id'],p['emoji'],p['name'],p['tagline'],p['color'])}
                  <div class="loading-shimmer" style="height:20px;margin-bottom:8px;border-radius:8px;"></div>
                  <div class="loading-shimmer" style="height:20px;width:80%;margin-bottom:8px;border-radius:8px;"></div>
                  <div class="loading-shimmer" style="height:20px;margin-bottom:8px;border-radius:8px;"></div>
                  <div class="loading-shimmer" style="height:20px;width:60%;margin-bottom:8px;border-radius:8px;"></div>
                  <div style="text-align:center;padding:1rem 0;font-size:13px;color:rgba(255,255,255,0.2);">
                    Click "Generate Recommendations" to view personalised suggestions
                  </div>
                </div>""")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Role fit panel ----------------------------------------------------------------------------------------
    st.html(f"""
    <div class="fade-up" style="margin:1rem 0 0.75rem;">
      <h3 style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#fff;margin:0;">
        Role Fit From Uploaded CV
      </h3>
      <p style="font-size:12px;color:rgba(255,255,255,0.3);margin-top:4px;">
        These are estimated from detected skills, the model prediction, and your selected target role.
      </p>
    </div>""")
    st.html(xcard(role_fit_cards_html(role_recommendations), glow="rgba(77,166,255,0.05)"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Project recommendations panel -------------------------------------------------------------------------
    st.html(f"""
    <div class="fade-up" style="margin:1rem 0 0.75rem;">
      <h3 style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#fff;margin:0;">
        Recommended Projects - <span style="color:#5effd8;">{active_role_html}</span>
      </h3>
      <p style="font-size:12px;color:rgba(255,255,255,0.3);margin-top:4px;">
        Built from your detected CV skills, missing gaps, selected target role, and current job-match signals. Prioritise the top 2.
      </p>
    </div>""")

    projects = project_recommendations
    pcols = st.columns(min(len(projects), 3))
    for i, (proj, pcol) in enumerate(zip(projects, pcols * 2)):
        with pcol:
            st.html(project_card_html(proj, i))

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Soft skills deep-dive --------------------------------------------------------------------------
    st.html(f"""
    <div class="fade-up" style="margin:0 0 0.75rem;">
      <h3 style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#fff;margin:0;">
        Soft Skills to Develop
      </h3>
    </div>""")

    missing_soft = [s for s in SOFT_SKILL_TIPS if s not in soft_skills][:4]
    present_soft = soft_skills[:4]

    sc1, sc2 = st.columns(2)
    with sc1:
        st.html(xcard(f"""
        <div style="font-size:10px;color:#5effd8;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">Detected in your resume</div>
        {"".join(f'<div style="display:flex;gap:10px;padding:8px 10px;background:rgba(94,255,216,0.05);border-radius:9px;margin-bottom:6px;"><span style="font-size:12px;color:#5effd8;font-weight:700;">OK</span><div style="font-size:12px;color:rgba(255,255,255,0.7);">{s.title()}<br><span style="color:rgba(255,255,255,0.35);font-size:11px;">{SOFT_SKILL_TIPS.get(s,"Keep reinforcing this - show evidence.")}</span></div></div>' for s in present_soft) or "<p style='color:rgba(255,255,255,0.25);font-size:12px;'>None detected yet</p>"}
        """))

    with sc2:
        st.html(xcard(f"""
        <div style="font-size:10px;color:#ff6b7a;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">Recommended to add</div>
        {"".join(f'<div style="display:flex;gap:10px;padding:8px 10px;background:rgba(255,107,122,0.05);border-left:2px solid rgba(255,107,122,0.3);border-radius:9px;margin-bottom:6px;"><span style="font-size:12px;color:#ff6b7a;font-weight:700;">ADD</span><div style="font-size:12px;color:rgba(255,255,255,0.7);">{s.title()}<br><span style="color:rgba(255,255,255,0.35);font-size:11px;">{SOFT_SKILL_TIPS.get(s,"Add concrete examples to your resume.")}</span></div></div>' for s in missing_soft) or "<p style='color:rgba(255,255,255,0.25);font-size:12px;'>Great - you have good soft skill coverage!</p>"}
        """, glow="rgba(255,107,122,0.04)"))


# ------------------------------------------------------------------------------------------------
# JOB MATCHES
# ------------------------------------------------------------------------------------------------
elif "Job Matches" in page:
    if top_jobs.empty:
        st.warning("No job data - place `data/jobs.csv` in the project directory.")
    else:
        st.html("""
        <div class="job-universe">
          <div class="role-orb role-data">DATA</div>
          <div class="role-orb role-ai">AI</div>
          <div class="role-orb role-code">DEV</div>
          <div class="role-orb role-cloud">CLOUD</div>
          <div class="role-orb role-design">UX</div>
          <div class="role-orb role-product">PM</div>
          <div class="role-orb role-ats">ATS</div>
        </div>
        """)
        fc,_,sc=st.columns([2,2,2])
        with fc: min_pct=st.slider("Min match %",0,100,50)
        with sc: sort_by=st.selectbox("Sort by",["Match %","Job_Title"])
        filtered=(top_jobs[top_jobs["Match %"]>=min_pct].sort_values(sort_by,ascending=(sort_by!="Match %")))
        st.markdown(f'<div style="font-size:12px;color:rgba(255,255,255,0.3);margin:.5rem 0 1rem;"><strong style="color:#5effd8;">{len(filtered)}</strong> results above {min_pct}%</div>',unsafe_allow_html=True)
        for i,(_,row) in enumerate(filtered.iterrows()):
            t=row.get("Job_Title","-"); co=row.get("Company","-"); d=row.get("Description","")[:500]; pct=row["Match %"]; clr=score_color(pct)
            with st.expander(f"{i+1}. {t} | {pct:.1f}%"):
                x1,x2=st.columns([3,1])
                with x1:
                    st.html(f"""
                    <div style="display:flex;align-items:flex-start;gap:12px;padding:.4rem 0;">
                      {job_badge(t, i)}
                      <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.7;">{html.escape(str(d))}...</div>
                    </div>""")
                with x2:
                    st.markdown(f'<div style="text-align:center;padding:1rem 0;"><div style="font-family:Syne,sans-serif;font-size:36px;font-weight:800;color:{clr};">{pct:.0f}%</div><div style="font-size:11px;color:rgba(255,255,255,0.3);">match</div></div>',unsafe_allow_html=True)
                    st.progress(int(pct)/100)


# ------------------------------------------------------------------------------------------------
# SKILL ANALYSIS
# ------------------------------------------------------------------------------------------------
elif "Skill Analysis" in page:
    t1,t2,t3=st.tabs(["Technical","Soft Skills","Gap Analysis"])
    with t1:
        hi=[s for s in tech_skills if s in HIGH_SIGNAL]; rest=[s for s in tech_skills if s not in HIGH_SIGNAL]
        x1,x2=st.columns(2)
        with x1:
            st.html(f"""
            <div style="margin-bottom:14px;">
              <div style="font-size:12px;font-weight:700;color:#5effd8;font-family:'Syne',sans-serif;margin-bottom:8px;">High-signal ({len(hi)})</div>
              <div style="line-height:2.4;">{"".join(chip(s,"hi") for s in hi) or '<span style="color:rgba(255,255,255,0.2);font-size:12px;">None detected</span>'}</div>
            </div>
            <div>
              <div style="font-size:12px;font-weight:700;color:#4da6ff;font-family:'Syne',sans-serif;margin-bottom:8px;">Other technical ({len(rest)})</div>
              <div style="line-height:2.4;">{"".join(chip(s,"blue") for s in rest) or '<span style="color:rgba(255,255,255,0.2);font-size:12px;">None detected</span>'}</div>
            </div>""")
        with x2:
            breakdown_html = xcard(f"""
            <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">Breakdown</div>
            {mini_bar("High-signal",min(len(hi)/8*100,100),"#5effd8")}
            {mini_bar("Total technical",min(len(tech_skills)/20*100,100),"#4da6ff")}
            {mini_bar("Soft coverage",min(len(soft_skills)/8*100,100),"#ffd166")}
            <div style="margin-top:14px;display:flex;justify-content:center;">{radar_svg(radar_data,185)}</div>
            """)
            st.html(breakdown_html)

    with t2:
        st.markdown(f'<div style="line-height:2.5;margin-bottom:1rem;">{"".join(chip(s,"soft") for s in soft_skills) or "<p style=color:rgba(255,255,255,0.25);>No soft skills detected.</p>"}</div>',unsafe_allow_html=True)
        st.progress(min(len(soft_skills)/8,1.0),text=f"{len(soft_skills)} / 8 recommended soft skills")

    with t3:
        st.html(f"""
        <div style="font-size:13px;color:rgba(255,255,255,0.4);margin-bottom:1rem;">Gap analysis for <strong style="color:#5effd8;">{active_role_html}</strong></div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1.5rem;">
          <div style="background:rgba(94,255,216,0.08);border:1px solid rgba(94,255,216,0.15);border-radius:12px;padding:12px 20px;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#5effd8;">{len(present)}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.35);">Present</div></div>
          <div style="background:rgba(255,107,122,0.08);border:1px solid rgba(255,107,122,0.15);border-radius:12px;padding:12px 20px;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#ff6b7a;">{len(gap)}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.35);">Missing</div></div>
        </div>
        <div style="font-size:12px;color:#ff6b7a;font-weight:600;margin-bottom:8px;">Skills to add:</div>
        <div style="line-height:2.4;">{"".join(chip("+ "+s,"gap") for s in gap) or chip("All key skills present!","hi")}</div>
        """)


# ------------------------------------------------------------------------------------------------
# ATS REPORT
# ------------------------------------------------------------------------------------------------
elif "ATS Report" in page:
    render_ats_report(ats, tech_skills)

    with st.expander("Raw resume text (first 2500 chars)"):
        st.text_area("Raw text", value=resume_text[:2500], height=280, disabled=True, label_visibility="collapsed")

# ------------------------------------------------------------------------------------------------
# CHAT ASSISTANT
# ------------------------------------------------------------------------------------------------
elif "💬 Chat Assistant" in page:
    # Build complete resume context for the AI
    resume_context = {
        "text": resume_text[:5000],
        "name": ats.get("name", ""),
        "email": ats.get("email", ""),
        "phone": ats.get("phone", ""),
        "skills": tech_skills + soft_skills,
        "experience": ats.get("experience_years", 0),
        "education": ats.get("education", []),
        "certs": ats.get("certifications", []),
        "predicted_role": prediction,
        "skill_gaps": gap,
        "ats_score": ats_score_breakdown(tech_skills, ats)["score"],
        "job_matches": top_jobs.to_dict('records')[:3] if not top_jobs.empty else []
    }
    render_chat(resume_context)

# ------------------------------------------------------------------------------------------------
# MOCK INTERVIEW
# ------------------------------------------------------------------------------------------------
elif "🎤 Mock Interview" in page:
    # Build complete resume context for the AI
    resume_context = {
        "text": resume_text[:5000],
        "name": ats.get("name", ""),
        "skills": tech_skills + soft_skills,
        "experience": ats.get("experience_years", 0),
        "education": ats.get("education", []),
        "certs": ats.get("certifications", []),
        "ats_score": ats_score_breakdown(tech_skills, ats)["score"],
        "predicted_role": prediction,
        "skill_gaps": gap,
    }
    render_mock_interview(resume_context)



