# Updated and fixed resume analyzer
import streamlit as st

# ========== PAGE CONFIG (MUST BE FIRST) ==========
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    layout="wide",
    page_icon="📄",
    menu_items={
        'Get Help': 'https://github.com/yourusername/resume-analyzer',
        'Report a bug': "https://github.com/yourusername/resume-analyzer/issues",
        'About': "# Advanced Resume Analyzer with Continuous Learning"
    }
)

# ========== IMPORTS (AFTER PAGE CONFIG) ==========
import spacy
import pdfplumber
import docx
import re
import pickle
import os
import sys
import subprocess
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ========== NLP INITIALIZATION ==========
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading language model... (one-time setup)")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to load NLP model: {str(e)}")
            st.stop()

nlp = load_nlp_model()

# ========== DATA STORAGE ==========
DATA_DIR = "resume_analyzer_data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_history():
    try:
        history_path = os.path.join(DATA_DIR, "analysis_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Error loading history: {str(e)}")
    return {"resumes": [], "jobs": [], "matches": [], "timestamps": []}

def save_history(data):
    try:
        temp_path = os.path.join(DATA_DIR, "temp_history.pkl")
        history_path = os.path.join(DATA_DIR, "analysis_history.pkl")
        
        with open(temp_path, "wb") as f:
            pickle.dump(data, f)
        
        if os.path.exists(temp_path):
            os.replace(temp_path, history_path)
    except Exception as e:
        st.warning(f"Error saving history: {str(e)}")

# ========== FILE PROCESSING ==========
def extract_text(file):
    if not file:
        return ""
    
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                text = []
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text() or page.crop(page.mediabox).extract_text()
                        text.append(page_text or "")
                    except:
                        text.append("")
                return " ".join(text).strip()
                
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return " ".join(para.text for para in doc.paragraphs if para.text)
    except Exception as e:
        st.error(f"File reading error: {str(e)}")
    return ""

# ========== SKILLS DATABASE ==========
SKILLS_DATABASE = {
    "technical": {
        "programming": ["python", "java", "javascript", "sql", "c++"],
        "cloud": ["aws", "azure", "docker", "kubernetes"],
        "data": ["pandas", "numpy", "machine learning"]
    },
    "soft": {
        "communication": ["presentation", "writing", "public speaking"],
        "management": ["leadership", "teamwork", "project management"]
    }
}

# ========== ANALYSIS FUNCTIONS ==========
def extract_skills(text):
    if not text:
        return defaultdict(set)
    
    try:
        text_lower = text.lower()
        found_skills = defaultdict(set)
        
        for category in SKILLS_DATABASE:
            if isinstance(SKILLS_DATABASE[category], dict):
                for subcategory, skills in SKILLS_DATABASE[category].items():
                    for skill in skills:
                        if skill in text_lower:
                            found_skills[category].add(skill)
        return found_skills
    except Exception as e:
        st.warning(f"Skill extraction error: {str(e)}")
        return defaultdict(set)

def analyze(resume_text, job_text):
    try:
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        
        results = {}
        for category in ["technical", "soft"]:
            job_set = job_skills.get(category, set())
            resume_set = resume_skills.get(category, set())
            
            match_pct = len(job_set & resume_set) / max(1, len(job_set)) * 100
            results[f"{category}_match"] = round(match_pct, 1)
            results[f"missing_{category}"] = sorted(job_set - resume_set)
        
        history = load_history()
        history["resumes"].append(resume_text)
        history["jobs"].append(job_text)
        history["matches"].append(results["technical_match"] / 100)
        history["timestamps"].append(datetime.now())
        save_history(history)
        
        return results
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            "technical_match": 0,
            "soft_match": 0,
            "missing_technical": [],
            "missing_soft": []
        }

# ========== MAIN APP ==========
def main():
    st.title("📄 Professional Resume Analyzer")
    st.write("""
    **AI-powered resume analysis with continuous learning**
    - 🛠️ Technical skill matching
    - 🧠 Soft skill evaluation
    - 📈 Gets smarter with each analysis
    """)
    
    with st.expander("How to use"):
        st.write("""
        1. Paste job description
        2. Upload resume (PDF/DOCX)
        3. Click Analyze
        4. View your matches
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        job_desc = st.text_area("Job Description:", height=250)
    with col2:
        resume_file = st.file_uploader("Upload Resume:", type=["pdf", "docx"])
    
    if st.button("Analyze Resume", type="primary"):
        if not job_desc or not resume_file:
            st.warning("Please provide both job description and resume")
        else:
            with st.spinner("Analyzing..."):
                resume_text = extract_text(resume_file)
                if not resume_text:
                    st.error("Failed to extract text from resume")
                else:
                    analysis = analyze(resume_text, job_desc)
                    
                    st.success("## Analysis Results")
                    cols = st.columns(2)
                    cols[0].metric("Technical Match", f"{analysis['technical_match']}%")
                    cols[1].metric("Soft Skills", f"{analysis['soft_match']}%")
                    
                    st.subheader("Improvement Suggestions")
                    for category in ["technical", "soft"]:
                        if analysis[f"missing_{category}"]:
                            st.write(f"**Missing {category} skills:**")
                            st.write(", ".join(analysis[f"missing_{category}"]))
                    
                    history = load_history()
                    st.caption(f"System has learned from {len(history['resumes'])} analyses")

if __name__ == "__main__":
    main()
