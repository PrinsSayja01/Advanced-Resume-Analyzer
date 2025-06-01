# professional_resume_analyzer.py
import streamlit as st

# ========== STREAMLIT CONFIG (MUST BE FIRST) ==========
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    layout="wide",
    page_icon="📄",
    menu_items={
        'Get Help': 'https://github.com/PrinsSayja01/Advanced-Resume-Analyzer',
        'Report a bug': "https://github.com/PrinsSayja01/Advanced-Resume-Analyzer/issues",
        'About': "# Advanced Resume Analyzer\n\nContinuous learning system for resume optimization"
    }
)

# ========== IMPORTS ==========
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
import numpy as np

# ========== NLP INITIALIZATION ==========
@st.cache_resource
def load_nlp_model():
    """Load spaCy model with automatic download if missing"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading language model (one-time setup)...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                          check=True, stderr=subprocess.PIPE)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Model download failed: {str(e)}")
            st.stop()

nlp = load_nlp_model()

# ========== DATA MANAGEMENT ==========
DATA_DIR = "resume_data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_history():
    """Safe history loading with error recovery"""
    try:
        history_file = os.path.join(DATA_DIR, "analysis_history.pkl")
        if os.path.exists(history_file):
            with open(history_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"History load error: {str(e)}")
    return {"resumes": [], "jobs": [], "matches": []}

def save_history(data):
    """Atomic history saving"""
    try:
        temp_file = os.path.join(DATA_DIR, "temp_history.pkl")
        history_file = os.path.join(DATA_DIR, "analysis_history.pkl")
        
        with open(temp_file, "wb") as f:
            pickle.dump(data, f)
        
        os.replace(temp_file, history_file)
    except Exception as e:
        st.warning(f"History save error: {str(e)}")

# ========== DOCUMENT PROCESSING ==========
def extract_text(file):
    """Robust text extraction from PDF/DOCX"""
    if not file:
        return ""
    
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                text = []
                for page in pdf.pages:
                    try:
                        # Primary extraction method
                        content = page.extract_text()
                        # Fallback to mediabox if needed
                        if not content:
                            content = page.crop(page.mediabox).extract_text()
                        text.append(content or "")
                    except Exception as e:
                        st.warning(f"Page processing warning: {str(e)}")
                        text.append("")
                return " ".join(text).strip()
                
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return " ".join(para.text for para in doc.paragraphs if para.text)
    except Exception as e:
        st.error(f"File processing failed: {str(e)}")
    return ""

# ========== SKILLS DATABASE ==========
SKILLS = {
    "technical": {
        "Programming": ["python", "java", "javascript", "sql", "c++"],
        "Cloud": ["aws", "azure", "docker", "kubernetes"],
        "Data": ["pandas", "numpy", "machine learning"]
    },
    "soft": {
        "Communication": ["presentation", "writing", "public speaking"],
        "Management": ["leadership", "teamwork", "project management"]
    }
}

# ========== CORE ANALYSIS ==========
def analyze_documents(resume_text, job_text):
    """Main analysis function with error handling"""
    try:
        # Skill extraction
        resume_skills = defaultdict(set)
        job_skills = defaultdict(set)
        
        text_pairs = [
            (resume_text.lower(), resume_skills),
            (job_text.lower(), job_skills)
        ]
        
        for text, skills_dict in text_pairs:
            for category in SKILLS:
                for subcategory, skill_list in SKILLS[category].items():
                    for skill in skill_list:
                        if skill in text:
                            skills_dict[category].add(skill)
        
        # Calculate matches
        results = {}
        for category in SKILLS:
            job_set = job_skills[category]
            resume_set = resume_skills[category]
            
            match_percent = len(job_set & resume_set) / max(1, len(job_set)) * 100
            results[f"{category}_match"] = round(match_percent, 1)
            results[f"missing_{category}"] = sorted(job_set - resume_set)
        
        # Update history
        history = get_history()
        history["resumes"].append(resume_text)
        history["jobs"].append(job_text)
        history["matches"].append(results["technical_match"] / 100)
        save_history(history)
        
        return results
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return {
            "technical_match": 0,
            "soft_match": 0,
            "missing_technical": [],
            "missing_soft": []
        }

# ========== STREAMLIT UI ==========
def main():
    st.title("📄 Professional Resume Analyzer")
    st.markdown("""
    **AI-powered resume optimization**  
    Continuous learning system that improves with each analysis
    """)
    
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Paste the job description
        2. Upload your resume (PDF or Word)
        3. Click Analyze
        4. View your match score and improvement suggestions
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        job_desc = st.text_area(
            "Job Description:", 
            height=300,
            placeholder="Paste the complete job description here..."
        )
    with col2:
        resume_file = st.file_uploader(
            "Upload Resume:", 
            type=["pdf", "docx"],
            help="PDF or Word documents only"
        )
    
    if st.button("Analyze Resume", type="primary"):
        if not job_desc or not resume_file:
            st.warning("Please provide both job description and resume")
        else:
            with st.spinner("Analyzing documents..."):
                resume_text = extract_text(resume_file)
                if not resume_text:
                    st.error("Failed to extract text from resume file")
                else:
                    analysis = analyze_documents(resume_text, job_desc)
                    
                    # Display results
                    st.success("## 📊 Analysis Results")
                    
                    # Match metrics
                    cols = st.columns(2)
                    cols[0].metric(
                        "Technical Skills Match", 
                        f"{analysis['technical_match']}%",
                        help="Percentage of required technical skills matched"
                    )
                    cols[1].metric(
                        "Soft Skills Match", 
                        f"{analysis['soft_match']}%",
                        help="Percentage of required soft skills matched"
                    )
                    
                    # Improvement suggestions
                    st.subheader("🔍 Improvement Suggestions")
                    for category in ["technical", "soft"]:
                        if analysis[f"missing_{category}"]:
                            with st.expander(f"Missing {category.capitalize()} Skills"):
                                st.write(", ".join(analysis[f"missing_{category}"]))
                    
                    # System status
                    history = get_history()
                    st.caption(f"💡 System has processed {len(history['resumes'])} resumes")

# ========== EXECUTION GUARD ==========
if __name__ == "__main__":
    if not st.runtime.exists():
        # Running in bare mode (testing/debugging)
        print("Running in test mode - some Streamlit features may be limited")
    main()
