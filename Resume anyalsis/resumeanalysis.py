# advanced_resume_analyzer.py
import spacy
import pdfplumber
import docx
import re
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Initialize NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ================== DATA STORAGE ==================
DATA_DIR = "resume_analyzer_data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_history():
    """Safe history loading with error handling"""
    try:
        history_path = os.path.join(DATA_DIR, "analysis_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Error loading history: {str(e)}")
    return {"resumes": [], "jobs": [], "matches": [], "timestamps": []}

def save_history(data):
    """Safe history saving"""
    try:
        history_path = os.path.join(DATA_DIR, "analysis_history.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Error saving history: {str(e)}")

# ================== ENHANCED PDF HANDLING ==================
def extract_text(file):
    """Robust text extraction from PDF/DOCX with error recovery"""
    if not file:
        return ""
    
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                text = []
                for page in pdf.pages:
                    try:
                        # Try normal extraction first
                        page_text = page.extract_text()
                        
                        # Fallback to mediabox if needed
                        if not page_text:
                            try:
                                page_text = page.crop(page.mediabox).extract_text()
                            except:
                                page_text = ""
                        
                        text.append(page_text or "")
                    except Exception as e:
                        st.warning(f"Page processing error: {str(e)}")
                        text.append("")
                return " ".join(text).strip()
                
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return " ".join(para.text for para in doc.paragraphs if para.text)
    except Exception as e:
        st.error(f"File reading error: {str(e)}")
    return ""

# ================== SKILLS DATABASE ==================
SKILLS_DATABASE = {
    "technical": {
        "programming": ["python", "java", "javascript", "sql", "c++"],
        "cloud": ["aws", "azure", "docker", "kubernetes"],
        "data": ["pandas", "numpy", "machine learning", "pytorch"]
    },
    "soft": {
        "communication": ["presentation", "writing", "public speaking"],
        "management": ["leadership", "teamwork", "project management"]
    },
    "languages": ["english", "spanish", "french", "german"]
}

# ================== MACHINE LEARNING MODEL ==================
class SkillPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.is_trained = False
    
    def train(self, X, y):
        """Train model with error handling"""
        try:
            X_vec = self.vectorizer.fit_transform(X)
            self.model.fit(X_vec, y)
            self.is_trained = True
            with open(os.path.join(DATA_DIR, "model.pkl"), "wb") as f:
                pickle.dump(self, f)
            return True
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            return False
    
    def predict(self, text):
        """Safe prediction with fallback"""
        if not self.is_trained:
            return 0.5
        try:
            return self.model.predict_proba(self.vectorizer.transform([text]))[0][1]
        except:
            return 0.5

# Initialize model
try:
    with open(os.path.join(DATA_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
except:
    model = SkillPredictor()

# ================== CORE ANALYSIS ==================
def extract_skills(text):
    """Robust skill extraction from text"""
    if not text:
        return defaultdict(set)
    
    try:
        doc = nlp(text.lower())
        found_skills = defaultdict(set)
        
        # Check multi-word phrases first
        for category in SKILLS_DATABASE:
            if isinstance(SKILLS_DATABASE[category], dict):  # Nested categories
                for subcategory, skills in SKILLS_DATABASE[category].items():
                    for skill in skills:
                        if skill in text.lower():
                            found_skills[category].add(skill)
            else:  # Flat lists (languages)
                for skill in SKILLS_DATABASE[category]:
                    if skill in text.lower():
                        found_skills[category].add(skill)
        
        return found_skills
    except Exception as e:
        st.warning(f"Skill extraction error: {str(e)}")
        return defaultdict(set)

def analyze(resume_text, job_text):
    """Complete analysis with learning"""
    try:
        # Basic skill matching
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        
        # Calculate matches
        results = {}
        for category in ["technical", "soft", "languages"]:
            job_set = job_skills.get(category, set())
            resume_set = resume_skills.get(category, set())
            
            match_pct = len(job_set & resume_set) / max(1, len(job_set)) * 100
            results[f"{category}_match"] = round(match_pct, 1)
            results[f"missing_{category}"] = sorted(job_set - resume_set)
        
        # Store for learning
        history = load_history()
        history["resumes"].append(resume_text)
        history["jobs"].append(job_text)
        history["matches"].append(results["technical_match"] / 100)
        history["timestamps"].append(datetime.now())
        save_history(history)
        
        # Add ML prediction if enough data
        if len(history["resumes"]) >= 5:
            results["ml_prediction"] = round(
                model.predict(resume_text + " " + job_text) * 100, 1
            )
            
            # Retrain periodically
            if len(history["resumes"]) % 5 == 0 and len(history["resumes"]) >= 10:
                model.train(history["resumes"] + history["jobs"], history["matches"])
        
        return results
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            "technical_match": 0,
            "soft_match": 0,
            "languages_match": 0,
            "missing_technical": [],
            "missing_soft": [],
            "missing_languages": []
        }

# ================== STREAMLIT UI ==================
def main():
    st.set_page_config(
        page_title="AI Resume Analyzer Pro",
        layout="wide",
        page_icon="📄"
    )
    
    st.title("📄 Advanced Resume Analyzer")
    st.markdown("""
    **Features:**
    - 🛠️ Technical skill matching
    - 🧠 Soft skill evaluation
    - 🌐 Language proficiency
    - 🤖 Machine learning improvements
    - 💾 Persistent learning across sessions
    """)
    
    with st.expander("How to use"):
        st.write("""
        1. Paste job description
        2. Upload your resume (PDF/DOCX)
        3. Click Analyze
        4. View matches and suggestions
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        job_desc = st.text_area("Job Description:", height=300)
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
                    
                    # Display results
                    st.success("## 📊 Analysis Results")
                    
                    # Match scores
                    cols = st.columns(4)
                    cols[0].metric("Technical", f"{analysis['technical_match']}%")
                    cols[1].metric("Soft Skills", f"{analysis['soft_match']}%")
                    cols[2].metric("Languages", f"{analysis['languages_match']}%")
                    if "ml_prediction" in analysis:
                        cols[3].metric("AI Prediction", f"{analysis['ml_prediction']}%")
                    
                    # Missing skills
                    st.subheader("🔍 Improvement Suggestions")
                    for category in ["technical", "soft", "languages"]:
                        if analysis[f"missing_{category}"]:
                            st.warning(f"Missing {category} skills:")
                            st.write(", ".join(analysis[f"missing_{category}"]))
                    
                    # Learning status
                    history = load_history()
                    st.caption(f"System has learned from {len(history['resumes'])} analyses")

if __name__ == "__main__":
    main()