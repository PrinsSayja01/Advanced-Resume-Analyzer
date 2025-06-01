# Professional Resume Analyzer 🤖📄

An AI-powered tool that helps job seekers optimise their resumes by comparing them against job descriptions. The system provides detailed skill matching and continuously improves through machine learning.

## Features ✨

- **Smart Skill Matching**: Identifies both technical and soft skills from resumes and job descriptions
- **Gap Analysis**: Highlights missing skills that need improvement
- **Learning System**: Gets smarter with each analysis (saves historical data)
- **Multi-format Support**: Works with PDF and Word documents
- **Professional Metrics**: Provides clear percentage match scores

## How It Works 🛠️

1. Paste the job description in the left panel
2. Upload your resume (PDF or Word format)
3. Click "Analyse Resume" (note British spelling)
4. View your:
   - Technical skills match percentage
   - Soft skills match percentage
   - Specific missing skills to address

## Technical Details ⚙️

- Built with Python using:
  - Streamlit for the web interface
  - spaCy for natural language processing
  - pdfplumber and python-docx for document parsing
- Uses TF-IDF vectorisation for text analysis
- Implements atomic file operations for data safety

## Installation Guide 📥

1. Clone the repository:
   ```bash
   git clone https://github.com/PrinsSayja01/Advanced-Resume-Analyzer.git
