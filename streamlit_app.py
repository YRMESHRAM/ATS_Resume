"""
=============================================================
  Mini ATS — Streamlit Web Interface
  Run with: streamlit run streamlit_app.py
=============================================================
"""

import os
import re
import tempfile
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    exit()

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ─── Reuse core functions from ats_system.py ───

SUFFIXES = ["ing", "tion", "ations", "izes", "ized", "ment", "ments",
            "ness", "ous", "ive", "ful", "less", "er", "est", "ly",
            "ed", "s", "es", "al", "ic", "ical"]

SKILL_KEYWORDS = [
    "python", "java", "sql", "javascript", "php", "html", "css", "r", "scala",
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision",
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "nltk",
    "spacy", "hugging face", "transformers", "bert", "word2vec",
    "numpy", "pandas", "matplotlib", "seaborn", "flask", "fastapi",
    "react", "node", "express", "bootstrap",
    "tfidf", "tf-idf", "tokenization", "stemming", "lemmatization",
    "cosine similarity", "word embedding", "ner", "named entity",
    "git", "docker", "aws", "gcp", "azure", "mlflow", "spark",
    "jupyter", "vs code", "postman", "figma",
    "mysql", "mongodb", "firebase", "postgresql",
    "data analysis", "data preprocessing", "feature engineering",
    "model deployment", "api", "rest api", "version control",
]


def simple_stem(word):
    word = word.lower()
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    stop_words = set(ENGLISH_STOP_WORDS)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)


def extract_skills(raw_text):
    text_lower = raw_text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(found)


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        if not PDF_SUPPORT:
            st.error("PyPDF2 not installed. Run: pip install PyPDF2")
            return ""
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def get_matched_keywords(job_raw, resume_raw, top_n=10):
    job_lower = job_raw.lower()
    resume_lower = resume_raw.lower()
    job_words = re.findall(r"[a-z]{3,}", job_lower)
    job_words = [w for w in job_words if w not in ENGLISH_STOP_WORDS]
    word_freq = Counter(job_words)
    matched = []
    for word, _ in word_freq.most_common(50):
        if re.search(r"\b" + re.escape(word) + r"\b", resume_lower):
            matched.append(word)
        if len(matched) >= top_n:
            break
    return matched


# ─── Streamlit UI ───

st.set_page_config(page_title="Mini ATS Resume Screener", page_icon="🧑‍💼", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 2rem; }
    h1 { color: #1e293b; }
    .stProgress > div > div > div > div { background-color: #6366f1; }
    </style>
""", unsafe_allow_html=True)

st.title("🧑‍💼 Mini ATS — Resume Screening System")
st.markdown("*Upload resumes and a job description to automatically rank candidates by match score.*")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Resumes")
    resume_files = st.file_uploader(
        "Choose resume files (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key="resumes"
    )

with col2:
    st.subheader("📄 Upload Job Description")
    job_file = st.file_uploader(
        "Choose job description file (.txt or .pdf)",
        type=["txt", "pdf"],
        key="jd"
    )
    top_n = st.slider("Show Top N Candidates", min_value=1, max_value=10, value=5)

st.markdown("---")

if st.button("🚀 Screen Resumes", use_container_width=True, type="primary"):
    if not resume_files:
        st.warning("Please upload at least one resume.")
    elif not job_file:
        st.warning("Please upload a job description.")
    else:
        with st.spinner("Processing resumes..."):
            job_raw = read_uploaded_file(job_file)
            job_preprocessed = preprocess_text(job_raw)
            job_skills = extract_skills(job_raw)

            resumes_raw, resumes_preprocessed, names = [], [], []
            for rf in resume_files:
                raw = read_uploaded_file(rf)
                resumes_raw.append(raw)
                resumes_preprocessed.append(preprocess_text(raw))
                names.append(rf.name)

            all_docs = [job_preprocessed] + resumes_preprocessed
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

            results = []
            for i, name in enumerate(names):
                r_skills = extract_skills(resumes_raw[i])
                common = [s for s in job_skills if s in r_skills]
                matched_kw = get_matched_keywords(job_raw, resumes_raw[i], top_n=10)
                results.append({
                    "Resume": name,
                    "Score (%)": round(scores[i] * 100, 2),
                    "Skills": r_skills,
                    "Common Skills": common,
                    "Matched Keywords": matched_kw,
                })

            results = sorted(results, key=lambda x: x["Score (%)"], reverse=True)

        st.success("✅ Screening complete!")
        st.markdown("### 📊 Ranked Results")

        for i, r in enumerate(results[:top_n]):
            with st.expander(f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '🏅'} Rank #{i+1} — {r['Resume']}  |  Match Score: {r['Score (%)']}%", expanded=(i==0)):
                st.progress(r["Score (%)"] / 100)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🔧 Extracted Skills:**")
                    if r["Skills"]:
                        st.write(", ".join(r["Skills"]))
                    else:
                        st.write("None found")
                    st.markdown("**✅ Common Skills with JD:**")
                    if r["Common Skills"]:
                        for s in r["Common Skills"]:
                            st.markdown(f"- `{s}`")
                    else:
                        st.write("None")
                with c2:
                    st.markdown("**🔑 Matched Keywords:**")
                    if r["Matched Keywords"]:
                        cols = st.columns(3)
                        for j, kw in enumerate(r["Matched Keywords"]):
                            cols[j % 3].markdown(f"`{kw}`")

        st.markdown("---")
        st.markdown("### 📋 Summary Table")
        df = pd.DataFrame([{
            "Rank": i+1,
            "Resume": r["Resume"],
            "Score (%)": r["Score (%)"],
            "Skills Found": len(r["Skills"]),
            "Common w/ JD": len(r["Common Skills"]),
        } for i, r in enumerate(results[:top_n])])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📌 Job Description Skills Detected")
        st.write(", ".join(job_skills) if job_skills else "None found")

st.markdown("---")
st.caption("Mini ATS | Built with Python, NLTK concepts, Scikit-learn, and Streamlit | NLP Subject Project")
