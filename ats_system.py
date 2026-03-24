"""
=============================================================
  Mini ATS - Resume Screening System using NLP
  Subject: Text Processing (NLP)
  
  Libraries used: Scikit-learn, Pandas, re (regex)
  Concepts: TF-IDF, Cosine Similarity, Tokenization,
            Stopword Removal, Stemming, Keyword Matching
=============================================================
"""

# ─────────────────────────────────────────────
# STEP 0: Import required libraries
# ─────────────────────────────────────────────
import os
import re
import math
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Try importing PyPDF2 for PDF support (optional)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ─────────────────────────────────────────────
# STEP 1: Read text from file (TXT or PDF)
# ─────────────────────────────────────────────
def read_file(filepath):
    """
    Reads a .txt or .pdf file and returns its text content.
    This is the input stage — we load resume data from files.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".pdf"):
        if not PDF_SUPPORT:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Only .txt and .pdf files are supported.")


# ─────────────────────────────────────────────
# STEP 2: Text Preprocessing using NLP
# ─────────────────────────────────────────────

# Define a simple stemmer (Porter-style suffix stripping)
SUFFIXES = ["ing", "tion", "ations", "izes", "ized", "ment", "ments",
            "ness", "ous", "ive", "ful", "less", "er", "est", "ly",
            "ed", "s", "es", "al", "ic", "ical"]

def simple_stem(word):
    """
    A lightweight stemmer that strips common English suffixes.
    Example: 'running' → 'runn', 'classification' → 'classif'
    """
    word = word.lower()
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess_text(text, use_stemming=True):
    """
    Full NLP preprocessing pipeline:
      1. Lowercase        → makes text uniform ("Python" = "python")
      2. Tokenization     → split into individual words
      3. Remove noise     → strip numbers, special characters
      4. Stopword removal → remove common words like "the", "is", "and"
      5. Stemming         → reduce words to their root form

    Returns: cleaned string of processed tokens
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Tokenize (split on non-alphabetic characters)
    tokens = re.findall(r"[a-z]+", text)

    # 3. & 4. Remove stopwords (using sklearn's built-in list)
    stop_words = set(ENGLISH_STOP_WORDS)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 5. Stemming (optional)
    if use_stemming:
        tokens = [simple_stem(t) for t in tokens]

    return " ".join(tokens)


# ─────────────────────────────────────────────
# STEP 3: Extract Key Skills via Keyword Matching
# ─────────────────────────────────────────────

# A curated list of tech skills to match against resumes
SKILL_KEYWORDS = [
    # Programming Languages
    "python", "java", "sql", "javascript", "php", "html", "css", "r", "scala",
    # ML / AI
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    # Libraries & Frameworks
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "nltk",
    "spacy", "hugging face", "transformers", "bert", "word2vec",
    "numpy", "pandas", "matplotlib", "seaborn", "flask", "fastapi",
    "react", "node", "express", "bootstrap",
    # NLP Techniques
    "tfidf", "tf-idf", "tokenization", "stemming", "lemmatization",
    "cosine similarity", "word embedding", "ner", "named entity",
    # Tools & Platforms
    "git", "docker", "aws", "gcp", "azure", "mlflow", "spark",
    "jupyter", "vs code", "postman", "figma",
    # Databases
    "mysql", "mongodb", "firebase", "postgresql",
    # Concepts
    "data analysis", "data preprocessing", "feature engineering",
    "model deployment", "api", "rest api", "version control",
]


def extract_skills(raw_text):
    """
    Scans the raw (un-preprocessed) resume text for known skill keywords.
    Returns a sorted list of matched skills.
    
    We use the original text here (not stemmed) for accurate multi-word matching.
    """
    text_lower = raw_text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        # Use word-boundary matching to avoid partial matches
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(found)


# ─────────────────────────────────────────────
# STEP 4: TF-IDF Vectorization + Cosine Similarity
# ─────────────────────────────────────────────

def compute_similarity(preprocessed_job_desc, preprocessed_resumes):
    """
    Converts text into TF-IDF vectors and computes cosine similarity.
    
    TF-IDF: Measures how important a word is to a document relative to
            all documents in the collection.
    
    Cosine Similarity: Measures the angle between two vectors.
                       Score of 1 = identical, 0 = completely different.
    
    Returns: list of similarity scores (one per resume)
    """
    # Combine all documents: job description first, then resumes
    all_docs = [preprocessed_job_desc] + preprocessed_resumes

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Job description is index 0, resumes are index 1 onwards
    job_vec = tfidf_matrix[0]
    resume_vecs = tfidf_matrix[1:]

    # Compute cosine similarity between job desc and each resume
    scores = cosine_similarity(job_vec, resume_vecs)[0]
    return scores


# ─────────────────────────────────────────────
# STEP 5: Highlight Matched Keywords
# ─────────────────────────────────────────────

def get_matched_keywords(job_raw, resume_raw, top_n=10):
    """
    Finds important keywords from the job description that appear in the resume.
    Uses TF-IDF weights to pick the most significant job keywords.
    """
    job_lower = job_raw.lower()
    resume_lower = resume_raw.lower()

    # Tokenize job description into unique words
    job_words = re.findall(r"[a-z]{3,}", job_lower)
    job_words = [w for w in job_words if w not in ENGLISH_STOP_WORDS]

    # Count word frequency in job description
    word_freq = Counter(job_words)
    
    # Find which job keywords appear in the resume
    matched = []
    for word, freq in word_freq.most_common(50):
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, resume_lower):
            matched.append(word)
        if len(matched) >= top_n:
            break

    return matched


# ─────────────────────────────────────────────
# STEP 6: Main Screening Function
# ─────────────────────────────────────────────

def screen_resumes(resume_paths, job_desc_path, top_n=5):
    """
    Main function that ties everything together:
    1. Reads all files
    2. Preprocesses text
    3. Extracts skills
    4. Computes TF-IDF + cosine similarity
    5. Ranks resumes
    6. Returns a structured results DataFrame
    """
    print("\n" + "="*60)
    print("   MINI ATS — RESUME SCREENING SYSTEM")
    print("="*60)

    # ── Read Job Description ──
    print(f"\n📄 Loading Job Description: {os.path.basename(job_desc_path)}")
    job_raw = read_file(job_desc_path)
    job_preprocessed = preprocess_text(job_raw)
    job_skills = extract_skills(job_raw)
    print(f"   ✔ Job Skills Found: {len(job_skills)}")

    # ── Read & Process All Resumes ──
    print(f"\n📁 Loading {len(resume_paths)} Resume(s)...")
    resumes_raw = []
    resumes_preprocessed = []
    resume_names = []

    for path in resume_paths:
        name = os.path.basename(path)
        raw = read_file(path)
        processed = preprocess_text(raw)
        resumes_raw.append(raw)
        resumes_preprocessed.append(processed)
        resume_names.append(name)
        print(f"   ✔ Loaded: {name}")

    # ── Compute Similarity Scores ──
    print("\n⚙  Computing TF-IDF Vectors and Cosine Similarity...")
    scores = compute_similarity(job_preprocessed, resumes_preprocessed)

    # ── Build Results Table ──
    results = []
    for i, name in enumerate(resume_names):
        resume_skills = extract_skills(resumes_raw[i])
        matched_kw = get_matched_keywords(job_raw, resumes_raw[i], top_n=8)
        
        # Identify which job skills are in resume
        common_skills = [s for s in job_skills if s in resume_skills]

        results.append({
            "Rank": 0,                                          # filled after sorting
            "Resume": name,
            "Score (%)": round(scores[i] * 100, 2),
            "Extracted Skills": resume_skills,
            "Matched Keywords": matched_kw,
            "Common Skills": common_skills,
        })

    # ── Sort by Score ──
    results = sorted(results, key=lambda x: x["Score (%)"], reverse=True)
    for i, r in enumerate(results):
        r["Rank"] = i + 1

    # ── Display Results ──
    print("\n" + "="*60)
    print("   📊 SCREENING RESULTS (Ranked by Match Score)")
    print("="*60)

    for r in results[:top_n]:
        print(f"\n🏆 Rank #{r['Rank']} — {r['Resume']}")
        print(f"   Match Score : {r['Score (%)']}%")
        print(f"   Skills Found: {', '.join(r['Extracted Skills'][:10]) if r['Extracted Skills'] else 'None'}")
        print(f"   Matched Keywords: {', '.join(r['Matched Keywords']) if r['Matched Keywords'] else 'None'}")
        print(f"   Common Skills with JD: {', '.join(r['Common Skills'][:8]) if r['Common Skills'] else 'None'}")
        print(f"   {'─'*50}")

    # ── Show Top Candidates ──
    print(f"\n🎯 TOP {min(top_n, len(results))} CANDIDATES:")
    for r in results[:top_n]:
        bar_len = int(r["Score (%)"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"   {r['Rank']}. {r['Resume']:<30} [{bar}] {r['Score (%)']}%")

    # ── Return as DataFrame for further use ──
    df = pd.DataFrame([{
        "Rank": r["Rank"],
        "Resume": r["Resume"],
        "Score (%)": r["Score (%)"],
        "Skills Count": len(r["Extracted Skills"]),
        "Common Skills": len(r["Common Skills"]),
        "Top Skills": ", ".join(r["Extracted Skills"][:5]),
    } for r in results])

    print("\n\n📋 SUMMARY TABLE:")
    print(df.to_string(index=False))

    return results, df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Configure paths ──
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "sample_data")

    # List of resume files to screen
    resume_files = [
        os.path.join(DATA_DIR, "resume_alice.txt"),
        os.path.join(DATA_DIR, "resume_bob.txt"),
        os.path.join(DATA_DIR, "resume_priya.txt"),
    ]

    # Job description file
    job_file = os.path.join(DATA_DIR, "job_description.txt")

    # Run the screening
    results, summary_df = screen_resumes(
        resume_paths=resume_files,
        job_desc_path=job_file,
        top_n=5
    )

    print("\n✅ Screening complete!")
    print("   The top candidate is:", results[0]["Resume"])
