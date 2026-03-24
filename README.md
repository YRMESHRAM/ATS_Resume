# 🧑‍💼 Mini ATS — Resume Screening System (NLP Project)

**Subject:** Text Processing / Natural Language Processing  
**Python Libraries:** `scikit-learn`, `pandas`, `re`, `PyPDF2` (optional), `streamlit` (optional)

---

## 📁 Project Structure

```
ats_project/
│
├── ats_system.py          ← Main Python script (command-line version)
├── streamlit_app.py       ← Optional: Web UI using Streamlit
│
└── sample_data/
    ├── resume_alice.txt   ← Sample Resume 1 (Python/ML Developer)
    ├── resume_bob.txt     ← Sample Resume 2 (Web Developer)
    ├── resume_priya.txt   ← Sample Resume 3 (ML Engineer)
    └── job_description.txt← Job Description (ML Engineer role)
```

---

## 🚀 How to Run

### Option 1: Command Line
```bash
pip install scikit-learn pandas PyPDF2
python ats_system.py
```

### Option 2: Streamlit Web App
```bash
pip install scikit-learn pandas PyPDF2 streamlit
streamlit run streamlit_app.py
```

---

## 🧠 Step-by-Step Explanation

### STEP 1 — Reading Input Files
- Resume files (.txt or .pdf) are read using Python's built-in `open()` or PyPDF2
- Each resume's text is extracted as a plain string
- The job description is similarly loaded

### STEP 2 — NLP Preprocessing

| Technique | What it does | Example |
|-----------|-------------|---------|
| **Lowercasing** | Uniform text | "Python" → "python" |
| **Tokenization** | Split into words | "I love Python" → ["I", "love", "Python"] |
| **Stopword Removal** | Remove common words | "is", "the", "and", "of" removed |
| **Stemming** | Reduce to root | "running" → "runn", "classification" → "classif" |

```python
# Example
text = "He is developing Machine Learning models"
# After preprocessing:
# → "develop machin learn model"
```

### STEP 3 — Feature Extraction

#### a) TF-IDF Vectorization
- **TF (Term Frequency):** How often a word appears in a document
- **IDF (Inverse Document Frequency):** How rare the word is across all documents
- Words that are frequent in one doc but rare overall get higher scores
- Each document becomes a **numerical vector** (array of TF-IDF scores)

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

#### b) Keyword/Skill Extraction
- A predefined list of ~60 technical skills is used
- Each resume is scanned using regex (`re.search`) for these keywords
- Multi-word skills like "machine learning" or "natural language processing" are also matched

### STEP 4 — Cosine Similarity
- The job description vector and each resume vector are compared
- Cosine similarity measures the **angle** between two vectors (0 = different, 1 = identical)
- Formula:

```
similarity = (A · B) / (||A|| × ||B||)
```

### STEP 5 — Ranking
- All resumes are sorted by their cosine similarity score (highest first)
- Scores are multiplied by 100 to display as a percentage

### STEP 6 — Output Display
- Ranked list with score, extracted skills, matched keywords, common skills
- Visual progress bar for quick comparison
- Summary table using Pandas DataFrame

---

## 📊 Sample Output

```
============================================================
   MINI ATS — RESUME SCREENING SYSTEM
============================================================

📄 Loading Job Description: job_description.txt
   ✔ Job Skills Found: 25

📁 Loading 3 Resume(s)...
   ✔ Loaded: resume_alice.txt
   ✔ Loaded: resume_bob.txt
   ✔ Loaded: resume_priya.txt

⚙  Computing TF-IDF Vectors and Cosine Similarity...

============================================================
   📊 SCREENING RESULTS (Ranked by Match Score)
============================================================

🏆 Rank #1 — resume_priya.txt
   Match Score : 52.03%
   Skills Found: aws, bert, cosine similarity, deep learning, docker, ...
   Common Skills with JD: aws, bert, cosine similarity, docker, keras, ...

🏆 Rank #2 — resume_alice.txt
   Match Score : 47.95%
   Skills Found: cosine similarity, data analysis, git, keras, ...
   Common Skills with JD: cosine similarity, docker, git, keras, ...

🏆 Rank #3 — resume_bob.txt
   Match Score : 12.63%
   Skills Found: api, aws, bootstrap, css, git, html, javascript, ...
   Common Skills with JD: aws, git, python

🎯 TOP 3 CANDIDATES:
   1. resume_priya.txt  [██████████░░░░░░░░░░] 52.03%
   2. resume_alice.txt  [█████████░░░░░░░░░░░] 47.95%
   3. resume_bob.txt    [██░░░░░░░░░░░░░░░░░░] 12.63%
```

---

## 🔑 Key Concepts Summary (For Viva)

| Concept | Definition |
|---------|-----------|
| **Tokenization** | Splitting text into individual words/tokens |
| **Stopwords** | Common words (is, the, and) that carry no meaning |
| **Stemming** | Reducing words to their root (running → run) |
| **TF-IDF** | Numerical measure of word importance in a document |
| **Cosine Similarity** | Mathematical measure of angle between two vectors |
| **NER** | Named Entity Recognition — extracting entities like names, skills |
| **ATS** | Applicant Tracking System — software to filter resumes |

---

## 📝 Requirements

```txt
scikit-learn>=1.0
pandas>=1.3
PyPDF2>=3.0   (optional, for PDF support)
streamlit>=1.20 (optional, for web interface)
```

---

*Submitted as part of Text Processing (NLP) Subject — College Practical*
