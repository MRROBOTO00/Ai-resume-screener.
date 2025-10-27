# src/app.py
import streamlit as st
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parser import extract_text_from_pdf_fileobj
from feature_extractor import normalize_text, extract_skills, extract_years_of_experience

st.set_page_config(page_title="AI Resume Screener", layout="centered")

st.title("🤖 AI Resume Screener")
st.markdown("Upload one or more resume PDFs and paste a Job Description (JD). The app returns a match score and basic extracted info.")

jd_text = st.text_area("Paste Job Description (JD) here", height=200)

uploaded_files = st.file_uploader("Upload resume PDF(s)", type=["pdf"], accept_multiple_files=True)

skill_file = st.file_uploader("Optional: upload `skills.txt` (one skill per line)", type=["txt"])

if uploaded_files and jd_text:
    # read resumes
    resumes_text = []
    filenames = []
    for up in uploaded_files:
        # Streamlit gives a BytesIO-like object
        try:
            text = extract_text_from_pdf_fileobj(up)
        except Exception:
            up.seek(0)
            raw = up.read()
            text = extract_text_from_pdf_fileobj(io.BytesIO(raw))
        resumes_text.append(normalize_text(text))
        filenames.append(up.name)

    # load skill list
    if skill_file:
        skill_list = [line.decode("utf-8").strip() for line in skill_file.getvalue().splitlines() if line.strip()]
    else:
        # fallback to a small built-in list
        skill_list = ["python", "java", "javascript", "react", "node", "django", "flask", "sql", "aws", "docker"]

    # compute TF-IDF similarity
    corpus = resumes_text + [normalize_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus)
    jd_vec = X[-1]
    resume_vecs = X[:-1]
    sims = cosine_similarity(resume_vecs, jd_vec).flatten()

    # prepare display
    results = []
    for fname, text, score in zip(filenames, resumes_text, sims):
        matched = extract_skills(text, skill_list)
        years = extract_years_of_experience(text)
        results.append({
            "filename": fname,
            "score": float(score),
            "matched_skills": matched,
            "years": years
        })

    # sort and show
    results = sorted(results, key=lambda r: r["score"], reverse=True)
    for r in results:
        st.subheader(f"{r['filename']} — {r['score']*100:.2f}% match")
        st.write("**Matched skills:**", ", ".join(r["matched_skills"]) if r["matched_skills"] else "None detected")
        st.write("**Years (heuristic):**", r["years"] if r["years"] else "N/A")
        # optional: show small extract
        with st.expander("View text snippet"):
            st.write((text[:1000] + "...") if len(text) > 1000 else text)

else:
    st.info("Paste a job description and upload at least one resume to start.")
