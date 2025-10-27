# src/scorer.py
import os
import argparse
import glob
import csv
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# When running as `python src/scorer.py`, the current working dir is repo root,
# and Python sets sys.path[0] to 'src', so we can import local modules as plain names.
from parser import extract_text_from_pdf_path
from feature_extractor import normalize_text, extract_skills, extract_years_of_experience, extract_name

def load_skill_list(path):
    with open(path, "r", encoding="utf-8") as f:
        skills = [line.strip() for line in f if line.strip()]
    return skills

def score_batch(resume_folder, jd_path_or_text, skills_path, output_csv="ranked_candidates.csv"):
    # load resume files (pdf or txt)
    resume_paths = sorted(glob.glob(os.path.join(resume_folder, "*")))
    resumes = []
    filenames = []
    for p in resume_paths:
        text = extract_text_from_pdf_path(p)
        resumes.append(normalize_text(text))
        filenames.append(os.path.basename(p))

    # job description (either a path or raw text)
    if os.path.exists(jd_path_or_text):
        with open(jd_path_or_text, "r", encoding="utf-8") as f:
            jd_text = normalize_text(f.read())
    else:
        jd_text = normalize_text(jd_path_or_text)

    if not resumes:
        print("No resumes found in folder:", resume_folder)
        return []

    # TF-IDF vectorization (resume corpus + JD)
    corpus = resumes + [jd_text]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus)
    jd_vec = X[-1]
    resume_vecs = X[:-1]
    sims = cosine_similarity(resume_vecs, jd_vec).flatten()

    skill_list = load_skill_list(skills_path)

    records = []
    for name, text, score in zip(filenames, resumes, sims):
        matched_skills = extract_skills(text, skill_list)
        years = extract_years_of_experience(text)
        candidate_name = extract_name(text) or name
        records.append({
            "filename": name,
            "candidate_name": candidate_name,
            "match_score": float(score),
            "matched_skills": ";".join(matched_skills),
            "years_experience": years if years is not None else "",
        })

    # sort by score descending
    records = sorted(records, key=lambda r: r["match_score"], reverse=True)

    # write CSV
    keys = ["filename", "candidate_name", "match_score", "matched_skills", "years_experience"]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # print summary
    print("Saved rankings to:", output_csv)
    for r in records:
        print(f"{r['candidate_name']} ({r['filename']}): {r['match_score']*100:.2f}% | skills: {r['matched_skills']} | years: {r['years_experience']}")
    return records

def main():
    parser = argparse.ArgumentParser(description="Score resumes against a job description.")
    parser.add_argument("--resumes", required=True, help="Folder with resume files")
    parser.add_argument("--jd", required=True, help="Job description file path or raw JD text")
    parser.add_argument("--skills", required=True, help="Path to skills.txt")
    parser.add_argument("--out", default="ranked_candidates.csv", help="Output CSV filename")
    args = parser.parse_args()

    score_batch(args.resumes, args.jd, args.skills, args.out)

if __name__ == "__main__":
    main()
