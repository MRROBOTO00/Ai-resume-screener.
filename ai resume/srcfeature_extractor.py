# src/feature_extractor.py
import re
import spacy

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    return _nlp

def normalize_text(text):
    if not text:
        return ""
    # normalize whitespace
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text, skill_list):
    """
    Simple skill detection: checks presence of skill tokens (case-insensitive).
    Returns a sorted list of matched skills (unique).
    """
    text_low = text.lower()
    found = set()
    for skill in skill_list:
        skill_low = skill.lower().strip()
        # match whole words/phrases
        pattern = r"\b" + re.escape(skill_low) + r"\b"
        if re.search(pattern, text_low):
            found.add(skill)
    return sorted(found)

def extract_years_of_experience(text):
    """
    Heuristic: find occurrences like 'X years', 'X+ years', 'X year'
    Returns maximum years found or None
    """
    matches = re.findall(r"(\d{1,2})(?:\+)?\s*(?:years|year)\b", text.lower())
    if not matches:
        return None
    nums = [int(m) for m in matches]
    return max(nums)

def extract_name(text):
    """
    Naive name extraction: use spaCy NER to find first PERSON entity.
    """
    nlp = _get_nlp()
    doc = nlp(text[:1000])  # only prefix for performance
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None
