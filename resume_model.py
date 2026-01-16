from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

SKILLS = [
    "java", "python", "c++", "c", "javascript",
    "html", "css",
    "spring", "spring boot", "springboot",
    "flask", "django",
    "mysql", "sql", "mongodb",
    "data structures", "algorithms",
    "machine learning",
    "oop", "object oriented programming",
    "rest api", "hibernate"
]

def clean_text(text):
    # remove special characters and extra spaces
    text = text.lower()
    text = re.sub(r"[^a-z0-9+ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def extract_skills(text):
    text = clean_text(text)
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return " ".join(set(found))  # remove duplicates

def match_resume(resume_text, job_desc):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_desc)

    # 🔍 DEBUG SAFETY CHECK
    if len(resume_skills) < 3 or len(jd_skills) < 3:
        return 0.0

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_skills, jd_skills])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return round(similarity * 100, 2)
