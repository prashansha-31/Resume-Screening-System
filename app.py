from flask import Flask, render_template, request
from resume_model import match_resume
from pdfminer.high_level import extract_text
import os

app = Flask(__name__)

# Folder to store uploaded resumes
UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    resume = request.files["resume"]
    job_desc = request.form["job_desc"]

    resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    resume.save(resume_path)

    # Extract text from PDF resume
    resume_text = extract_text(resume_path)

    # Call ML model
    score = match_resume(resume_text, job_desc)

    # Decision logic
    status = "Shortlisted" if score >= 50 else "Not Shortlisted"

    return render_template(
        "result.html",
        score=score,
        status=status
    )

if __name__ == "__main__":
    app.run(debug=True)
