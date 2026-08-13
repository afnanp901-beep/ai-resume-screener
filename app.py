import os
import re
import numpy as np
from flask import Flask, render_template, request
from pypdf import PdfReader
from langsmith import traceable

try:
    from sentence_transformers import SentenceTransformer, util
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False

app = Flask(__name__)

if ML_LIBRARIES_AVAILABLE:
    print("Loading SBERT Semantic Transformer model...")
    # Using the lightweight contextual model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
else:
    sbert_model = None
    print("Warning: ML libraries are not fully installed. Running in mock/compatibility mode.")

def extract_text_from_pdf(file_stream):
    """Helper function to extract clean text streams from an uploaded PDF file stream."""
    try:
        reader = PdfReader(file_stream)
        text_content = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        
        raw_text = " ".join(text_content)
        clean_text = re.sub(r'\s+', ' ', raw_text).strip()
        return clean_text
    except Exception as e:
        print(f"Error reading PDF text: {e}")
        return ""

@traceable(name="doc2vec_scoring")
def calculate_doc2vec_similarities(jd_text, resume_texts):
    """
    Trains an optimized Doc2Vec model in PV-DBOW mode for small corpora,
    inferring distinct directional vectors to prevent structural similarity collapse.
    """
    if not ML_LIBRARIES_AVAILABLE:
        # Fallback values if libraries are missing
        return [50.0] * len(resume_texts)

    def tokenize(text):
        # Strip punctuation and tokenize to clean lowercase words
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    # Pre-process job description and resumes
    documents = [jd_text] + resume_texts
    tagged_data = [TaggedDocument(words=tokenize(doc), tags=[str(i)]) for i, doc in enumerate(documents)]

    # Use Distributed Bag of Words (dm=0) and lower vector dimensions to prevent overfitting on tiny document sets
    model = Doc2Vec(vector_size=24, min_count=1, epochs=150, seed=42, dm=0)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Use high-epoch vector inference to get distinct spatial points for each resume
    jd_vector = model.infer_vector(tokenize(jd_text), epochs=100)
    doc2vec_scores = []

    for resume_text in resume_texts:
        resume_vector = model.infer_vector(tokenize(resume_text), epochs=100)
        
        # Calculate Cosine Similarity
        dot_product = np.dot(jd_vector, resume_vector)
        norm_jd = np.linalg.norm(jd_vector)
        norm_resume = np.linalg.norm(resume_vector)
        
        if norm_jd > 0 and norm_resume > 0:
            similarity = dot_product / (norm_jd * norm_resume)
            # Map similarity [-1, 1] to human-friendly percentage bounds
            percentage = float((similarity + 1) / 2 * 100)
            doc2vec_scores.append(round(percentage, 1))
        else:
            doc2vec_scores.append(0.0)

    return doc2vec_scores

def analyze_keyword_overlap(jd_text, resume_text):
    """
    Scans and isolates important nouns/tech terms between the JD and resumes,
    returning matched and missing technical concepts for customized feedback.
    """
    def clean_and_tokenize(text):
        return set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))

    jd_words = clean_and_tokenize(jd_text)
    resume_words = clean_and_tokenize(resume_text)

    # Filter out standard non-technical filler stop words
    common_stops = {
        'the', 'and', 'for', 'you', 'with', 'this', 'that', 'from', 'are', 'your', 'will', 'have', 'not', 
        'but', 'our', 'all', 'can', 'has', 'out', 'one', 'use', 'web', 'app', 'job', 'role', 'work', 
        'team', 'skills', 'experience', 'description', 'requirements', 'responsibilities', 'development', 
        'strong', 'candidate', 'required', 'knowledge', 'years', 'using', 'files', 'methods', 'given', 'about'
    }

    important_jd_words = {word for word in jd_words if word not in common_stops}
    
    # Identify matches and gaps
    matched = [word.capitalize() for word in important_jd_words if word in resume_words]
    missing = [word.capitalize() for word in important_jd_words if word not in resume_words]

    return sorted(matched)[:4], sorted(missing)[:4]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@traceable(name="resume_screening_request")
def predict():
    jd_text = request.form.get('job_description', '').strip()
    uploaded_files = request.files.getlist('resumes')

    if not jd_text:
        return "Please provide a valid Job Description.", 400
    if not uploaded_files or uploaded_files[0].filename == '':
        return "Please upload at least one PDF resume.", 400

    raw_resumes = []
    valid_filenames = []

    # 1. Parse and extract text from all incoming PDFs
    for file in uploaded_files:
        if file and file.filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file)
            if extracted_text:
                raw_resumes.append(extracted_text)
                valid_filenames.append(file.filename)

    if not raw_resumes:
        return "Could not extract text from any of the uploaded files. Please check your PDF formats.", 400

    sbert_scores = []
    doc2vec_scores = []

    # 2. Compute Dual Model Similarity Arrays
    if ML_LIBRARIES_AVAILABLE and sbert_model is not None:
        # SBERT Contextual Semantic Match
        jd_embedding = sbert_model.encode(jd_text, convert_to_tensor=True)
        resume_embeddings = sbert_model.encode(raw_resumes, convert_to_tensor=True)
        
        cosine_results = util.cos_sim(jd_embedding, resume_embeddings)[0]
        for score in cosine_results:
            percentage = float(score.item() * 100)
            percentage = max(0.0, min(100.0, percentage))
            sbert_scores.append(round(percentage, 1))

        # Optimized Doc2Vec Paragraph Match
        doc2vec_scores = calculate_doc2vec_similarities(jd_text, raw_resumes)
    else:
        # Mock values for offline fallbacks
        for i in range(len(valid_filenames)):
            sbert_scores.append(round(45.0 + (i * 4.3) % 15.0, 1))
            doc2vec_scores.append(round(52.0 + (i * 6.7) % 25.0, 1))

    compiled_results = []
    for i in range(len(valid_filenames)):
        avg_score = round((sbert_scores[i] + doc2vec_scores[i]) / 2, 1)
        
        # Pull candidate-specific matches and missing keywords
        matched, missing = analyze_keyword_overlap(jd_text, raw_resumes[i])
        
        # Categorize base fit level
        if avg_score >= 65:
            status = "Strong Match"
        elif avg_score >= 40:
            status = "Moderate Match"
        else:
            status = "Low Match"

        # Dynamically build highly individualized feedback
        feedback_parts = [f"[{status}]"]
        if matched:
            feedback_parts.append(f"Identified alignment on: {', '.join(matched)}.")
        if missing:
            feedback_parts.append(f"Gaps found in: {', '.join(missing)}.")
        else:
            feedback_parts.append("All primary skills are present.")

        summary = " ".join(feedback_parts)

        compiled_results.append({
            'filename': valid_filenames[i],
            'sbert_score': sbert_scores[i],
            'doc2vec_score': doc2vec_scores[i],
            'avg_score': avg_score,
            'summary': summary
        })

    # Sort results list based on the combined average rating (Highest Rank first)
    ranked_results = sorted(compiled_results, key=lambda x: x['avg_score'], reverse=True)

    return render_template('index.html', results=ranked_results, jd_text=jd_text)

if __name__ == '__main__':
    app.run(debug=True)