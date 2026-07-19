Intelligent Resume Screener



An enterprise-ready, automated HR candidate ranking and matching dashboard built with Flask and Python. This application allows recruiters to upload multiple PDF resumes, input a job description (JD), and evaluate candidates from two distinct analytical angles using state-of-the-art NLP models.



🚀 Key Features



Dual-Model Similarity Engine: Matches candidates using semantic context (SBERT) and stylistic/thematic structure (Doc2Vec) simultaneously.



Multi-PDF Text Extraction: Parses text streams dynamically from multiple uploaded files in parallel using pypdf.



Dynamic Skill Gap \& Match Analysis: Performs localized keyword extraction to highlight precisely where a candidate aligns with the job description and identifies potential technical skill gaps.



Modern Enterprise Dashboard: An interactive, responsive, and highly polished user interface styled with Tailwind CSS and FontAwesome icons.



Production-Ready Core: Migrated from a prototyping layout (Streamlit) to a scalable, production-grade structure (Flask + Gunicorn).



🧠 System Architecture \& Algorithms



To rank and filter candidates objectively, the system evaluates documents from two different mathematical perspectives:



\[Uploaded Resumes] ──> \[pypdf Parsing] ──> \[Text Normalization] 

&#x20;                                                   │

&#x20;                  ┌────────────────────────────────┴────────────────────────────────┐

&#x20;                  ▼                                                                 ▼

&#x20;        \[SBERT (Transformer)]                                               \[Doc2Vec Model]

&#x20; Contextual \& Semantic Embeddings                                    Paragraph-level Co-occurrences

&#x20;                  │                                                                 │

&#x20;                  └────────────────────────────────┬────────────────────────────────┘

&#x20;                                                   ▼

&#x20;                                         \[Cosine Similarity]

&#x20;                                                   │

&#x20;                                                   ▼

&#x20;                                  \[Combined Match \& Rank Output]





1\. Semantic Match (SBERT)



Using the lightweight, transformer-based all-MiniLM-L6-v2 model, the system converts candidate profiles and job requirements into dense, 384-dimensional vector embeddings. It understands contextual meaning rather than relying on strict keyword matches.



2\. Thematic Paragraph Match (Doc2Vec)



An optimized Doc2Vec network is trained dynamically in Distributed Bag of Words (PV-DBOW) mode. This model assesses document themes, phrasing density, and vocabulary layout profiles to measure structural alignment.



3\. Cosine Similarity Measurement



Once the text is converted into multi-dimensional vectors, the system computes the spatial angle between the job description vector ($A$) and each resume vector ($B$):



$$Cosine Similarity = \\frac{A \\cdot B}{\\Vert{}A\\Vert{} \\Vert{}B\\Vert{}}$$



Both model metrics are integrated to output an objective Combined Match Score on which candidates are instantly ranked.



🛠️ Technology Stack



Backend Framework: Flask (Python)



Production WSGI Server: Gunicorn



Deep Learning \& Transformers: PyTorch, Sentence-Transformers (BERT)



Paragraph Embeddings: Gensim (Doc2Vec)



PDF Utility: PyPDF



Mathematical Operations: NumPy



Frontend UI: HTML5, Tailwind CSS, FontAwesome



💻 Local Installation \& Setup



If you want to run this project locally on your machine, follow these steps:



Prerequisites



Make sure you have Python 3.9+ installed.



1\. Clone the Repository



git clone https://github.com/afnanp901-beep/ai-resume-screener.git

cd ai-resume-screener





2\. Install PyTorch (CPU Version)



Installing the lightweight CPU build of PyTorch keeps the installation fast and clean:



pip install torch --extra-index-url https://download.pytorch.org/whl/cpu





3\. Install Dependencies



pip install -r requirements.txt





4\. Run the Flask Server



python app.py





Open your browser and navigate to http://127.0.0.1:5000 to start screening resumes!



☁️ Public Cloud Deployment (Render)



This project is configured to run smoothly on the free cloud tier of Render.com linked directly to your GitHub repository.



Configuration Settings



When setting up a new Web Service on Render, apply these configurations:



Runtime: Python



Build Command: pip install -r requirements.txt



Start Command: gunicorn app:app



Instance Type: Free



Note: Since the continuous deployment pipeline is configured via GitHub, any changes pushed to the main branch will instantly rebuild and update your live website.

