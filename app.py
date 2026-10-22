import streamlit as st
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI CareerPilot",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #2855a6);
    color: white;
    margin-bottom: 25px;
}
.hero h1 { color: white; }
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e4e8f0;
}
</style>
""", unsafe_allow_html=True)

SKILLS = [
    "python", "java", "javascript", "html", "css",
    "react", "node.js", "sql", "mysql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "data analysis", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "computer vision", "nlp",
    "streamlit", "flask", "django", "git", "github",
    "communication", "teamwork", "problem solving",
    "project management", "excel", "power bi", "tableau",
    "data visualization", "rest api", "fastapi", "aws",
    "docker", "linux", "statistics"
]


def extract_pdf(uploaded_file):
    text = ""
    document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in document:
        text += page.get_text() + "\n"
    document.close()
    return text


def find_skills(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(skill.title())
    return sorted(set(found))


def calculate_match(resume, job_description):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([resume, job_description])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 1)


def open_resume_analyzer():
    """Callback for the Dashboard button; runs before widgets are instantiated."""
    st.session_state["page"] = "Resume Analyzer"


st.sidebar.title("🚀 AI CareerPilot")

if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Resume Analyzer", "About"],
    key="page"
)

st.sidebar.info(
    "Build your career with smarter resume insights."
)

if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>🚀 AI CareerPilot</h1>
        <p>Your personal Resume Analyzer & Job Matching Assistant</p>
        <p>Discover skills, compare your resume with a job, and plan your next steps.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔎 Resume Analysis", "Available")
    c2.metric("🎯 Job Matching", "Available")
    c3.metric("🧠 Skill Insights", "Available")

    st.subheader("Get started")
    st.write(
        "Upload a PDF resume and paste a job description "
        "to discover matching skills and areas to improve."
    )

    st.button(
        "Analyze My Resume",
        type="primary",
        key="dashboard_analyze",
        on_click=open_resume_analyzer
    )

elif page == "Resume Analyzer":
    st.title("📄 Resume Analyzer")
    st.write("Upload your resume and compare it with your target job.")

    uploaded_file = st.file_uploader(
        "Upload resume (PDF only)",
        type=["pdf"]
    )

    job_description = st.text_area(
        "Paste job description",
        height=220,
        placeholder="Paste the job requirements here..."
    )

    if st.button("Analyze Resume", type="primary"):
        if uploaded_file is None:
            st.error("Please upload your resume PDF first.")
        elif not job_description.strip():
            st.error("Please paste a job description.")
        else:
            try:
                resume_text = extract_pdf(uploaded_file)

                if len(resume_text.strip()) < 30:
                    st.error(
                        "Resume text nahi mil raha. "
                        "Agar PDF scanned image hai, text-based PDF try karein."
                    )
                else:
                    resume_skills = find_skills(resume_text)
                    job_skills = find_skills(job_description)
                    matched = sorted(
                        set(resume_skills) & set(job_skills)
                    )
                    missing = sorted(
                        set(job_skills) - set(resume_skills)
                    )
                    score = calculate_match(
                        resume_text, job_description
                    )

                    st.session_state["analysis"] = {
                        "score": score,
                        "resume_skills": resume_skills,
                        "job_skills": job_skills,
                        "matched": matched,
                        "missing": missing,
                        "resume_text": resume_text
                    }

            except ValueError:
                st.error("PDF file read nahi ho saki. Doosri PDF try karein.")
            except Exception as error:
                st.error(f"Analysis mein issue aaya: {error}")

    result = st.session_state.get("analysis")

    if result:
        st.divider()
        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Text Similarity", f'{result["score"]}%')
        col2.metric("Matched Skills", len(result["matched"]))
        col3.metric("Skills to Learn", len(result["missing"]))

        st.caption(
            "Note: Text similarity is not an official ATS score, "
            "job probability, or guarantee of selection."
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### ✅ Matching Skills")
            if result["matched"]:
                for skill in result["matched"]:
                    st.write(f"✓ {skill}")
            else:
                st.write("No matching skills detected from the built-in list.")

        with right:
            st.markdown("#### 📚 Skills to Improve")
            if result["missing"]:
                for skill in result["missing"]:
                    st.write(f"• {skill}")
            else:
                st.write("No missing skills detected from the built-in list.")

        if result["job_skills"]:
            chart = go.Figure(data=[
                go.Bar(
                    x=["Matched", "Missing"],
                    y=[
                        len(result["matched"]),
                        len(result["missing"])
                    ]
                )
            ])
            chart.update_layout(
                title="Job Skills Comparison",
                yaxis_title="Number of Skills",
                template="plotly_white"
            )
            st.plotly_chart(chart, use_container_width=True)

        st.subheader("🗓️ Suggested 30-Day Learning Plan")

        if result["missing"]:
            for i, skill in enumerate(result["missing"][:4]):
                start = i * 7 + 1
                end = min(start + 6, 30)
                st.write(f"**Days {start}–{end}:** Learn and practise {skill}.")
            st.write(
                "**Days 29–30:** Update your resume and build a small "
                "project using the skills you learned."
            )
        else:
            st.write(
                "Choose one job skill to deepen and build a portfolio project."
            )

        st.subheader("💡 Resume Improvement Tips")
        tips = [
            "Add measurable achievements where possible.",
            "Use clear headings: Education, Skills, Projects, Experience.",
            "Include relevant projects and the technologies you used.",
            "Customize your resume for each job description.",
            "Proofread spelling, grammar, dates, and contact details."
        ]
        for tip in tips:
            st.write(f"• {tip}")

elif page == "About":
    st.title("About AI CareerPilot")
    st.write("""
    AI CareerPilot is a beginner-friendly career support project.

    **Technologies**
    - Python
    - Streamlit
    - PyMuPDF for PDF text extraction
    - Scikit-learn for TF-IDF text similarity
    - Plotly for interactive charts

    **Limitations**
    - Skill detection uses a built-in keyword list.
    - Similarity is a text comparison, not an official ATS rating.
    - Scanned PDFs may not work without OCR.
    - Results are suggestions, not hiring decisions.
    """)
    st.info(
        "Privacy tip: use a sample resume for public demos. "
        "Avoid uploading sensitive personal information."
    )

st.sidebar.caption("Built with Python & Streamlit")
