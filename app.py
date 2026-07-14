import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("placement_model (1).pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================


st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}

[data-testid="stMetricLabel"] {
    color: white !important;
    font-size: 18px;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 30px;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg,#1d4ed8,#6d28d9);
}

h1 {
    color: #1e293b;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.title("🎓 AI-Powered Student Placement Prediction")

st.info(
    "Predict placement chances using Machine Learning, technical skills, projects and academic performance."
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("🎓 Student Details")

branch = st.sidebar.selectbox(
    "Branch",
    ["CSE", "IT", "ECE", "ME", "Civil"]
)

college_tier = st.sidebar.selectbox(
    "College Tier",
    [1, 2, 3]
)

cgpa = st.sidebar.slider(
    "CGPA",
    5.0,
    10.0,
    7.0
)

backlogs = st.sidebar.slider(
    "Backlogs",
    0,
    10,
    0
)

coding_skills = st.sidebar.slider(
    "Coding Skills",
    0.0,
    10.0,
    5.0
)

dsa_score = st.sidebar.slider(
    "DSA Score",
    0.0,
    10.0,
    5.0
)

aptitude_score = st.sidebar.slider(
    "Aptitude Score",
    0.0,
    100.0,
    50.0
)

communication_skills = st.sidebar.slider(
    "Communication Skills",
    0.0,
    10.0,
    5.0
)

ml_knowledge = st.sidebar.slider(
    "ML Knowledge",
    0.0,
    10.0,
    5.0
)

system_design = st.sidebar.slider(
    "System Design",
    0.0,
    10.0,
    5.0
)

internships = st.sidebar.slider(
    "Internships",
    0,
    10,
    1
)

projects_count = st.sidebar.slider(
    "Projects",
    0,
    20,
    2
)

certifications = st.sidebar.slider(
    "Certifications",
    0,
    20,
    1
)

hackathons = st.sidebar.slider(
    "Hackathons",
    0,
    20,
    1
)

open_source_contributions = st.sidebar.slider(
    "Open Source Contributions",
    0,
    20,
    0
)

extracurriculars = st.sidebar.slider(
    "Extracurricular Activities",
    0,
    10,
    1
)

# =========================
# BRANCH ENCODING
# =========================

branch_map = {
    "CSE": 0,
    "IT": 1,
    "ECE": 2,
    "ME": 3,
    "Civil": 4
}

branch = branch_map[branch]

# =========================
# PREDICT BUTTON
# =========================

if st.button("🚀 Predict Placement"):

    data = [[
        branch,
        college_tier,
        cgpa,
        backlogs,
        coding_skills,
        dsa_score,
        aptitude_score,
        communication_skills,
        ml_knowledge,
        system_design,
        internships,
        projects_count,
        certifications,
        hackathons,
        open_source_contributions,
        extracurriculars
    ]]

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    readiness_score = (
        
        (cgpa/10)*25 +
        (coding_skills/10)*20 +
        (dsa_score/10)*20 +
        (communication_skills/10)*15 +
        min(internships,5)*4 +
        min(projects_count,5)*2
)

    readiness_score = round(min(readiness_score,100),0)



    # =====================
    # METRICS
    # =====================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Placement Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Readiness Score",
            f"{readiness_score:.0f}/100"
        )

    with col3:
        st.metric(
            "Projects",
            projects_count
        )

    # =====================
    # RESULT
    # =====================

    if probability >= 0.80:
        st.success("🏆 High Placement Potential")

    elif probability >= 0.60:
        st.info("📈 Moderate Placement Potential")

    else:
        st.warning("⚠️ Low Placement Potential")

    if prediction == 1:
        st.success("🎉 Student Likely To Be Placed")
    else:
        st.error("❌ Student May Not Be Placed")

    # =====================
    # PROGRESS BAR
    # =====================

    st.subheader("📈 Placement Readiness")

    st.progress(int(readiness_score))

    # =====================
    # RADAR CHART
    # =====================

    st.subheader("🎯 Skill Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                coding_skills,
                dsa_score,
                communication_skills,
                ml_knowledge,
                system_design
            ],
            theta=[
                "Coding",
                "DSA",
                "Communication",
                "ML",
                "System Design"
            ],
            fill="toself"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False
    )

    st.plotly_chart( fig,  width="stretch")

    # =====================
    # RECOMMENDATIONS
    # =====================

    st.subheader("💡 Recommendations")

    if coding_skills < 6:
        st.warning("Improve Coding Skills")

    if dsa_score < 6:
        st.warning("Practice DSA Regularly")

    if communication_skills < 6:
        st.warning("Improve Communication Skills")

    if internships == 0:
        st.warning("Complete At Least One Internship")

    if projects_count < 3:
        st.warning("Build More Projects")

    # =====================
    # STRENGTHS
    # =====================

    st.subheader("💪 Strengths")

    strengths = []

    if coding_skills >= 7:
        strengths.append("Coding Skills")

    if dsa_score >= 7:
        strengths.append("DSA")

    if communication_skills >= 7:
        strengths.append("Communication Skills")

    if internships >= 1:
        strengths.append("Industry Exposure")

    if projects_count >= 3:
        strengths.append("Project Experience")

    if len(strengths) > 0:
        for item in strengths:
            st.success(f"✔ {item}")

    # =====================
    # IMPROVEMENT AREAS
    # =====================

    st.subheader("📌 Areas To Improve")

    weaknesses = []

    if coding_skills < 6:
        weaknesses.append("Coding Skills")

    if dsa_score < 6:
        weaknesses.append("DSA")

    if communication_skills < 6:
        weaknesses.append("Communication Skills")

    if internships == 0:
        weaknesses.append("Internships")

    if projects_count < 3:
        weaknesses.append("Projects")

    if len(weaknesses) > 0:
        for item in weaknesses:
            st.error(f"✖ {item}")

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Built with Streamlit • Scikit-Learn • Machine Learning"
)