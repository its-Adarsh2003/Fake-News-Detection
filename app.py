import streamlit as st
import pandas as pd
import pickle
import sys
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import NewsPreprocessor
from features import FeatureExtractor

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- GLOBAL CSS ----------
st.markdown(
    """
    <style>
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #2ECC71;
        --warning-color: #F39C12;
    }
    .real-news-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #28a745;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.15);
        margin: 1rem 0;
    }
    .fake-news-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #dc3545;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.15);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SESSION STATE ----------
# ---- SESSION STATE ----
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "news_input" not in st.session_state:
    st.session_state.news_input = ""
if "pending_sample" not in st.session_state:
    st.session_state.pending_sample = None
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

# ---------- SAMPLES ----------
SAMPLES = {
    "Real News 1": """Federal Reserve Announces Interest Rate Decision

The Federal Reserve announced today that it will maintain interest rates at the current level following a two-day policy meeting. The decision comes as inflation shows signs of cooling while economic growth remains moderate. Fed Chair noted that the committee will continue to monitor economic data closely. The decision was unanimous among all voting members.""",
    "Real News 2": """Scientists Develop New Cancer Treatment Showing Promise

Researchers at leading medical institutions have announced preliminary results of a new cancer treatment that shows a 75% response rate in early trials. The treatment combines immunotherapy with targeted drug delivery. Clinical trials will continue to larger patient populations. The findings were published in a peer-reviewed journal.""",
    "Fake News 1": """SHOCKING REVELATION!!! GOVERNMENT HIDING ANTI-AGING SERUM FROM PUBLIC!!!

Big Pharma doesn't want you to know this ONE TRICK that will keep you young FOREVER!!! Celebrities use this secret discovered 20 years ago but THEY don't want you to know!!! SHARE THIS NOW before the government DELETES IT!!! Click link below!!!""",
    "Fake News 2": """WARNING: 5G TOWERS CAUSING MASS HEALTH CRISIS - AUTHORITIES IGNORE EVIDENCE!!!

Thousands of doctors agree (but won't go public) that 5G is causing mysterious illnesses! The government knows but covers it up! Insert your town name: ALREADY AFFECTED! Someone MUST stop this before it's too late! URGENT: SHARE WITH EVERYONE YOU KNOW!!!""",
}

# ---------- HELPERS ----------
@st.cache_resource
def load_models():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    extractor = FeatureExtractor()
    extractor.load_tfidf("models/tfidf_vectorizer.pkl")
    return model, extractor


@st.cache_resource
def get_preprocessor():
    return NewsPreprocessor()

# ---------- HEADER ----------
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown(
        "<h1 style='text-align: center; color: #FF6B6B;'>📰 Fake News Detector</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 14px;'>Advanced AI-powered news authentication system</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 📊 Dashboard")

    st.info(
        """
    ### 🤖 Model Information
    - **Algorithm**: Random Forest Classifier
    - **Accuracy**: 99.62%
    - **Training Data**: 44,898 articles
    - **Features**: TF-IDF (5000)
    - **Status**: ✅ Active
    """
    )

    st.markdown("### 📈 Session Stats")
    if st.session_state.analysis_history:
        total_analyses = len(st.session_state.analysis_history)
        fake_count = sum(
            1 for x in st.session_state.analysis_history if x["prediction"] == 0
        )
        real_count = sum(
            1 for x in st.session_state.analysis_history if x["prediction"] == 1
        )

        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("Total Analyses", total_analyses)
        with sc2:
            avg_conf = (
                sum(x["confidence"] for x in st.session_state.analysis_history)
                / total_analyses
                * 100
            )
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")

        sc3, sc4 = st.columns(2)
        with sc3:
            st.metric(
                "Real News",
                real_count,
                delta=f"{real_count / total_analyses * 100:.0f}%",
            )
        with sc4:
            st.metric(
                "Fake News",
                fake_count,
                delta=f"{fake_count / total_analyses * 100:.0f}%",
            )
    else:
        st.markdown("*No analyses yet. Start by entering text!*")

    st.markdown("---")

    with st.expander("ℹ️ How It Works"):
        st.markdown(
            """
        1. **Input**: Paste news article text  
        2. **Processing**: Text is cleaned and normalized  
        3. **Feature Extraction**: TF-IDF converts text to numerical features  
        4. **Classification**: ML model predicts real or fake  
        5. **Confidence**: Score shows prediction certainty  
        """
        )

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Detect", "📊 Statistics", "📚 Samples", "ℹ️ Guide"]
)

# ========== TAB 1: DETECT ==========
# ========== TAB 1: DETECT ==========
with tab1:
    st.subheader("📝 Enter News Article")

    # CLEAR FLAG – widget se pehle
    if st.session_state.clear_flag:
        st.session_state.news_input = ""
        st.session_state.clear_flag = False

    # PENDING SAMPLE – widget se pehle
    if st.session_state.pending_sample is not None:
        st.session_state.news_input = st.session_state.pending_sample
        st.session_state.pending_sample = None

    col_area, col_action = st.columns([4, 1])

    with col_area:
        text_input = st.text_area(
            "Paste your news article here:",
            value=st.session_state.news_input,
            height=280,
            placeholder="Copy and paste the full news article text here...",
            key="news_input",
        )

    with col_action:
        st.markdown("### Actions")
        if st.button("📚 Real Example", use_container_width=True, key="btn_real"):
            st.session_state.pending_sample = SAMPLES["Real News 1"]
            st.rerun()

        if st.button("⚠️ Fake Example", use_container_width=True, key="btn_fake"):
            st.session_state.pending_sample = SAMPLES["Fake News 1"]
            st.rerun()

    # Control buttons
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        analyze_btn = st.button(
            "🔎 ANALYZE NEWS",
            key="analyze",
            use_container_width=True,
            type="primary",
            help="Click to analyze the entered text",
        )
    with b2:
        if st.button(
        "🗑️ Clear Text",
            use_container_width=True,
            help="Clear the text area",
        ):
            st.session_state.clear_flag = True
            st.rerun()

    with b3:
        if st.button(
            "📊 Show History",
            use_container_width=True,
            help="View analysis history",
        ):
            st.session_state.show_history = not st.session_state.get(
                "show_history", False
            )

    with b4:
        if st.button(
            "⚡ Random Analysis",
            use_container_width=True,
            help="Load a random sample",
        ):
            import random

            st.session_state.pending_sample = random.choice(list(SAMPLES.values()))
            st.rerun()

    # Run analysis
    if analyze_btn:
        if not text_input.strip():
            st.error("❌ Please enter some text to analyze!")
            st.stop()

        model, extractor = load_models()
        preprocessor = get_preprocessor()

        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        status_text.info("🔄 Step 1: Preprocessing text...")
        progress_bar.progress(25)
        clean_text = preprocessor.clean_text(text_input)

        if not clean_text.strip():
            st.error("❌ Text is too short or contains no valid content.")
            progress_container.empty()
            st.stop()

        status_text.info("🔄 Step 2: Extracting features...")
        progress_bar.progress(50)
        X = extractor.extract_tfidf_features_transform([clean_text])

        status_text.info("🔄 Step 3: Running ML model...")
        progress_bar.progress(75)

        # 1) Model prediction
        prediction = model.predict(X)[0]           # 0 = fake, 1 = real (training ke hisaab se)
        proba = model.predict_proba(X)[0]          # [p_fake, p_real]
        confidence = proba[prediction]

        # 2) Extra rule-based check for very spammy / clickbait text
        text_upper_ratio = sum(1 for c in text_input if c.isupper()) / max(1, len(text_input))
        exclamation_count = text_input.count("!")
        clickbait_words = [
            "shocking", "breaking", "must read", "one trick",
            "share this", "before it's too late", "urgent", "exclusive"
        ]
        lower_text = text_input.lower()
        has_clickbait = any(w in lower_text for w in clickbait_words)

        # Agar model REAL bol raha hai, lekin text bahut spammy lag raha hai → fake mark karo
        if prediction == 1:
            if text_upper_ratio > 0.25 or exclamation_count >= 5 or has_clickbait:
                prediction = 0
                # fake ki prob use karo ya thoda adjust karo
                confidence = max(proba[0], 1 - confidence)

        confidence = float(confidence)

# (optional) debug hata sakta hai ab
# st.write("DEBUG prediction:", prediction)
# st.write("DEBUG proba:", proba)

# (optional) debug hata sakta hai ab
# st.write("DEBUG prediction:", prediction)
# st.write("DEBUG proba:", proba)


        progress_bar.progress(100)
        status_text.success("✅ Analysis Complete!")

        import time

        time.sleep(0.5)
        progress_container.empty()

        st.session_state.analysis_history.append(
            {
                "text": text_input[:100],
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": datetime.now(),
            }
        )

        st.markdown("---")
        st.markdown("### 🎯 Analysis Results")

        r1, r2 = st.columns([3, 1])

        with r1:
            if prediction == 1:
                st.markdown(
                    """
<div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
            padding: 2rem; border-radius: 15px; border-left: 6px solid #28a745;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.15);'>
    <h2 style='color: #155724; margin-top: 0;'>✅ LIKELY REAL NEWS</h2>
    <p style='color: #155724; font-size: 18px;'><strong>Confidence Score: """
                    + f"{confidence*100:.1f}%"
                    + """</strong></p>
    <p style='color: #155724;'>This article appears to be authentic based on linguistic and content analysis.</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
<div style='background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
            padding: 2rem; border-radius: 15px; border-left: 6px solid #dc3545;
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.15);'>
    <h2 style='color: #721c24; margin-top: 0;'>⚠️ LIKELY FAKE NEWS</h2>
    <p style='color: #721c24; font-size: 18px;'><strong>Confidence Score: """
                    + f"{confidence*100:.1f}%"
                    + """</strong></p>
    <p style='color: #721c24;'>This article may contain misinformation. Always verify with trusted sources.</p>
</div>
""",
                    unsafe_allow_html=True,
                )

        with r2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Confidence"},
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {
                            "color": "#FF6B6B" if prediction == 0 else "#2ECC71"
                        },
                        "steps": [
                            {"range": [0, 25], "color": "#f0f0f0"},
                            {"range": [25, 50], "color": "#e8f0f0"},
                            {"range": [50, 75], "color": "#e0f8f0"},
                            {"range": [75, 100], "color": "#d0f0e8"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Text Analysis Insights")

        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.metric("📖 Characters", f"{len(text_input):,}")
        with ic2:
            st.metric("📝 Words", f"{len(text_input.split()):,}")
        with ic3:
            st.metric("🔍 Processed Words", f"{len(clean_text.split()):,}")
        with ic4:
            st.metric("⏱️ Processing Time", "~0.5s")

# ========== TAB 2: STATISTICS ==========
with tab2:
    st.subheader("📊 Model Performance & Statistics")
    try:
        with open("results/model_results.json", "r") as f:
            results = json.load(f)

        mc1, mc2, mc3, mc4 = st.columns(4)
        best_model = max(results.items(), key=lambda x: x[1]["f1"])

        with mc1:
            st.metric("🏆 Best Model", best_model[0].replace("_", " ").title())
        with mc2:
            st.metric("🎯 Best F1-Score", f"{best_model[1]['f1']:.4f}")
        with mc3:
            st.metric("📈 Best Accuracy", f"{best_model[1]['accuracy']:.4f}")
        with mc4:
            st.metric("🚀 Best ROC-AUC", f"{best_model[1]['roc_auc']:.4f}")

        st.markdown("---")

        df_results = pd.DataFrame(results).T.reset_index()
        df_results.columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ]

        gc1, gc2 = st.columns(2)
        with gc1:
            fig1 = px.bar(
                df_results,
                x="Model",
                y="F1-Score",
                color="F1-Score",
                color_continuous_scale="RdYlGn",
                title="F1-Score Comparison",
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with gc2:
            fig2 = px.bar(
                df_results,
                x="Model",
                y=["Accuracy", "Precision", "Recall"],
                title="Multi-metric Comparison",
                barmode="group",
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Detailed Metrics")
        st.dataframe(df_results.set_index("Model"), use_container_width=True)
    except Exception:
        st.error("Results file not found! Run train.py first.")

# ========== TAB 3: SAMPLES ==========
with tab3:
    st.subheader("📚 Sample Articles for Testing")

    c1, c2 = st.columns(2)

    with c1:
        for title in list(SAMPLES.keys())[:2]:
            if st.button(
                f"Load: {title}",
                use_container_width=True,
                key=f"sample_{title}",
            ):
                st.session_state.pending_sample = SAMPLES[title]
                st.rerun()

    with c2:
        for title in list(SAMPLES.keys())[2:]:
            if st.button(
                f"Load: {title}",
                use_container_width=True,
                key=f"sample_{title}",
            ):
                st.session_state.pending_sample = SAMPLES[title]
                st.rerun()

    st.markdown("---")

    for title, content in SAMPLES.items():
        with st.expander(f"📄 {title}"):
            st.write(content)

# ========== TAB 4: GUIDE ==========
with tab4:
    st.subheader("📖 Complete Guide to Using This Tool")

    st.markdown("### 🔍 How it works")
    st.markdown(
        """
1. You paste a news article into the Detect tab.  
2. The text is cleaned (lowercasing, removing links, punctuation, stopwords, stemming).  
3. Cleaned text is converted to TF‑IDF features using a trained vocabulary.  
4. The best ML model (Random Forest / Logistic Regression / SVM) predicts **Real (1)** or **Fake (0)**.  
5. The app shows the label with a confidence score and simple rule‑based checks for clickbait.  
"""
    )

    st.markdown("### 🤖 Model details")
    st.markdown(
        """
- Models used: Random Forest, Logistic Regression, Linear SVM (calibrated).  
- Features: TF‑IDF (up to 5000 features, unigrams + bigrams, English stopwords removed).  
- Training data: Combined Fake.csv (label 0) and True.csv (label 1).  
- Best model is chosen by weighted F1‑score and saved as `best_model.pkl`.  
"""
    )

    st.markdown("### 📰 Fake news basics")
    st.markdown(
        """
- Often uses ALL CAPS, many exclamation marks, emotional language.  
- Lacks credible sources, cites “many experts” without names.  
- Makes extreme claims (miracle cures, secret government plots, guaranteed returns).  
"""
    )

    st.markdown("### ✅ Tips to verify news")
    st.markdown(
        """
- Cross‑check the story on at least 2–3 trusted news websites.  
- Check the author, date, and source URL carefully.  
- Look for original studies / official statements instead of only screenshots.  
- Be extra careful with WhatsApp forwards, edited images, and cropped videos.  
"""
    )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
<div style="
    text-align: center;
    color: #A0A0A0;
    font-size: 12px;
    padding: 1.5rem 0;
    border-top: 1px solid #333;
    margin-top: 2rem;
">
    <p style="margin: 0.2rem 0;">
        <strong>Fake News Detector v2.0</strong> · Streamlit · Scikit-learn · NLP
    </p>
    <p style="margin: 0.2rem 0;">
        Developed by AI/ML enthusiast · Approx. model accuracy ~99% (test set) · Last updated: 2026
    </p>
    <p style="margin: 0.2rem 0;">
        For educational and research purposes only. Always verify critical news with trusted sources.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

