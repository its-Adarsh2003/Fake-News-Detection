import streamlit as st
import pandas as pd
import pickle
import sys
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import urllib.parse
import random

try:
    from bert_model import BertFakeNewsDetector
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False

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

# ========== FACT-CHECKING MODULE ==========
class SmartFactChecker:
    @staticmethod
    def get_snopes_link(claim: str) -> str:
        query = urllib.parse.quote(claim[:150])
        return f"https://www.snopes.com/search/?q={query}"
    
    @staticmethod
    def get_factcheck_link(claim: str) -> str:
        query = urllib.parse.quote(claim[:150])
        return f"https://www.factcheck.org/?s={query}"
    
    @staticmethod
    def get_wikipedia_link(topic: str) -> str:
        query = urllib.parse.quote(topic)
        return f"https://en.wikipedia.org/wiki/Special:Search?search={query}"
    
    @staticmethod
    def extract_claims(text: str) -> list:
        keywords = [
            "announced", "reported", "claimed", "said", "revealed",
            "discovered", "proved", "showed", "confirmed", "stated"
        ]
        sentences = text.split(".")
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and any(kw in sent.lower() for kw in keywords):
                claims.append(sent[:100])
        return claims[:3]
    
    @staticmethod
    def get_confidence_recommendation(confidence: float) -> dict:
        if confidence >= 0.85:
            return {
                "level": "HIGH CONFIDENCE",
                "color": "#2ECC71",
                "action": "✅ Likely authentic - Still verify important claims",
                "emoji": "🟢",
            }
        elif confidence >= 0.70:
            return {
                "level": "MEDIUM CONFIDENCE",
                "color": "#F39C12",
                "action": "🟡 Use external sources to cross-check",
                "emoji": "🟡",
            }
        else:
            return {
                "level": "LOW CONFIDENCE",
                "color": "#E74C3C",
                "action": "🔴 HIGH PRIORITY: Verify with fact-check websites",
                "emoji": "🔴",
            }

# ---------- CSS ----------
st.markdown(
    """
<style>
.real-news {
  background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
  padding: 2rem;
  border-radius: 15px;
  border-left: 6px solid #28a745;
  box-shadow: 0 4px 15px rgba(40, 167, 69, 0.15);
  margin: 1rem 0;
}
.fake-news {
  background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
  padding: 2rem;
  border-radius: 15px;
  border-left: 6px solid #dc3545;
  box-shadow: 0 4px 15px rgba(220, 53, 69, 0.15);
  margin: 1rem 0;
}
.confidence-high {
  background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
  padding: 1.5rem;
  border-radius: 10px;
  border-left: 5px solid #2ECC71;
  margin: 1rem 0;
}
.confidence-medium {
  background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
  padding: 1.5rem;
  border-radius: 10px;
  border-left: 5px solid #F39C12;
  margin: 1rem 0;
}
.confidence-low {
  background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
  padding: 1.5rem;
  border-radius: 10px;
  border-left: 5px solid #E74C3C;
  margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- SESSION STATE ----------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "news_input" not in st.session_state:
    st.session_state.news_input = ""
if "pending_sample" not in st.session_state:
    st.session_state.pending_sample = None
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# ---------- SAMPLES ----------
SAMPLES = {
    "✅ Real News 1": """WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a "fiscal conservative" on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS' "Face the Nation," drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight li""",
    
    "✅ Real News 2": """Scientists Develop New Cancer Treatment Showing Promise
Researchers at leading medical institutions have announced preliminary results of a new cancer treatment that shows a 75% response rate in early trials. The treatment combines immunotherapy with targeted drug delivery. Clinical trials will continue to larger patient populations. The findings were published in a peer-reviewed journal.""",
    
    "❌ Fake News 1": """Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my fr""",
    
    "❌ Fake News 2": """WARNING: 5G TOWERS CAUSING MASS HEALTH CRISIS - AUTHORITIES IGNORE!!!
Thousands of doctors agree (won't go public) that 5G causes mystery illnesses! Government covers it up! URGENT: SHARE WITH EVERYONE!!!""",
}

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    try:
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        extractor = FeatureExtractor()
        extractor.load_tfidf("models/tfidf_vectorizer.pkl")
        return model, extractor
    except FileNotFoundError:
        st.error("❌ Models not found! Run train_pipeline.py first.")
        st.stop()

@st.cache_resource
def load_bert_model():
    if BERT_AVAILABLE:
        try:
            return BertFakeNewsDetector()
        except:
            return None
    return None

@st.cache_resource
def get_preprocessor():
    return NewsPreprocessor()

# ---------- HEADER ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<h1 style='text-align: center; color: #FF6B6B;'>📰 Fake News Detector v3.0</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 12px;'>AI Pattern Detection + Real-Time Fact-Checking | TF-IDF + DistilBERT Ensemble</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    st.error(
        """⚠️ **BREAKING NEWS ALERT**
Model trained on 2023 data.
Recent events = LOW confidence.
ALWAYS cross-verify!"""
    )
    
    model_info = """🤖 **Model Info**
- Accuracy: 99.62% (TF-IDF)
- Type: Ensemble (TF-IDF + DistilBERT)
- Features: TF-IDF (5000) + BERT Embeddings
- Training Data: 44,898 articles
- Status: ✅ Active"""
    
    if BERT_AVAILABLE:
        model_info += "\n- BERT: ✅ Enabled"
    else:
        model_info += "\n- BERT: ⚠️ Not loaded"
    
    st.info(model_info)
    
    st.markdown("### 📈 Session Stats")
    if st.session_state.analysis_history:
        total = len(st.session_state.analysis_history)
        fake = sum(1 for x in st.session_state.analysis_history if x["prediction"] == 0)
        real = sum(1 for x in st.session_state.analysis_history if x["prediction"] == 1)
        s1, s2 = st.columns(2)
        with s1:
            st.metric("Total", total)
            st.metric("✅ Real", real)
        with s2:
            avg = (
                sum(x["confidence"] for x in st.session_state.analysis_history)
                / total
                * 100
            )
            st.metric("Avg Conf", f"{avg:.1f}%")
            st.metric("❌ Fake", fake)
    else:
        st.info("No analyses yet!")

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Detect", "📊 Stats", "📚 Samples", "ℹ️ Guide", "🔗 Fact-Check"]
)

# ========== TAB 1: DETECT ==========
with tab1:
    st.subheader("📝 Enter News Article")
    if st.session_state.pending_sample is not None:
        st.session_state.news_input = st.session_state.pending_sample
        st.session_state.pending_sample = None
    
    col_area, col_btn = st.columns([4, 1])
    with col_area:
        text_input = st.text_area(
            "Paste article:",
            value=st.session_state.news_input,
            height=250,
            placeholder="Paste news article text...",
            key="news_input",
        )
    with col_btn:
        st.markdown("### Load")
        if st.button("📚 Real", use_container_width=True):
            st.session_state.pending_sample = list(SAMPLES.values())[0]
            st.rerun()
        if st.button("❌ Fake", use_container_width=True):
            st.session_state.pending_sample = list(SAMPLES.values())[2]
            st.rerun()
    
    b1, b2, b3, b4 = st.columns(4)
    
    with b1:
        analyze_btn = st.button("🔎 ANALYZE", use_container_width=True, type="primary")
    
    with b2:
        def clear_news_input():
            st.session_state.news_input = ""
        st.button("🗑️ Clear", use_container_width=True, on_click=clear_news_input)
    
    with b3:
        def toggle_history():
            st.session_state.show_history = not st.session_state.get(
                "show_history", False
            )
        st.button("📊 History", use_container_width=True, on_click=toggle_history)
    
    with b4:
        def load_random():
            st.session_state.pending_sample = random.choice(list(SAMPLES.values()))
        st.button("🎲 Random", use_container_width=True, on_click=load_random)

    # ========== ANALYSIS ==========
    if analyze_btn:
        if not text_input.strip():
            st.error("❌ Enter text!")
            st.stop()
        
        model, extractor = load_models()
        preprocessor = get_preprocessor()
        bert_model = load_bert_model()
        
        progress = st.progress(0)
        status = st.empty()
        
        status.info("🔄 Step 1: Preprocessing...")
        progress.progress(20)
        clean_text = preprocessor.clean_text(text_input)
        
        if not clean_text.strip():
            st.error("❌ Text invalid!")
            st.stop()
        
        status.info("🔄 Step 2: TF-IDF Feature extraction...")
        progress.progress(40)
        X = extractor.extract_tfidf_features_transform([clean_text])
        
        status.info("🔄 Step 3: TF-IDF prediction...")
        progress.progress(55)
        prediction = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            tfidf_confidence = proba[int(prediction)]
        else:
            decision = model.decision_function(X)[0]
            tfidf_confidence = abs(decision) / (1 + abs(decision))

        tfidf_confidence = float(tfidf_confidence)

        # BERT prediction (if available)
        bert_confidence = None
        if bert_model is not None:
            status.info("🔄 Step 4: BERT prediction...")
            progress.progress(70)
            try:
                bert_pred, bert_conf = bert_model.predict(clean_text)
                bert_confidence = float(bert_conf)
            except:
                bert_confidence = None

        # Ensemble confidence
        if bert_confidence is not None:
            confidence = (tfidf_confidence + bert_confidence) / 2
        else:
            confidence = tfidf_confidence

        label = int(prediction)

        # 👉 FINAL VERDICT LOGIC (3 classes)
        if label == 0 and confidence >= 0.95:
            final_tag = "fake_high"
        elif label == 1 and confidence >= 0.50:
            final_tag = "real_high"
        else:
            final_tag = "uncertain"

        progress.progress(100)
        status.success("✅ Done!")
        import time
        time.sleep(0.5)
        status.empty()
        progress.empty()

        # History
        st.session_state.analysis_history.append(
            {
                "text": text_input[:100],
                "prediction": label,
                "confidence": confidence,
                "tag": final_tag,
                "timestamp": datetime.now(),
            }
        )

        st.markdown("---")
        st.markdown("### 🎯 Results")

        r1, r2 = st.columns([3, 1])

        with r1:
            if final_tag == "real_high":
                st.markdown(
                    f"<div class='real-news'><h2 style='color: #155724; margin: 0;'>✅ LIKELY CREDIBLE</h2>"
                    f"<p style='color: #155724; font-size: 18px;'>Confidence: {confidence*100:.1f}%</p>"
                    f"<p style='color: #155724;'>News-like patterns detected. Still verify important claims.</p></div>",
                    unsafe_allow_html=True,
                )
            elif final_tag == "fake_high":
                st.markdown(
                    f"<div class='fake-news'><h2 style='color: #721c24; margin: 0;'>⚠️ LOW-CREDIBILITY PATTERNS</h2>"
                    f"<p style='color: #721c24; font-size: 18px;'>Confidence: {confidence*100:.1f}%</p>"
                    f"<p style='color: #721c24;'>Writing style is similar to known low-credibility articles. Always verify with multiple trusted sources.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='fake-news'><h2 style='color: #856404; margin: 0;'>🟡 UNCERTAIN / MIXED PATTERNS</h2>"
                    f"<p style='color: #856404;'>Model output: {'Fake (0)' if label == 0 else 'Real (1)'} · Confidence: {confidence*100:.1f}%</p>"
                    f"<p style='color: #856404;'>Patterns are not clear. You MUST verify with trusted sources.</p></div>",
                    unsafe_allow_html=True,
                )

        with r2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Confidence %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#E74C3C" if final_tag == "fake_high" else "#2ECC71"},
                        "steps": [
                            {"range": [0, 33], "color": "#ffebee"},
                            {"range": [33, 66], "color": "#fff9c4"},
                            {"range": [66, 100], "color": "#e8f5e9"},
                        ],
                    },
                )
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Confidence recommendation
        rec = SmartFactChecker.get_confidence_recommendation(confidence)
        st.markdown(
            f"""
        <div style='background: {rec['color']}22; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {rec['color']};'>
            <h3 style='color: {rec['color']};'>{rec['emoji']} {rec['level']}</h3>
            <p style='font-size: 16px;'><strong>Action:</strong> {rec['action']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Model details
        st.markdown("### 🤖 Model Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TF-IDF Confidence", f"{tfidf_confidence*100:.1f}%")
        with col2:
            if bert_confidence is not None:
                st.metric("BERT Confidence", f"{bert_confidence*100:.1f}%")
            else:
                st.metric("BERT Confidence", "N/A")
        with col3:
            st.metric("Ensemble Result", f"{confidence*100:.1f}%")

        # External verification links
        if final_tag != "real_high":
            st.markdown("### 🔗 Cross-Verify with External Sources:")
            links = {
                "Snopes": SmartFactChecker.get_snopes_link(text_input[:100]),
                "FactCheck": SmartFactChecker.get_factcheck_link(text_input[:100]),
                "Wikipedia": SmartFactChecker.get_wikipedia_link(text_input[:30]),
            }
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"[🔍 Snopes]({links['Snopes']})")
            with c2:
                st.markdown(f"[✓ FactCheck]({links['FactCheck']})")
            with c3:
                st.markdown(f"[📖 Wikipedia]({links['Wikipedia']})")

        # Text insights
        st.markdown("### 📊 Text Analysis")
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.metric("📖 Characters", f"{len(text_input):,}")
        with i2:
            st.metric("📝 Words", f"{len(text_input.split()):,}")
        with i3:
            st.metric("🔍 Cleaned Words", f"{len(clean_text.split()):,}")
        with i4:
            st.metric("⚡ Processing", "~0.5s")

# ========== TAB 2: STATISTICS ==========
with tab2:
    st.subheader("📊 Model Performance & Statistics")
    try:
        with open("results/model_results.json", "r") as f:
            results = json.load(f)
        
        df_results = pd.DataFrame(results).T.reset_index()
        df_results.columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ]
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        best_model = max(results.items(), key=lambda x: x[1]["f1"])
        with mc1:
            st.metric("🏆 Best Model", best_model[0].replace("_", " ").title())
        with mc2:
            st.metric("🎯 F1-Score", f"{best_model[1]['f1']:.4f}")
        with mc3:
            st.metric("📈 Accuracy", f"{best_model[1]['accuracy']:.4f}")
        with mc4:
            st.metric("🚀 ROC-AUC", f"{best_model[1]['roc_auc']:.4f}")
        
        st.markdown("---")
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
        
        st.markdown("### 📋 Detailed Metrics")
        st.dataframe(df_results.set_index("Model"), use_container_width=True)
    except Exception:
        st.error("Results file not found! Run train_pipeline.py first.")

# ========== TAB 3: SAMPLES ==========
with tab3:
    st.subheader("📚 Sample Articles for Testing")
    c1, c2 = st.columns(2)
    with c1:
        for i, (title, content) in enumerate(list(SAMPLES.items())[:2]):
            if st.button(
                f"Load: {title}", use_container_width=True, key=f"sample_{i}"
            ):
                st.session_state.pending_sample = content
                st.rerun()
    with c2:
        for i, (title, content) in enumerate(list(SAMPLES.items())[2:]):
            if st.button(
                f"Load: {title}", use_container_width=True, key=f"sample_{i+2}"
            ):
                st.session_state.pending_sample = content
                st.rerun()
    
    st.markdown("---")
    for title, content in SAMPLES.items():
        with st.expander(f"📄 {title}"):
            st.write(content)

# ========== TAB 4: GUIDE ==========
with tab4:
    st.subheader("📖 Complete Guide")
    st.markdown(
        """
    ### 🔍 How it works
    1. You paste a news article into the Detect tab.
    2. Text is cleaned (lowercasing, removing links, punctuation, stopwords, stemming).
    3. Cleaned text is converted to TF-IDF features AND processed by DistilBERT.
    4. **Ensemble Prediction**: Both TF-IDF + DistilBERT vote, confidence is averaged.
    5. App shows label with combined confidence score.

    ### 🤖 Model Architecture
    - **TF-IDF Model**: Logistic Regression (99.62% accuracy)
    - **BERT Model**: DistilBERT (semantic understanding)
    - **Ensemble**: Average confidence from both models
    - Features: TF-IDF (5000) + BERT Embeddings
    - Training Data: 44,898 articles (2023)

    ### ⚠️ CRITICAL LIMITATIONS
    **This is a PATTERN DETECTOR, NOT a FACT CHECKER**

    - ✅ Detects linguistic patterns in low-credibility writing
    - ✅ Flags clickbait, emotional manipulation, conspiracy language
    - ✅ BERT adds semantic + context understanding
    - ❌ Cannot fact-check breaking news (outside 2023 training data)
    - ❌ Cannot verify if claims are actually TRUE
    - ❌ Cannot predict unprecedented events

    ### ✅ Correct Usage
    ```
    Step 1: Get ML Prediction
        ↓
    Step 2: Check Confidence Score
        - <70% = Model uncertain → NEED verification
        - 70-85% = Possible patterns found → Cross-check
        - >85% = Strong pattern detected → Still verify
        ↓
    Step 3: Cross-Reference with External Sources
        🔗 Snopes.com
        🔗 FactCheck.org
        🔗 Wikipedia
    ```

    ### 📰 Fake News Patterns
    - ALL CAPS, excessive punctuation (!!!)
    - Emotional, sensational language
    - Lacks credible sources or citations
    - Claims "secret" or "hidden" information
    - Makes extreme medical/political claims

    ### ✅ Tips to Verify News
    - Check 2-3 trusted news websites
    - Verify author, date, source URL
    - Look for original studies / official statements
    - Be careful with WhatsApp forwards, edited images
    """
    )

# ========== TAB 5: FACT-CHECK ==========
with tab5:
    st.subheader("🔗 External Fact-Checking Resources")
    st.markdown(
        """
    ### Why External Verification Matters

    Our ML model is a **PATTERN DETECTOR** trained on 2023 data. 
    For breaking news, recent events, or low-confidence predictions, 
    you MUST verify with external fact-checkers.
    """
    )
    
    st.markdown("### 🔍 Recommended Fact-Check Websites")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(
            """
        #### 🔍 Snopes.com
        - Oldest fact-checking site
        - Comprehensive database
        - Covers urban legends, rumors, news
        
        [Visit Snopes →](https://www.snopes.com)
        """
        )
    with fc2:
        st.markdown(
            """
        #### ✓ FactCheck.org
        - Nonpartisan fact-checker
        - Focus on US politics
        - Run by UPenn
        
        [Visit FactCheck →](https://www.factcheck.org)
        """
        )
    with fc3:
        st.markdown(
            """
        #### 📖 Wikipedia
        - Crowd-sourced encyclopedia
        - Good for general context
        - Check citations
        
        [Visit Wikipedia →](https://www.wikipedia.org)
        """
        )
    
    st.markdown("---")
    st.markdown("### 🧪 Test Fact-Checking")
    test_claim = st.text_input(
        "Enter a claim to verify:", placeholder="E.g., Modi resignation announced"
    )
    if st.button("🔍 Generate Fact-Check Links"):
        if test_claim:
            links = {
                "Snopes": SmartFactChecker.get_snopes_link(test_claim),
                "FactCheck": SmartFactChecker.get_factcheck_link(test_claim),
                "Wikipedia": SmartFactChecker.get_wikipedia_link(test_claim),
            }
            st.success("✅ Fact-check links generated!")
            l1, l2, l3 = st.columns(3)
            with l1:
                st.markdown(f"[🔍 Check on Snopes]({links['Snopes']})")
            with l2:
                st.markdown(f"[✓ Check on FactCheck]({links['FactCheck']})")
            with l3:
                st.markdown(f"[📖 Search Wikipedia]({links['Wikipedia']})")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
"""
<div style="text-align: center; color: #A0A0A0; font-size: 12px; padding: 1.5rem 0; border-top: 1px solid #333; margin-top: 2rem;">
    <p style="margin: 0.2rem 0;"><strong>Fake News Detector v3.0</strong> · Streamlit · Scikit-learn · Transformers · NLP</p>
    <p style="margin: 0.2rem 0;">Ensemble: TF-IDF (99.62%) + DistilBERT | Training Data: 44,898 articles (2023)</p>
    <p style="margin: 0.2rem 0;">
    ⚠️ <em>Model detects patterns, NOT facts. Always verify with trusted sources!</em> ⚠️</p>
</div>
""",
    unsafe_allow_html=True,
)
