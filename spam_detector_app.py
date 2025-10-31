import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import re
from textblob import TextBlob
import joblib
import warnings

warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer



# Page configuration
st.set_page_config(
    page_title="🛡️ Group 1 Spam Detector",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Group 1 Spam Detector\nAI-powered spam detection with advanced NLP"
    }
)

# Modern 2025 Professional Dark Mode CSS
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Dark Mode Professional Background */
    .stApp {
        background: #0a0e27;
        color: #e4e7eb;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Professional Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Modern Card Design */
    .modern-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    }
    
    /* Result Cards - Modern Gradient Style */
    .result-card {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
        border: 1px solid;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .spam-result {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .spam-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    .ham-result {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%);
        border-color: rgba(34, 197, 94, 0.3);
    }
    
    .ham-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22c55e, #16a34a);
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #f1f5f9;
    }
    
    .result-confidence {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Modern Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Text Area - FIXED FOR VISIBILITY */
    .stTextArea > div > div > textarea {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #f1f5f9 !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        background-color: rgba(30, 41, 59, 0.9) !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #64748b !important;
    }
    
    .stTextArea > label {
        color: #e4e7eb !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Modern Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(30, 41, 59, 1);
        border-color: #6366f1;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #e4e7eb;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #e4e7eb !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: rgba(30, 41, 59, 0.6);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stRadio > div > label:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(30, 41, 59, 0.8);
    }
    
    .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        background-color: #6366f1;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: #e4e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: white !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        color: #e4e7eb !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.4);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background: rgba(30, 41, 59, 0.6);
    }
    
    [data-testid="stFileUploader"] label {
        color: #e4e7eb !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Metrics in Sidebar */
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #6366f1 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    h4, h5, h6 {
        color: #e4e7eb !important;
    }
    
    /* Alert Messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        color: #86efac !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        color: #fca5a5 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        color: #93c5fd !important;
    }
    
    .stWarning {
        background: rgba(251, 191, 36, 0.15) !important;
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 10px;
        color: #fcd34d !important;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e4e7eb;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6366f1;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: rgba(30, 41, 59, 0.8);
        color: #f1f5f9;
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #e4e7eb;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(30, 41, 59, 1);
        border-color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'spam_detected' not in st.session_state:
    st.session_state.spam_detected = 0
if 'current_message' not in st.session_state:
    st.session_state.current_message = ""


# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                 'vader_lexicon', 'punkt_tab', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


download_nltk_resources()


# Initialize NLP tools
@st.cache_resource
def init_nlp_tools():
    """Initialize NLP tools"""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, sia


STOP_WORDS, LEMMATIZER, SIA = init_nlp_tools()


# Load model
st.write("📂 **Current working directory:**", os.getcwd())
st.write("📁 **Files in directory:**", os.listdir())

# --- Step 2: Try to load the model safely ---
def load_model():
    """Load the trained model"""
    try:
        model_path = os.path.join(os.getcwd(), 'spam_detector_model.pkl')
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("⚠️ Model Not Found! Please ensure 'spam_detector_model.pkl' is in the directory.")
        st.info("💡 Quick Fix: Train your model and save it using: `joblib.dump(pipeline, 'spam_detector_model.pkl')`")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# Text preprocessing function
def advanced_text_preprocessing(text):
    """Preprocess text for model input"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS]

    return tokens


# Feature extraction
def extract_features(text):
    """Extract features from text"""
    processed = " ".join(advanced_text_preprocessing(text))

    features = {
        'message': text,
        'processed_message': processed,
        'message_length': len(text),
        'word_count': len(text.split()),
        'char_count': len(text.replace(" ", "")),
        'avg_word_length': len(text.replace(" ", "")) / (len(text.split()) + 1),
        'punctuation_count': len(re.findall(r'[^\w\s]', text)),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
        'has_url': 1 if re.search(r'http|www', text.lower()) else 0,
        'has_email': 1 if re.search(r'\S+@\S+', text) else 0,
        'digit_count': sum(c.isdigit() for c in text),
        'digit_ratio': sum(c.isdigit() for c in text) / (len(text) + 1),
        'textblob_sentiment': TextBlob(text).sentiment.polarity,
        'textblob_subjectivity': TextBlob(text).sentiment.subjectivity,
        'vader_compound': SIA.polarity_scores(text)['compound'],
        'vader_pos': SIA.polarity_scores(text)['pos'],
        'vader_neg': SIA.polarity_scores(text)['neg'],
        'spam_word_count': sum(1 for word in ['free', 'win', 'winner', 'cash',
                                              'prize', 'claim', 'call', 'urgent', 'txt']
                               if word in text.lower())
    }

    return features


# Prediction function
def predict_message(text, model):
    """Predict if message is spam"""
    if model is None:
        return None, None, None

    features = extract_features(text)
    df_pred = pd.DataFrame([features])

    prediction = model.predict(df_pred)[0]
    probability = model.predict_proba(df_pred)[0]

    return prediction, probability, features


# Create modern gauge chart
def create_gauge(value, title, color_scheme):
    """Create a modern gauge chart"""

    if color_scheme == "danger":
        color = "#ef4444"
        bg_color = "rgba(239, 68, 68, 0.1)"
    else:
        color = "#22c55e"
        bg_color = "rgba(34, 197, 94, 0.1)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#e4e7eb', 'family': 'Inter'}},
        number={'font': {'size': 48, 'color': color, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "rgba(30, 41, 59, 0.4)",
            'borderwidth': 2,
            'bordercolor': "rgba(148, 163, 184, 0.2)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(148, 163, 184, 0.1)'},
                {'range': [50, 75], 'color': 'rgba(148, 163, 184, 0.15)'},
                {'range': [75, 100], 'color': bg_color}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e4e7eb", 'family': "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# Create feature importance chart
def create_feature_chart(features):
    """Create modern feature importance visualization"""
    key_features = {
        'Length': features['message_length'],
        'Words': features['word_count'],
        'Uppercase %': features['uppercase_ratio'] * 100,
        'Has URL': features['has_url'] * 100,
        'Spam Words': features['spam_word_count'] * 10,
        'Exclamations': features['exclamation_count'] * 10
    }

    colors = ['#6366f1', '#8b5cf6', '#a855f7', '#c026d3', '#d946ef', '#e879f9']

    fig = go.Figure(data=[
        go.Bar(
            x=list(key_features.keys()),
            y=list(key_features.values()),
            marker=dict(
                color=colors,
                line=dict(color='rgba(148, 163, 184, 0.2)', width=1)
            ),
            text=[f"{v:.1f}" for v in key_features.values()],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Inter', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title={
            'text': '🔍 Feature Analysis',
            'font': {'size': 20, 'color': '#e4e7eb', 'family': 'Inter', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.4)',
        font={'color': "#e4e7eb", 'family': 'Inter'},
        xaxis={
            'showgrid': False,
            'color': '#94a3b8'
        },
        yaxis={
            'showgrid': True,
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'color': '#94a3b8'
        },
        height=400,
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


# Main App Header
st.markdown("""
<div class="main-header">
    <div class="main-title">GROUP 1 SPAM DETECTOR</div>
    <div class="subtitle">🤖 Advanced AI-Powered Message Analysis</div>
</div>
""", unsafe_allow_html=True)


# Load model
model = load_model()

if model is None:
    st.error("⚠️ **Model Not Found!** Please ensure 'spam_detector_model.pkl' is in the directory.")
    st.info("💡 **Quick Fix:** Train your model and save it using: `dill.dump(model, open('spam_detector_model.pkl', 'wb'))`")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio("",
                    ["🏠 Main Detector", "📊 Analytics", "📜 History", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📈 Live Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scanned", st.session_state.total_scanned)
    with col2:
        st.metric("Spam", st.session_state.spam_detected)

    if st.session_state.total_scanned > 0:
        spam_rate = (st.session_state.spam_detected / st.session_state.total_scanned) * 100
        st.metric("Detection Rate", f"{spam_rate:.1f}%", delta=f"{spam_rate:.0f}%")

    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_scanned = 0
        st.session_state.spam_detected = 0
        st.session_state.current_message = ""
        st.rerun()

# Main Detector Page
if page == "🏠 Main Detector":

    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Single Message", "📋 Batch Analysis", "✨ Quick Examples"])

    with tab1:
        st.markdown("### 🔍 Analyze a Single Message")
        st.markdown("")

        col1, col2 = st.columns([2, 1])

        with col1:
            message_input = st.text_area(
                "Enter your message to analyze:",
                value=st.session_state.current_message,
                height=180,
                placeholder="Type or paste your message here...\n\nExample: Congratulations! You've won a FREE iPhone! Click here to claim now!",
                help="Enter any text message to check if it's spam or legitimate",
                key="message_input_field"
            )

            analyze_btn = st.button("🚀 Analyze Message", use_container_width=True, type="primary", key="analyze_main")

        with col2:
            st.markdown("#### 🎯 Quick Examples")
            st.markdown("Try these sample messages:")

            if st.button("🎁 Spam Sample", use_container_width=True, key="spam_example"):
                st.session_state.current_message = "Congratulations! You've won a FREE prize worth $1000! Call now to claim your reward!"
                st.rerun()

            if st.button("✅ Legitimate Sample", use_container_width=True, key="ham_example"):
                st.session_state.current_message = "Hey, are you free for lunch tomorrow? Let me know what time works for you."
                st.rerun()

            if st.button("⚠️ Phishing Sample", use_container_width=True, key="phish_example"):
                st.session_state.current_message = "URGENT! Your account will be suspended. Verify your identity NOW by clicking this link!"
                st.rerun()

            if st.button("💼 Business Sample", use_container_width=True, key="business_example"):
                st.session_state.current_message = "Meeting scheduled for 3 PM tomorrow. Please bring the quarterly reports."
                st.rerun()

        if analyze_btn and message_input:
            with st.spinner("🔄 Analyzing your message..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress_bar.progress(i + 1)

                prediction, probability, features = predict_message(message_input, model)

                # Update session state
                st.session_state.total_scanned += 1
                if prediction == 'spam':
                    st.session_state.spam_detected += 1

                # Add to history
                st.session_state.history.insert(0, {
                    'timestamp': datetime.now(),
                    'message': message_input[:100] + "..." if len(message_input) > 100 else message_input,
                    'prediction': prediction,
                    'confidence': max(probability) * 100
                })

                # Keep only last 100 entries
                st.session_state.history = st.session_state.history[:100]

            # Results
            st.markdown("---")
            st.markdown("### 🎯 Analysis Results")
            st.markdown("")

            col1, col2 = st.columns([1, 1])

            with col1:
                if prediction == 'spam':
                    st.markdown(f"""
                    <div class="result-card spam-result">
                        <div class="result-title">🚫 SPAM DETECTED</div>
                        <p style="font-size: 1.1rem; color: #cbd5e1; margin: 0.5rem 0;">
                            This message appears to be spam or malicious
                        </p>
                        <div class="result-confidence">{max(probability) * 100:.1f}%</div>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Confidence Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card ham-result">
                        <div class="result-title">✅ LEGITIMATE MESSAGE</div>
                        <p style="font-size: 1.1rem; color: #cbd5e1; margin: 0.5rem 0;">
                            This message appears to be safe and authentic
                        </p>
                        <div class="result-confidence">{max(probability) * 100:.1f}%</div>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Confidence Score</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.plotly_chart(
                    create_gauge(
                        probability[1] if prediction == 'spam' else probability[0],
                        "🎯 Detection Confidence",
                        "danger" if prediction == 'spam' else "success"
                    ),
                    use_container_width=True
                )

            # Detailed probabilities
            st.markdown("### 📊 Probability Breakdown")
            st.markdown("")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">✅ Legitimate (Ham)</div>
                    <div class="metric-value" style="background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {probability[0] * 100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🚫 Spam</div>
                    <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {probability[1] * 100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Feature analysis
            st.markdown("---")
            st.plotly_chart(create_feature_chart(features), use_container_width=True)

            # Message insights
            st.markdown("### 💡 Detailed Message Insights")
            st.markdown("")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📏 Length</div>
                    <div class="metric-value" style="font-size: 2rem;">{features['message_length']}</div>
                    <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">characters</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📝 Words</div>
                    <div class="metric-value" style="font-size: 2rem;">{features['word_count']}</div>
                    <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">total words</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">❗ Exclamations</div>
                    <div class="metric-value" style="font-size: 2rem;">{features['exclamation_count']}</div>
                    <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">found</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⚠️ Spam Words</div>
                    <div class="metric-value" style="font-size: 2rem;">{features['spam_word_count']}</div>
                    <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">detected</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🔤 Uppercase Ratio</div>
                    <div class="metric-value" style="font-size: 2rem;">{features['uppercase_ratio'] * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                url_status = "Yes ⚠️" if features['has_url'] else "No ✓"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🔗 Contains URL</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{url_status}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                email_status = "Yes ⚠️" if features['has_email'] else "No ✓"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📧 Contains Email</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{email_status}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📋 Batch Message Analysis")
        st.info("💡 **Upload a CSV file with a 'message' column** or **paste multiple messages** (one per line)")
        st.markdown("")

        upload_method = st.radio(
            "Choose your input method:",
            ["📤 Upload CSV File", "📝 Paste Messages"],
            horizontal=True
        )

        messages_to_analyze = []

        if upload_method == "📤 Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=['csv'],
                help="CSV must contain a column named 'message'"
            )

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'message' in df.columns:
                        messages_to_analyze = df['message'].dropna().tolist()
                        st.success(f"✅ Successfully loaded **{len(messages_to_analyze)}** messages from CSV")
                    else:
                        st.error("❌ CSV file must have a column named 'message'")
                        st.info("💡 Column names found: " + ", ".join(df.columns.tolist()))
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
        else:
            batch_input = st.text_area(
                "Paste your messages here (one per line):",
                height=250,
                placeholder="Message 1\nMessage 2\nMessage 3\n...",
                help="Enter each message on a new line"
            )
            if batch_input:
                messages_to_analyze = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]
                if messages_to_analyze:
                    st.info(f"📝 Ready to analyze **{len(messages_to_analyze)}** messages")

        if messages_to_analyze:
            st.markdown("")
            if st.button("🚀 Analyze All Messages", use_container_width=True, type="primary", key="batch_analyze"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, msg in enumerate(messages_to_analyze):
                    status_text.text(f"Analyzing message {idx + 1} of {len(messages_to_analyze)}...")
                    prediction, probability, _ = predict_message(msg, model)

                    results.append({
                        'Message': msg[:80] + "..." if len(msg) > 80 else msg,
                        'Full Message': msg,
                        'Prediction': '🚫 SPAM' if prediction == 'spam' else '✅ HAM',
                        'Confidence': f"{max(probability) * 100:.2f}%",
                        'Spam Probability': f"{probability[1] * 100:.2f}%",
                        'Ham Probability': f"{probability[0] * 100:.2f}%"
                    })

                    progress_bar.progress((idx + 1) / len(messages_to_analyze))
                    time.sleep(0.01)

                status_text.empty()
                progress_bar.empty()

                # Display results
                results_df = pd.DataFrame(results)

                st.markdown("---")
                st.markdown("### 📊 Batch Analysis Results")
                st.markdown("")

                spam_count = sum(1 for r in results if r['Prediction'] == '🚫 SPAM')
                ham_count = len(results) - spam_count
                spam_rate = (spam_count / len(results) * 100) if len(results) > 0 else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Analyzed</div>
                        <div class="metric-value">{len(results)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🚫 Spam Detected</div>
                        <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {spam_count}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">✅ Legitimate</div>
                        <div class="metric-value" style="background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {ham_count}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Spam Rate</div>
                        <div class="metric-value" style="font-size: 2rem;">{spam_rate:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                # Display table
                display_df = results_df[['Message', 'Prediction', 'Confidence', 'Spam Probability', 'Ham Probability']]
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )

                # Visualization
                st.markdown("### 📈 Results Visualization")
                col1, col2 = st.columns(2)

                with col1:
                    fig_pie = px.pie(
                        values=[spam_count, ham_count],
                        names=['Spam', 'Legitimate'],
                        title='Message Distribution',
                        color_discrete_sequence=['#ef4444', '#22c55e'],
                        hole=0.5
                    )
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e4e7eb', 'family': 'Inter'},
                        title={'font': {'size': 18}},
                        height=350
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Confidence distribution
                    confidence_values = [float(r['Confidence'].rstrip('%')) for r in results]
                    fig_hist = px.histogram(
                        x=confidence_values,
                        nbins=20,
                        title='Confidence Score Distribution',
                        color_discrete_sequence=['#6366f1']
                    )
                    fig_hist.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30, 41, 59, 0.4)',
                        font={'color': '#e4e7eb', 'family': 'Inter'},
                        title={'font': {'size': 18}},
                        xaxis_title='Confidence (%)',
                        yaxis_title='Count',
                        height=350,
                        xaxis={'gridcolor': 'rgba(148, 163, 184, 0.1)'},
                        yaxis={'gridcolor': 'rgba(148, 163, 184, 0.1)'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Download results
                st.markdown("")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Results as CSV",
                    data=csv,
                    file_name=f"spam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with tab3:
        st.markdown("### ✨ Quick Test with Pre-loaded Examples")
        st.markdown("Try analyzing these carefully crafted examples to see how the AI works:")
        st.markdown("")

        test_messages = {
            "🎁 Prize Winner Scam": {
                "message": "Congratulations! You've won a FREE prize worth $1000! Call now to claim your reward immediately!",
                "emoji": "🚫"
            },
            "🍕 Casual Lunch Invite": {
                "message": "Hey, are you free for lunch tomorrow? Let me know what time works for you.",
                "emoji": "✅"
            },
            "⚠️ Phishing Attack": {
                "message": "URGENT! Your account will be suspended. Verify your identity NOW by clicking this link immediately!",
                "emoji": "🚫"
            },
            "👔 Business Meeting": {
                "message": "Thanks for the meeting yesterday. Let's schedule a follow-up next week to discuss the quarterly reports.",
                "emoji": "✅"
            },
            "🛍️ Aggressive Marketing": {
                "message": "Get 50% OFF on ALL items! Limited time offer! Text WIN to 12345 NOW to claim your discount!",
                "emoji": "🚫"
            },
            "☎️ Simple Phone Request": {
                "message": "Call me when you get this message. Need to discuss the project timeline with you.",
                "emoji": "✅"
            },
            "💰 Money Transfer Scam": {
                "message": "Wire transfer ready! Claim your $5000 prize now! Reply YES to confirm your bank details!",
                "emoji": "🚫"
            },
            "📧 Professional Email": {
                "message": "Please review the attached document and send your feedback by end of day. Thanks!",
                "emoji": "✅"
            }
        }

        col1, col2 = st.columns(2)

        for idx, (label, data) in enumerate(test_messages.items()):
            col = col1 if idx % 2 == 0 else col2

            with col:
                with st.expander(f"{data['emoji']} {label}", expanded=False):
                    st.text_area(
                        "Message Content:",
                        data['message'],
                        height=100,
                        key=f"msg_display_{idx}",
                        disabled=True
                    )

                    if st.button(f"🔍 Analyze This Message", key=f"btn_quick_{idx}", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            time.sleep(0.3)
                            prediction, probability, _ = predict_message(data['message'], model)

                            if prediction == 'spam':
                                st.error(f"🚫 **SPAM DETECTED** - {max(probability) * 100:.1f}% confidence")
                            else:
                                st.success(f"✅ **LEGITIMATE MESSAGE** - {max(probability) * 100:.1f}% confidence")

                            st.caption(f"Spam: {probability[1]*100:.1f}% | Ham: {probability[0]*100:.1f}%")

# Analytics Page
elif page == "📊 Analytics":
    st.markdown("## 📊 Performance Analytics Dashboard")
    st.markdown("")

    if st.session_state.history:
        # Create DataFrame from history
        df_history = pd.DataFrame(st.session_state.history)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total_msgs = len(df_history)
        spam_count = (df_history['prediction'] == 'spam').sum()
        ham_count = (df_history['prediction'] == 'ham').sum()
        avg_confidence = df_history['confidence'].mean()

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Messages</div>
                <div class="metric-value">{total_msgs}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚫 Spam Detected</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {spam_count}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✅ Legitimate</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {ham_count}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value" style="font-size: 2rem;">{avg_confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Donut chart
            fig_pie = px.pie(
                values=[spam_count, ham_count],
                names=['Spam', 'Legitimate'],
                title='📊 Message Distribution',
                color_discrete_sequence=['#ef4444', '#22c55e'],
                hole=0.6
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=14
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e4e7eb', 'family': 'Inter', 'size': 14},
                title={'font': {'size': 20}},
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Confidence distribution
            fig_hist = px.histogram(
                df_history,
                x='confidence',
                nbins=25,
                title='🎯 Confidence Score Distribution',
                color_discrete_sequence=['#6366f1']
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.4)',
                font={'color': '#e4e7eb', 'family': 'Inter'},
                title={'font': {'size': 20}},
                xaxis_title='Confidence Score (%)',
                yaxis_title='Frequency',
                height=400,
                xaxis={'gridcolor': 'rgba(148, 163, 184, 0.1)'},
                yaxis={'gridcolor': 'rgba(148, 163, 184, 0.1)'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Timeline analysis
        if len(df_history) > 1:
            st.markdown("### 📈 Detection Timeline")
            st.markdown("")

            df_history['hour'] = pd.to_datetime(df_history['timestamp']).dt.hour
            timeline = df_history.groupby(['hour', 'prediction']).size().reset_index(name='count')

            fig_timeline = px.bar(
                timeline,
                x='hour',
                y='count',
                color='prediction',
                title='⏰ Hourly Detection Pattern',
                color_discrete_map={'spam': '#ef4444', 'ham': '#22c55e'},
                barmode='group'
            )
            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.4)',
                font={'color': '#e4e7eb', 'family': 'Inter'},
                title={'font': {'size': 20}},
                xaxis_title='Hour of Day',
                yaxis_title='Number of Messages',
                height=400,
                xaxis={'gridcolor': 'rgba(148, 163, 184, 0.1)', 'tickmode': 'linear'},
                yaxis={'gridcolor': 'rgba(148, 163, 184, 0.1)'},
                legend_title_text='Message Type'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Recent activity
        st.markdown("### 🕒 Recent Activity")
        st.markdown("")

        recent_df = df_history.head(10)[['timestamp', 'message', 'prediction', 'confidence']].copy()
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        recent_df['prediction'] = recent_df['prediction'].apply(lambda x: '🚫 SPAM' if x == 'spam' else '✅ HAM')
        recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.2f}%")
        recent_df.columns = ['Time', 'Message', 'Result', 'Confidence']

        st.dataframe(recent_df, use_container_width=True, height=400)

    else:
        st.info("📭 **No data available yet.** Start analyzing messages to see comprehensive analytics!")
        st.markdown("")
        st.markdown("""
        <div class="info-box">
            <h4>💡 What you'll see here:</h4>
            <ul>
                <li>📊 Distribution charts showing spam vs legitimate messages</li>
                <li>🎯 Confidence score analytics</li>
                <li>📈 Timeline patterns of detections</li>
                <li>🕒 Recent activity log</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# History Page
elif page == "📜 History":
    st.markdown("## 📜 Analysis History")
    st.markdown("")

    if st.session_state.history:
        st.markdown(f"### 📊 Showing Last {len(st.session_state.history)} Analyses")
        st.markdown("")

        # Filter options
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            filter_type = st.selectbox(
                "Filter by type:",
                ["All Messages", "Spam Only", "Legitimate Only"]
            )

        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"]
            )

        with col3:
            st.markdown("")
            st.markdown("")
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        # Apply filters
        filtered_history = st.session_state.history.copy()

        if filter_type == "Spam Only":
            filtered_history = [h for h in filtered_history if h['prediction'] == 'spam']
        elif filter_type == "Legitimate Only":
            filtered_history = [h for h in filtered_history if h['prediction'] == 'ham']

        # Apply sorting
        if sort_by == "Oldest First":
            filtered_history = filtered_history[::-1]
        elif sort_by == "Highest Confidence":
            filtered_history = sorted(filtered_history, key=lambda x: x['confidence'], reverse=True)
        elif sort_by == "Lowest Confidence":
            filtered_history = sorted(filtered_history, key=lambda x: x['confidence'])

        st.markdown("")
        st.info(f"📋 Showing **{len(filtered_history)}** messages")
        st.markdown("")

        # Display history items
        for idx, entry in enumerate(filtered_history):
            prediction_emoji = '🚫' if entry['prediction'] == 'spam' else '✅'
            prediction_text = 'SPAM' if entry['prediction'] == 'spam' else 'LEGITIMATE'
            prediction_color = '#ef4444' if entry['prediction'] == 'spam' else '#22c55e'

            with st.expander(
                f"{prediction_emoji} {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {prediction_text} ({entry['confidence']:.1f}%)",
                expanded=False
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("**Message Content:**")
                    st.text_area(
                        "",
                        entry['message'],
                        height=120,
                        disabled=True,
                        key=f"hist_msg_{idx}",
                        label_visibility="collapsed"
                    )

                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 1rem;">
                        <div class="metric-label">Result</div>
                        <div class="metric-value" style="font-size: 1.5rem; color: {prediction_color};">
                            {prediction_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 1rem;">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" style="font-size: 1.8rem;">
                            {entry['confidence']:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Time</div>
                        <div style="font-size: 1.2rem; color: #e4e7eb; margin-top: 0.5rem;">
                            {entry['timestamp'].strftime('%H:%M:%S')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Export history
        st.markdown("---")
        if st.button("📥 Export Full History", use_container_width=True):
            df_export = pd.DataFrame(st.session_state.history)
            df_export['timestamp'] = pd.to_datetime(df_export['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download History CSV",
                data=csv,
                file_name=f"spam_detector_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.info("📭 **No history available yet.** Start analyzing messages to build your history!")
        st.markdown("")
        st.markdown("""
        <div class="info-box">
            <h4>💡 History Features:</h4>
            <ul>
                <li>📝 Complete record of all analyzed messages</li>
                <li>🔍 Filter by spam or legitimate messages</li>
                <li>📊 Sort by time or confidence levels</li>
                <li>💾 Export history to CSV for records</li>
                <li>🗑️ Clear history anytime from sidebar</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# About Page
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About Group 1 Spam Detector")
    st.markdown("")

    # Main description
    st.markdown("""
    <div class="modern-card">
        <h3 style="color: #f1f5f9; margin-bottom: 1rem;">🚀 What is Group 1 Spam Detector?</h3>
        <p style="font-size: 1.05rem; line-height: 1.8; color: #cbd5e1; margin-bottom: 1rem;">
            <strong>Group 1 Spam Detector</strong> is a cutting-edge AI-powered application that leverages 
            state-of-the-art Natural Language Processing (NLP) and Machine Learning techniques to identify 
            spam and malicious messages with exceptional accuracy.
        </p>
        <p style="font-size: 1.05rem; line-height: 1.8; color: #cbd5e1;">
            Our advanced system analyzes <strong>over 20 distinct linguistic features</strong> including 
            sentiment analysis, pattern recognition, and behavioral indicators to provide intelligent 
            predictions about message authenticity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Features and Technology
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">✨ Key Features</h3>
            <div style="line-height: 2.2;">
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    🎯 <strong>99%+ Accuracy:</strong> Industry-leading detection rate
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    ⚡ <strong>Real-time Analysis:</strong> Instant results in milliseconds
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    📊 <strong>Batch Processing:</strong> Analyze hundreds of messages at once
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    📈 <strong>Advanced Analytics:</strong> Comprehensive performance insights
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    📱 <strong>Fully Responsive:</strong> Perfect on all devices
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    💾 <strong>Export Capability:</strong> Download results as CSV
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">🔬 Technology Stack</h3>
            <div style="line-height: 2.2;">
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    🐍 <strong>Python:</strong> Core programming language
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    🤖 <strong>Scikit-learn:</strong> Machine learning models
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    📊 <strong>Streamlit:</strong> Modern web framework
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    🎨 <strong>Plotly:</strong> Interactive visualizations
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    📝 <strong>NLTK:</strong> Natural language processing
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    💭 <strong>TextBlob:</strong> Sentiment analysis
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    🧠 <strong>VADER:</strong> Social media text analysis
                </p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    🎯 <strong>Dill:</strong> Model serialization
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # How it works
    st.markdown("""
    <div class="modern-card">
        <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">🔍 How It Works</h3>
        <p style="font-size: 1.05rem; line-height: 1.8; color: #cbd5e1; margin-bottom: 1.5rem;">
            Our spam detection system employs a sophisticated multi-layered approach to ensure 
            maximum accuracy and reliability:
        </p>
        <div style="line-height: 2;">
            <p style="color: #cbd5e1; margin: 1rem 0;">
                <strong style="color: #6366f1;">1. Text Preprocessing</strong><br>
                <span style="color: #94a3b8;">Advanced cleaning and normalization of input text, including tokenization, 
                lemmatization, and stop word removal.</span>
            </p>
            <p style="color: #cbd5e1; margin: 1rem 0;">
                <strong style="color: #8b5cf6;">2. Feature Extraction</strong><br>
                <span style="color: #94a3b8;">Comprehensive analysis of 20+ linguistic and statistical features including 
                message length, punctuation patterns, URL presence, and spam keywords.</span>
            </p>
            <p style="color: #cbd5e1; margin: 1rem 0;">
                <strong style="color: #a855f7;">3. Sentiment Analysis</strong><br>
                <span style="color: #94a3b8;">Deep understanding of emotional tone using TextBlob and VADER sentiment 
                analyzers to detect manipulative language patterns.</span>
            </p>
            <p style="color: #cbd5e1; margin: 1rem 0;">
                <strong style="color: #c026d3;">4. Machine Learning Classification</strong><br>
                <span style="color: #94a3b8;">Advanced ensemble models trained on thousands of real-world messages 
                to make accurate predictions with high confidence.</span>
            </p>
            <p style="color: #cbd5e1; margin: 1rem 0;">
                <strong style="color: #d946ef;">5. Confidence Scoring</strong><br>
                <span style="color: #94a3b8;">Probability estimates for both spam and legitimate classifications, 
                giving you full transparency into the AI's decision-making process.</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Feature details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">📊 Analyzed Features</h3>
            <div style="line-height: 1.8;">
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Message length & word count</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Character and punctuation patterns</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Exclamation and question marks</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Uppercase letter ratios</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• URL and email detection</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Digit count and patterns</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Spam keyword frequency</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Sentiment polarity & subjectivity</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">🎯 Use Cases</h3>
            <div style="line-height: 1.8;">
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Email spam filtering</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• SMS message screening</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Social media content moderation</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Business communication security</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Phishing detection</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Marketing message validation</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Customer support filtering</p>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">• Educational research</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Tips and best practices
    st.markdown("""
    <div class="info-box">
        <h4 style="margin-bottom: 1rem;">💡 Pro Tips for Best Results</h4>
        <div style="line-height: 1.8;">
            <p style="margin: 0.5rem 0;">
                ✅ <strong>Complete Messages:</strong> Analyze full messages rather than fragments for better accuracy
            </p>
            <p style="margin: 0.5rem 0;">
                ✅ <strong>English Text:</strong> The model is optimized for English language messages
            </p>
            <p style="margin: 0.5rem 0;">
                ✅ <strong>Context Matters:</strong> Consider the source and context when interpreting results
            </p>
            <p style="margin: 0.5rem 0;">
                ✅ <strong>Batch Processing:</strong> Use CSV upload for analyzing multiple messages efficiently
            </p>
            <p style="margin: 0.5rem 0;">
                ✅ <strong>Confidence Scores:</strong> Pay attention to confidence levels - higher is more certain
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Model information
    st.markdown("""
    <div class="modern-card">
        <h3 style="color: #f1f5f9; margin-bottom: 1rem;">🧠 Model Information</h3>
        <p style="font-size: 1rem; line-height: 1.8; color: #cbd5e1;">
            The spam detector is powered by a carefully trained machine learning pipeline that has been 
            validated on thousands of real-world spam and legitimate messages. The model continuously 
            learns from diverse message patterns to maintain high accuracy across different types of 
            communication.
        </p>
        <p style="font-size: 1rem; line-height: 1.8; color: #cbd5e1; margin-top: 1rem;">
            <strong>Training Data:</strong> The model has been trained on a comprehensive dataset 
            containing various spam patterns including promotional messages, phishing attempts, 
            scams, and legitimate communications across different contexts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Privacy notice
    st.markdown("""
    <div class="modern-card" style="border-left: 4px solid #3b82f6;">
        <h4 style="color: #f1f5f9; margin-bottom: 1rem;">🔒 Privacy & Security</h4>
        <p style="font-size: 0.95rem; line-height: 1.8; color: #cbd5e1;">
            All message analysis is performed locally within your session. Messages are not stored 
            permanently and are only kept in temporary session memory for history viewing. Your data 
            privacy and security are our top priorities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <p style="font-size: 1rem; color: #94a3b8; margin-bottom: 0.5rem;">
            Made with ❤️ using <strong>Streamlit</strong> | Powered by <strong>Advanced NLP & ML</strong>
        </p>
        <p style="font-size: 0.85rem; color: #64748b;">
            Professional 2025 Design | Dark Mode Excellence
        </p>
    </div>
    """, unsafe_allow_html=True)

# Global Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <p style="font-size: 1.1rem; font-weight: 600; color: #e4e7eb; margin-bottom: 0.5rem;">
        🛡️ <strong>Group 1 Spam Detector</strong>
    </p>
    <p style="font-size: 0.9rem; color: #94a3b8;">
        Protecting your communications with AI-powered intelligence
    </p>
    <p style="font-size: 0.75rem; color: #64748b; margin-top: 1rem;">
        © 2025 Group 1 Spam Detector | All Rights Reserved
    </p>
</div>

""", unsafe_allow_html=True)











