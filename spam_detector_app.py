"""
GROUP 1 SPAM DETECTOR
Modern, Professional UI/UX Design - 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import dill
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
    page_title="Group 1 Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Professional CSS - Based on Linear, Notion, Vercel Design Systems
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables - Design System */
    :root {
        --bg-primary: #0A0B0D;
        --bg-secondary: #111213;
        --bg-tertiary: #1A1B1E;
        --bg-elevated: #1E1F23;
        
        --text-primary: #ECECED;
        --text-secondary: #9B9C9E;
        --text-tertiary: #6B6C6F;
        
        --border-color: #26272B;
        --border-hover: #3A3B40;
        
        --accent-primary: #3B82F6;
        --accent-hover: #2563EB;
        
        --success: #10B981;
        --success-bg: rgba(16, 185, 129, 0.1);
        
        --error: #EF4444;
        --error-bg: rgba(239, 68, 68, 0.1);
        
        --warning: #F59E0B;
        --warning-bg: rgba(245, 158, 11, 0.1);
        
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
        
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    * {
        font-family: var(--font-family);
    }
    
    /* Base Styles */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .block-container {
        padding-top: var(--spacing-xl);
        padding-bottom: var(--spacing-xl);
        max-width: 1200px;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: var(--spacing-xl) 0;
        margin-bottom: var(--spacing-xl);
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-sm);
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        transition: border-color 0.2s ease;
    }
    
    .card:hover {
        border-color: var(--border-hover);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 10px 24px;
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        height: 40px;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-tertiary);
        border-color: var(--border-hover);
    }
    
    /* Text Input & Text Area */
    .stTextArea textarea,
    .stTextInput input {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.875rem !important;
        padding: 12px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: var(--text-tertiary) !important;
    }
    
    .stTextArea > label,
    .stTextInput > label {
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        margin-bottom: var(--spacing-sm) !important;
    }
    
    /* Result Cards */
    .result-card {
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        margin: var(--spacing-md) 0;
        border: 1px solid;
    }
    
    .result-spam {
        background: var(--error-bg);
        border-color: var(--error);
    }
    
    .result-ham {
        background: var(--success-bg);
        border-color: var(--success);
    }
    
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: var(--spacing-sm);
    }
    
    .result-spam .result-title {
        color: var(--error);
    }
    
    .result-ham .result-title {
        color: var(--success);
    }
    
    .result-description {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-bottom: var(--spacing-md);
    }
    
    .result-confidence {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: var(--spacing-xs);
    }
    
    .result-label {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: var(--border-hover);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: var(--spacing-xs);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: var(--spacing-md);
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div > label {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 10px 16px;
        transition: all 0.2s ease;
        cursor: pointer;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .stRadio > div > label:hover {
        border-color: var(--border-hover);
        background: var(--bg-tertiary);
    }
    
    .stRadio > div > label[data-baseweb="radio"] {
        background: var(--bg-tertiary);
        border-color: var(--accent-primary);
        color: var(--text-primary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--spacing-sm);
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border: none;
        border-radius: 0;
        border-bottom: 2px solid transparent;
        padding: 12px 16px;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom-color: var(--accent-primary) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: var(--accent-primary);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
        font-weight: 500;
        padding: 12px 16px !important;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--border-hover);
        background: var(--bg-tertiary);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--spacing-xl);
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-primary);
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        color: var(--text-primary);
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    /* Info/Warning/Error/Success */
    .stAlert {
        border-radius: var(--radius-md);
        border: 1px solid;
        padding: var(--spacing-md);
        font-size: 0.875rem;
    }
    
    .stSuccess {
        background: var(--success-bg) !important;
        border-color: var(--success) !important;
        color: var(--success) !important;
    }
    
    .stError {
        background: var(--error-bg) !important;
        border-color: var(--error) !important;
        color: var(--error) !important;
    }
    
    .stWarning {
        background: var(--warning-bg) !important;
        border-color: var(--warning) !important;
        color: var(--warning) !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    h1 { font-size: 2rem; letter-spacing: -0.02em; }
    h2 { font-size: 1.5rem; letter-spacing: -0.01em; }
    h3 { font-size: 1.25rem; }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-elevated);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-hover);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    
    .stDownloadButton > button:hover {
        background: var(--bg-tertiary);
        border-color: var(--border-hover);
    }
    
    /* Divider */
    hr {
        border-color: var(--border-color);
        margin: var(--spacing-xl) 0;
    }
    
    /* Info Box */
    .info-box {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--accent-primary);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        margin: var(--spacing-md) 0;
        font-size: 0.875rem;
        line-height: 1.6;
    }
    
    .info-box h4 {
        margin-top: 0;
        margin-bottom: var(--spacing-sm);
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Logistic Regression"
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
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, sia

STOP_WORDS, LEMMATIZER, SIA = init_nlp_tools()

def simple_tokenizer(text):
    """Split text into tokens (already preprocessed)"""
    return text.split()

# Load model
@st.cache_resource
def load_all_models():
    """Load all available models"""
    models = {}
    # Try to load Tuned Logistic Regression
    try:
        models['Logistic Regression'] = joblib.load('logistic_regression_spam_detector_model.pkl')
    except:
        pass

    # Try to load Tuned Naive Bayes
    try:
        models['Naive Bayes'] = joblib.load('naive_bayes_spam_detector.pkl')
    except:
        pass

    return models

# Text preprocessing
def advanced_text_preprocessing(text):
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

# Prediction
def predict_message(text, model):
    if model is None:
        return None, None, None
    features = extract_features(text)
    df_pred = pd.DataFrame([features])
    prediction = model.predict(df_pred)[0]
    probability = model.predict_proba(df_pred)[0]
    return prediction, probability, features

# Create gauge chart
def create_gauge(value, color_scheme):
    color = "#EF4444" if color_scheme == "danger" else "#10B981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#3A3B40"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#1A1B1E",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': "#111213"},
                {'range': [50, 100], 'color': "#1E1F23"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ECECED"},
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# Create feature chart
def create_feature_chart(features):
    key_features = {
        'Length': features['message_length'],
        'Words': features['word_count'],
        'Uppercase': features['uppercase_ratio'] * 100,
        'URLs': features['has_url'] * 100,
        'Spam Words': features['spam_word_count'] * 10,
        'Exclamations': features['exclamation_count'] * 10
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(key_features.keys()),
            y=list(key_features.values()),
            marker=dict(color='#3B82F6'),
            text=[f"{v:.0f}" for v in key_features.values()],
            textposition='outside',
        )
    ])

    fig.update_layout(
        title="Feature Analysis",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1A1B1E',
        font={'color': "#ECECED"},
        xaxis={'showgrid': False},
        yaxis={'showgrid': True, 'gridcolor': '#26272B'},
        height=350,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">🛡️ Group 1 Spam Detector</div>
    <div class="app-subtitle">AI-powered message analysis with machine learning</div>
</div>
""", unsafe_allow_html=True)

# Load all models
all_models = load_all_models()

if not all_models:
    st.error("⚠️ No model files found. Please ensure at least one model file is in the directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["Detector", "Analytics", "History", "About"],
                    label_visibility="collapsed")

    st.divider()

    # MODEL SELECTOR - ADD THIS NEW SECTION
    st.markdown("### Model Selection")

    available_models = list(all_models.keys())

    # Display model selector
    selected_model_name = st.selectbox(
        "Choose Model",
        available_models,
        index=available_models.index(
            st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        key="model_selector"
    )

    # Update session state
    if selected_model_name != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_name
        st.rerun()

    # Get the active model
    model = all_models[st.session_state.selected_model]

    # Show model info
    st.caption(f"Active: {st.session_state.selected_model}")

    st.divider()
    # END OF MODEL SELECTOR

    st.markdown("### Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scanned", st.session_state.total_scanned)
    with col2:
        st.metric("Spam", st.session_state.spam_detected)

    if st.session_state.total_scanned > 0:
        spam_rate = (st.session_state.spam_detected / st.session_state.total_scanned) * 100
        st.metric("Spam Rate", f"{spam_rate:.1f}%")

    st.divider()
    if st.button("Clear Data", use_container_width=True, type="secondary"):
        st.session_state.history = []
        st.session_state.total_scanned = 0
        st.session_state.spam_detected = 0
        st.session_state.current_message = ""
        st.rerun()

# Main Content
if page == "🏠 Detector":
    tab1, tab2, tab3 = st.tabs(["Single Message", "Batch Analysis", "Examples"])

    with tab1:
        st.markdown("### Analyze Message")

        # Example buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🎁 Spam Example", use_container_width=True, type="secondary", key="spam_ex"):
                st.session_state.current_message = "Congratulations! You've won $1000! Click here now!"
                st.rerun()
        with col2:
            if st.button("✅ Legit Example", use_container_width=True, type="secondary", key="legit_ex"):
                st.session_state.current_message = "Hey, are you free for lunch tomorrow?"
                st.rerun()
        with col3:
            if st.button("⚠️ Phishing Example", use_container_width=True, type="secondary", key="phish_ex"):
                st.session_state.current_message = "Your account has been suspended. Click here immediately to verify your information: bit.ly/secure-login-2847"
                st.rerun()
        with col4:
            if st.button("💼 Business Example", use_container_width=True, type="secondary", key="biz_ex"):
                st.session_state.current_message = "Meeting at 3 PM. Please bring the reports."
                st.rerun()

        # Text area with session state value
        message_input = st.text_area(
            "Enter message to analyze",
            value=st.session_state.current_message,
            height=150,
            placeholder="Type or paste your message here...",
            key="message_textarea"
        )

        st.markdown("")
        if st.button("🚀 Analyze Message", use_container_width=True, type="primary"):
            if message_input:
                # Update session state with current message
                st.session_state.current_message = message_input

                with st.spinner("Analyzing..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.005)
                        progress.progress(i + 1)

                    prediction, probability, features = predict_message(message_input, model)

                    st.session_state.total_scanned += 1
                    if prediction == 'spam':
                        st.session_state.spam_detected += 1

                    st.session_state.history.insert(0, {
                        'timestamp': datetime.now(),
                        'message': message_input[:100] + "..." if len(message_input) > 100 else message_input,
                        'prediction': prediction,
                        'confidence': max(probability) * 100
                    })
                    st.session_state.history = st.session_state.history[:100]

                st.divider()
                st.markdown("### Results")

                col1, col2 = st.columns([1, 1])

                with col1:
                    if prediction == 'spam':
                        st.markdown(f"""
                        <div class="result-card result-spam">
                            <div class="result-title">🚫 Spam Detected</div>
                            <div class="result-description">This message appears to be spam or malicious</div>
                            <div class="result-confidence">{max(probability)*100:.1f}%</div>
                            <div class="result-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card result-ham">
                            <div class="result-title">✅ Legitimate Message</div>
                            <div class="result-description">This message appears to be safe</div>
                            <div class="result-confidence">{max(probability)*100:.1f}%</div>
                            <div class="result-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.plotly_chart(
                        create_gauge(
                            probability[1] if prediction == 'spam' else probability[0],
                            "danger" if prediction == 'spam' else "success"
                        ),
                        use_container_width=True
                    )

                st.markdown("### Probabilities")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{probability[0]*100:.2f}%</div>
                        <div class="metric-label">Legitimate</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{probability[1]*100:.2f}%</div>
                        <div class="metric-label">Spam</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                st.plotly_chart(create_feature_chart(features), use_container_width=True)

                st.markdown("### Message Details")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['message_length']}</div>
                        <div class="metric-label">Characters</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['word_count']}</div>
                        <div class="metric-label">Words</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['exclamation_count']}</div>
                        <div class="metric-label">Exclamations</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['spam_word_count']}</div>
                        <div class="metric-label">Spam Words</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a message to analyze")

    with tab2:
        st.markdown("### Batch Analysis")
        st.info("Upload a CSV with a 'message' column or paste messages (one per line)")

        method = st.radio("Input method", ["Upload CSV", "Paste Messages"], horizontal=True)

        messages = []
        if method == "Upload CSV":
            file = st.file_uploader("Upload CSV file", type=['csv'])
            if file:
                try:
                    df = pd.read_csv(file)
                    if 'message' in df.columns:
                        messages = df['message'].dropna().tolist()
                        st.success(f"Loaded {len(messages)} messages")
                    else:
                        st.error("CSV must have a 'message' column")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            batch_input = st.text_area("Paste messages (one per line)", height=200)
            if batch_input:
                messages = [m.strip() for m in batch_input.split('\n') if m.strip()]
                st.info(f"Ready to analyze {len(messages)} messages")

        if messages and st.button("Analyze All", use_container_width=True, type="primary"):
            results = []
            progress = st.progress(0)
            status = st.empty()

            for idx, msg in enumerate(messages):
                status.text(f"Analyzing {idx+1}/{len(messages)}")
                prediction, probability, _ = predict_message(msg, model)
                results.append({
                    'Message': msg[:80] + "..." if len(msg) > 80 else msg,
                    'Prediction': 'SPAM' if prediction == 'spam' else 'HAM',
                    'Confidence': f"{max(probability)*100:.1f}%",
                    'Spam Prob': f"{probability[1]*100:.1f}%"
                })
                progress.progress((idx + 1) / len(messages))

            status.empty()
            progress.empty()

            results_df = pd.DataFrame(results)
            spam_count = sum(1 for r in results if r['Prediction'] == 'SPAM')

            st.divider()
            st.markdown("### Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", len(results))
            with col2:
                st.metric("Spam", spam_count)
            with col3:
                st.metric("Legitimate", len(results) - spam_count)

            st.dataframe(results_df, use_container_width=True, height=400)

            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

    with tab3:
        st.markdown("### Example Messages")

        examples = {
            "Prize Winner": "Congratulations! You've won $1000! Call now!",
            "Lunch Invite": "Hey, are you free for lunch tomorrow?",
            "Account Alert": "URGENT! Your account will be closed. Click now!",
            "Meeting": "Meeting at 3 PM. Bring the reports."
        }

        col1, col2 = st.columns(2)
        for idx, (label, msg) in enumerate(examples.items()):
            col = col1 if idx % 2 == 0 else col2
            with col:
                with st.expander(label):
                    st.text_area("", msg, height=80, disabled=True, key=f"ex_{idx}", label_visibility="collapsed")
                    if st.button("Analyze", key=f"btn_ex_{idx}", use_container_width=True):
                        prediction, probability, _ = predict_message(msg, model)
                        if prediction == 'spam':
                            st.error(f"SPAM ({max(probability)*100:.1f}%)")
                        else:
                            st.success(f"HAM ({max(probability)*100:.1f}%)")

elif page == "📊 Analytics":
    st.markdown("### Analytics Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        col1, col2, col3, col4 = st.columns(4)
        spam_count = (df['prediction'] == 'spam').sum()
        ham_count = (df['prediction'] == 'ham').sum()

        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("Spam", spam_count)
        with col3:
            st.metric("Legitimate", ham_count)
        with col4:
            st.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=[spam_count, ham_count], names=['Spam', 'Legitimate'],
                        title="Distribution", color_discrete_sequence=['#EF4444', '#10B981'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ECECED"}, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x='confidence', nbins=20, title="Confidence Distribution",
                             color_discrete_sequence=['#3B82F6'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1B1E',
                            font={'color': "#ECECED"}, height=350,
                            xaxis={'gridcolor': '#26272B'}, yaxis={'gridcolor': '#26272B'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available. Start analyzing messages to see analytics.")

elif page == "📜 History":
    st.markdown("### Analysis History")

    if st.session_state.history:
        filter_type = st.selectbox("Filter", ["All", "Spam Only", "Legitimate Only"])

        history = st.session_state.history.copy()
        if filter_type == "Spam Only":
            history = [h for h in history if h['prediction'] == 'spam']
        elif filter_type == "Legitimate Only":
            history = [h for h in history if h['prediction'] == 'ham']

        st.info(f"Showing {len(history)} messages")

        for idx, entry in enumerate(history[:20]):
            emoji = '🚫' if entry['prediction'] == 'spam' else '✅'
            pred = entry['prediction'].upper()
            with st.expander(f"{emoji} {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {pred} ({entry['confidence']:.1f}%)"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text_area("Message", entry['message'], height=100, disabled=True,
                               key=f"hist_{idx}", label_visibility="collapsed")
                with col2:
                    st.metric("Confidence", f"{entry['confidence']:.1f}%")
    else:
        st.info("No history available. Start analyzing messages!")

else:  # About
    st.markdown("### About")

    st.markdown("""
    <div class="info-box">
        <h4>Group 1 Spam Detector</h4>
        <p>
            An AI-powered spam detection system using advanced Natural Language Processing 
            and Machine Learning techniques. The system analyzes over 20 linguistic features 
            to accurately classify messages as spam or legitimate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Key Features</h4>
            <p>• 99%+ accuracy rate<br>
            • Real-time analysis<br>
            • Batch processing<br>
            • Comprehensive analytics<br>
            • Export capabilities</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Technology Stack</h4>
            <p>• Python & Scikit-learn<br>
            • NLTK & TextBlob<br>
            • Streamlit<br>
            • Plotly<br>
            • Advanced NLP</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: var(--text-tertiary); font-size: 0.875rem; padding: var(--spacing-lg) 0;">
    Group 1 Spam Detector © 2025 | Built with Streamlit
</div>
""", unsafe_allow_html=True)
