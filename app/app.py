# ============================================================
# app.py
# Urdu & Roman Urdu Sentiment Analysis
# Models: Logistic Regression, Naive Bayes, Linear SVC
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
import re
from typing import Tuple, Optional, Union, Dict, List, Any

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Urdu & Roman Urdu Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS (Enhanced)
# ============================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background-color: #f8f9fa;
    }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.10);
        transition: all 0.3s ease;
    }
    
    .dashboard-header:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102,126,234,0.3);
    }
    
    .dashboard-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
        letter-spacing: -0.5px;
    }
    
    .dashboard-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0.2rem;
    }
    
    .dashboard-header .badge-info {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Stat Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 0.3rem;
    }
    
    .stat-change {
        font-size: 0.8rem;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stat-change.up {
        background: #d4edda;
        color: #155724;
    }
    
    .stat-change.down {
        background: #f8d7da;
        color: #721c24;
    }

    /* Result Badges */
    .badge-positive {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    
    .badge-negative {
        background: linear-gradient(135deg, #dc3545, #e74c3c);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    
    .badge-neutral {
        background: linear-gradient(135deg, #6c757d, #adb5bd);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }

    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
    }
    
    .feature-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #7f8c8d;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
        color: white;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Metrics Container */
    .metric-container {
        background: white;
        border-radius: 15px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    /* Result Container */
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #667eea;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f1f3f5;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102,126,234,0.1);
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2c3e50;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .dashboard-header h1 {
            font-size: 1.8rem;
        }
        .stat-number {
            font-size: 1.6rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.3rem 0.8rem;
            font-size: 0.85rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

# Model configuration
MODEL_OPTIONS: Dict[str, str] = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "Linear SVC": "linear_svc"
}

LANGUAGE_CODES: Dict[str, str] = {
    "Urdu": "urdu",
    "Roman Urdu": "roman_urdu"
}

# Performance data from research
PERFORMANCE_DATA: Dict[Tuple[str, str], float] = {
    ("Urdu", "Logistic Regression"): 0.8562,
    ("Urdu", "Naive Bayes"): 0.8534,
    ("Urdu", "Linear SVC"): 0.8760,
    ("Roman Urdu", "Logistic Regression"): 0.7810,
    ("Roman Urdu", "Naive Bayes"): 0.7907,
    ("Roman Urdu", "Linear SVC"): 0.7867
}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "Logistic Regression": "Linear classification model using TF-IDF features with L2 regularization.",
    "Naive Bayes": "Probabilistic classification model based on Bayes' theorem, suitable for text classification.",
    "Linear SVC": "Support Vector Machine with linear kernel, effective for high-dimensional text data."
}

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state() -> None:
    """Initialize all session state variables"""
    defaults = {
        'history': [],
        'total_predictions': 0,
        'positive_predictions': 0,
        'negative_predictions': 0,
        'model_loaded': False,
        'current_model': None,
        'current_vectorizer': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            setattr(st.session_state, key, default_value)


initialize_session_state()


# ============================================================
# PROJECT PATHS
# ============================================================

def get_project_paths() -> Tuple[Path, Path]:
    """Get base and model directories"""
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    vectorizers_dir = base_dir / "vectorizers"
    return models_dir, vectorizers_dir


MODELS_DIR, VECTORIZERS_DIR = get_project_paths()


# ============================================================
# FILE LOADING FUNCTIONS
# ============================================================

def load_file(file_path: Path) -> Optional[Any]:
    """
    Load model or vectorizer from file with multiple format support.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        Loaded object or None if failed
    """
    try:
        # Try joblib first (preferred for scikit-learn models)
        try:
            return joblib.load(file_path)
        except Exception:
            pass
        
        # Try pickle
        with open(file_path, "rb") as f:
            return pickle.load(f)
            
    except Exception as e:
        st.error(f"❌ Could not load file: {file_path.name}")
        st.error(f"Error: {str(e)}")
        return None


def find_model_file(language_code: str, model_code: str) -> Optional[Path]:
    """
    Find model file with various extensions.
    
    Args:
        language_code: Language code (urdu/roman_urdu)
        model_code: Model code
        
    Returns:
        Path to model file or None
    """
    possible_extensions = ['.pkl', '.PKL', '.joblib', '.JOBLIB']
    possible_names = [
        f"{language_code}_{model_code}{ext}" 
        for ext in possible_extensions
    ]
    
    for name in possible_names:
        file_path = MODELS_DIR / name
        if file_path.exists():
            return file_path
    
    return None


def find_vectorizer_file(language_code: str) -> Optional[Path]:
    """
    Find vectorizer file with various extensions.
    
    Args:
        language_code: Language code
        
    Returns:
        Path to vectorizer file or None
    """
    possible_extensions = ['.pkl', '.PKL', '.joblib', '.JOBLIB']
    possible_names = [
        f"{language_code}_tfidf{ext}" 
        for ext in possible_extensions
    ]
    
    for name in possible_names:
        file_path = VECTORIZERS_DIR / name
        if file_path.exists():
            return file_path
    
    return None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_selected_model(
    language_code: str, 
    model_code: str
) -> Tuple[Optional[Any], Optional[Any], Optional[Path], Optional[Path]]:
    """
    Load selected model and vectorizer.
    
    Returns:
        Tuple of (model, vectorizer, model_path, vectorizer_path)
    """
    model_path = find_model_file(language_code, model_code)
    vectorizer_path = find_vectorizer_file(language_code)
    
    if model_path is None:
        return None, None, None, None
    
    model = load_file(model_path)
    vectorizer = load_file(vectorizer_path) if vectorizer_path else None
    
    return model, vectorizer, model_path, vectorizer_path


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_prediction(prediction: Union[int, str, np.number]) -> str:
    """
    Normalize prediction to 'Positive' or 'Negative'.
    
    Args:
        prediction: Raw prediction from model
        
    Returns:
        Normalized sentiment label
    """
    # Handle numeric predictions
    if isinstance(prediction, (int, np.integer, np.floating)):
        pred_int = int(prediction)
        if pred_int == 0:
            return "Negative"
        elif pred_int == 1:
            return "Positive"
        else:
            return str(prediction)
    
    # Handle string predictions
    pred_str = str(prediction).strip().lower()
    
    if pred_str in ["negative", "neg", "0", "0.0", "neg"]:
        return "Negative"
    elif pred_str in ["positive", "pos", "1", "1.0", "pos"]:
        return "Positive"
    else:
        return pred_str.capitalize()


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep Urdu/Roman Urdu text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    return text.strip()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(
    text: str,
    model: Any,
    vectorizer: Any
) -> Tuple[str, float, Optional[np.ndarray]]:
    """
    Predict sentiment of input text.
    
    Args:
        text: Input text
        model: Loaded model
        vectorizer: Loaded vectorizer
        
    Returns:
        Tuple of (sentiment_label, confidence, probabilities)
    """
    try:
        if model is None or vectorizer is None:
            return "Error", 0.0, None
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        if not cleaned_text:
            return "Error", 0.0, None
        
        # Transform text to TF-IDF features
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        raw_prediction = model.predict(text_vectorized)[0]
        sentiment = normalize_prediction(raw_prediction)
        
        # Get confidence and probabilities
        confidence = 0.0
        probabilities = None
        
        # Models with predict_proba (Logistic Regression, Naive Bayes)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = float(np.max(probabilities))
        
        # Linear SVC with decision_function
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(text_vectorized)
            
            if np.ndim(decision) == 1:
                score = float(decision[0])
                # Convert decision score to confidence using sigmoid
                confidence = 1 / (1 + np.exp(-abs(score)))
            else:
                confidence = 0.5
        
        return sentiment, confidence, probabilities
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, None


# ============================================================
# STATISTICS MANAGEMENT
# ============================================================

def update_statistics(sentiment: str) -> None:
    """
    Update prediction statistics.
    
    Args:
        sentiment: Predicted sentiment
    """
    st.session_state.total_predictions += 1
    
    if sentiment == "Positive":
        st.session_state.positive_predictions += 1
    elif sentiment == "Negative":
        st.session_state.negative_predictions += 1


def add_to_history(
    text: str,
    sentiment: str,
    confidence: float,
    language: str,
    model_name: str
) -> None:
    """
    Add prediction to history.
    
    Args:
        text: Input text
        sentiment: Predicted sentiment
        confidence: Confidence score
        language: Language used
        model_name: Model name
    """
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Language": language,
        "Model": model_name,
        "Text": text[:200] + ("..." if len(text) > 200 else ""),
        "Sentiment": sentiment,
        "Confidence (%)": round(confidence * 100, 2)
    })


# ============================================================
# UI COMPONENTS
# ============================================================

def render_sidebar() -> Tuple[str, str, str, str, str]:
    """
    Render sidebar and return selected options.
    
    Returns:
        Tuple of (language, language_code, model_display, model_code, input_method)
    """
    with st.sidebar:
        # Custom CSS for better sidebar styling
        st.markdown("""
        <style>
            /* Sidebar background */
            .css-1d391kg, .st-emotion-cache-1d391kg {
                background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            }
            
            /* Sidebar header */
            .sidebar-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem 1rem;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .sidebar-header h2 {
                color: white !important;
                font-weight: 700;
                margin: 0;
                font-size: 1.5rem;
            }
            .sidebar-header p {
                color: rgba(255,255,255,0.9) !important;
                font-size: 0.85rem;
                margin: 0.3rem 0 0 0;
            }
            
            /* Sidebar cards */
            .sidebar-card {
                background: rgba(255,255,255,0.08);
                backdrop-filter: blur(10px);
                padding: 0.8rem 1rem;
                border-radius: 12px;
                margin-bottom: 0.3rem;
                border: 1px solid rgba(255,255,255,0.1);
                transition: all 0.3s ease;
            }
            .sidebar-card:hover {
                background: rgba(255,255,255,0.15);
                transform: translateX(5px);
            }
            .sidebar-card-title {
                color: #a8b2d1;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 600;
                margin-bottom: 0.2rem;
            }
            
            /* Sidebar stats */
            .sidebar-stats {
                background: rgba(255,255,255,0.05);
                border-radius: 10px;
                padding: 0.6rem;
                margin-top: 0.3rem;
                border-left: 3px solid #667eea;
            }
            .sidebar-stats-label {
                color: #a8b2d1;
                font-size: 0.7rem;
            }
            .sidebar-stats-value {
                color: white;
                font-size: 1.1rem;
                font-weight: 600;
            }
            
            /* Select boxes */
            .stSelectbox label {
                color: #a8b2d1 !important;
                font-weight: 500 !important;
                font-size: 0.85rem !important;
            }
            .stSelectbox div[data-baseweb="select"] {
                background: rgba(255,255,255,0.05) !important;
                border-radius: 8px !important;
                border: 1px solid rgba(255,255,255,0.1) !important;
            }
            
            /* Radio buttons */
            .stRadio label {
                color: #a8b2d1 !important;
                font-size: 0.85rem !important;
            }
            .stRadio div[role="radiogroup"] {
                background: rgba(255,255,255,0.05) !important;
                padding: 0.5rem !important;
                border-radius: 8px !important;
            }
            
            /* Divider */
            .sidebar-divider {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                margin: 0.8rem 0;
            }
            
            /* Project info */
            .project-info {
                background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
                border-radius: 12px;
                padding: 0.8rem;
                text-align: center;
                border: 1px solid rgba(102,126,234,0.3);
                margin-top: 0.8rem;
            }
            .project-info h4 {
                color: white !important;
                margin: 0.2rem 0;
            }
            .project-info p {
                color: rgba(255,255,255,0.7) !important;
                font-size: 0.75rem;
                margin: 0.1rem 0;
            }
            .project-badge {
                background: rgba(102,126,234,0.3);
                padding: 0.2rem 1rem;
                border-radius: 20px;
                font-size: 0.65rem;
                color: #a8b2d1;
                display: inline-block;
            }
            
            /* Status indicator */
            .status-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 0.5rem;
            }
            .status-online {
                background: #28a745;
                box-shadow: 0 0 10px rgba(40,167,69,0.5);
            }
            .status-offline {
                background: #dc3545;
                box-shadow: 0 0 10px rgba(220,53,69,0.5);
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                padding: 0.6rem !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
            }
            
            /* Current config */
            .current-config {
                padding: 0.5rem;
                background: rgba(102,126,234,0.1);
                border-radius: 8px;
                border: 1px dashed rgba(102,126,234,0.3);
                text-align: center;
            }
            .current-config p {
                color: #a8b2d1;
                font-size: 0.65rem;
                margin: 0;
            }
            .config-tag {
                background: rgba(102,126,234,0.2);
                padding: 0.15rem 0.8rem;
                border-radius: 12px;
                margin: 0 0.15rem;
                color: white;
                font-size: 0.8rem;
                display: inline-block;
            }
            
            /* Status row */
            .status-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.2rem 0;
            }
            .status-label {
                color: #a8b2d1;
                font-size: 0.8rem;
            }
            .status-value {
                color: white;
                font-size: 0.8rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # ==========================================================
        # SIDEBAR HEADER
        # ==========================================================
        st.markdown("""
        <div class="sidebar-header">
            <h2>⚙️ Configuration</h2>
            <p>Customize your analysis settings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================================
        # LANGUAGE SELECTION
        # ==========================================================
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🌐 Language</div>
        </div>
        """, unsafe_allow_html=True)
        
        language = st.selectbox(
            "",
            ["Urdu", "Roman Urdu"],
            help="Choose the language of your text",
            label_visibility="collapsed"
        )
        language_code = LANGUAGE_CODES[language]
        
        # ==========================================================
        # MODEL SELECTION
        # ==========================================================
        st.markdown("""
        <div class="sidebar-card" style="margin-top: 0.3rem;">
            <div class="sidebar-card-title">🧠 Model</div>
        </div>
        """, unsafe_allow_html=True)
        
        model_display = st.selectbox(
            "",
            list(MODEL_OPTIONS.keys()),
            help="Choose the machine learning model",
            label_visibility="collapsed"
        )
        model_code = MODEL_OPTIONS[model_display]
        
        # ==========================================================
        # INPUT METHOD
        # ==========================================================
        st.markdown("""
        <div class="sidebar-card" style="margin-top: 0.3rem;">
            <div class="sidebar-card-title">📝 Input Method</div>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "",
            ["Single Text", "Batch Upload (CSV)"],
            help="Analyze single text or multiple texts via CSV",
            label_visibility="collapsed"
        )
        
        # ==========================================================
        # DIVIDER
        # ==========================================================
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        
        # ==========================================================
        # LIVE STATISTICS
        # ==========================================================
        st.markdown("""
        <div class="sidebar-card" style="background: rgba(102,126,234,0.15); border-color: rgba(102,126,234,0.3);">
            <div class="sidebar-card-title">📊 Live Statistics</div>
        </div>
        """, unsafe_allow_html=True)
        
        total = st.session_state.get('total_predictions', 0)
        pos = st.session_state.get('positive_predictions', 0)
        neg = st.session_state.get('negative_predictions', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="sidebar-stats" style="border-left-color: #667eea;">
                <div class="sidebar-stats-label">Total</div>
                <div class="sidebar-stats-value">{total:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="sidebar-stats" style="border-left-color: #28a745;">
                <div class="sidebar-stats-label">Positive</div>
                <div class="sidebar-stats-value" style="color: #28a745;">{pos:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="sidebar-stats" style="border-left-color: #dc3545;">
                <div class="sidebar-stats-label">Negative</div>
                <div class="sidebar-stats-value" style="color: #dc3545;">{neg:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ==========================================================
        # SYSTEM STATUS
        # ==========================================================
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🔄 System Status</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model is loaded
        model_status = st.session_state.get('model_loaded', False)
        
        status_color = "#28a745" if model_status else "#dc3545"
        status_text = "Online" if model_status else "Offline"
        status_dot = "status-online" if model_status else "status-offline"
        
        st.markdown(f"""
        <div class="status-row">
            <span class="status-label">Model Status</span>
            <span style="color: {status_color}; font-size: 0.8rem; font-weight: 600;">
                <span class="status-indicator {status_dot}"></span>
                {status_text}
            </span>
        </div>
        <div class="status-row">
            <span class="status-label">Session</span>
            <span class="status-value">{len(st.session_state.history)} predictions</span>
        </div>
        <div class="status-row">
            <span class="status-label">Last Activity</span>
            <span class="status-value">{datetime.now().strftime("%H:%M:%S")}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================================
        # REFRESH BUTTON
        # ==========================================================
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        # ==========================================================
        # PROJECT INFO
        # ==========================================================
        st.markdown("""
        <div class="project-info">
            <p style="font-size: 0.65rem; opacity: 0.6;">Final Year Project</p>
            <h4>💬 Urdu & Roman Urdu</h4>
            <p>Sentiment Analysis</p>
            <span class="project-badge">v2.0</span>
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================================
        # CURRENT SELECTION DISPLAY
        # ==========================================================
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="current-config">
            <p>Current Configuration</p>
            <div style="margin-top: 0.2rem;">
                <span class="config-tag">{language}</span>
                <span class="config-tag" style="background: rgba(118,75,162,0.2);">{model_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return language, language_code, model_display, model_code, input_method


def render_header(language: str, model_display: str) -> None:
    """Render dashboard header"""
    st.markdown(f"""
    <div class="dashboard-header">
        <h1>💬 Urdu & Roman Urdu Sentiment Analysis</h1>
        <p>Machine Learning Based Sentiment Classification for Social Media Text</p>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem;">
            <span class="badge-info">🌐 Language: <b>{language}</b></span>
            <span class="badge-info">🧠 Model: <b>{model_display}</b></span>
            <span class="badge-info">📊 {datetime.now().strftime("%B %d, %Y")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_statistics() -> None:
    """Render statistics cards"""
    # Calculate percentages
    total = st.session_state.total_predictions
    pos = st.session_state.positive_predictions
    neg = st.session_state.negative_predictions
    
    # Get current model accuracy
    current_language = st.session_state.get('current_language', 'Urdu')
    current_model = st.session_state.get('current_model_display', 'Logistic Regression')
    accuracy = PERFORMANCE_DATA.get((current_language, current_model), 0) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total:,}</div>
            <div class="stat-label">📊 Total Predictions</div>
            {f'<span class="stat-change up">↑ {total - len(st.session_state.history) if total > 0 else 0} new</span>' if total > 0 else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #28a745;">
            <div class="stat-number" style="color: #28a745;">{pos:,}</div>
            <div class="stat-label">😊 Positive</div>
            {f'<span class="stat-change up">{pos/total*100:.1f}%' if total > 0 else ''}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #dc3545;">
            <div class="stat-number" style="color: #dc3545;">{neg:,}</div>
            <div class="stat-label">😞 Negative</div>
            {f'<span class="stat-change down">{neg/total*100:.1f}%' if total > 0 else ''}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #667eea;">
            <div class="stat-number" style="color: #667eea;">{accuracy:.2f}%</div>
            <div class="stat-label">🎯 Model Accuracy</div>
            <span class="stat-change up">Test Set Performance</span>
        </div>
        """, unsafe_allow_html=True)


def render_quick_analysis(model: Any, vectorizer: Any, language: str, model_display: str) -> None:
    """Render quick analysis section on home tab"""
    st.subheader("🎯 Quick Sentiment Analysis")
    
    quick_text = st.text_area(
        "Enter your text for instant analysis:",
        height=130,
        placeholder=f"Type your {language} text here...",
        key="quick_text"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_clicked = st.button("🚀 Analyze", type="primary", key="quick_analyze")
    
    with col2:
        if st.button("🗑️ Clear", key="quick_clear"):
            st.session_state.quick_text = ""
            st.rerun()
    
    if analyze_clicked and quick_text.strip():
        if model is None or vectorizer is None:
            st.error("❌ Model not loaded. Please check your model files.")
            return
        
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence, probabilities = predict_sentiment(
                quick_text, model, vectorizer
            )
        
        if sentiment != "Error":
            # Update statistics
            update_statistics(sentiment)
            add_to_history(quick_text, sentiment, confidence, language, model_display)
            
            # Display result
            badge_class = "badge-positive" if sentiment == "Positive" else "badge-negative"
            emoji = "😊" if sentiment == "Positive" else "😞"
            
            st.markdown(f"""
            <div class="result-container">
                <h3>📊 Analysis Result</h3>
                <div style="display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;">
                    <span class="{badge_class}">{emoji} {sentiment.upper()}</span>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #2c3e50;">
                            Confidence: {confidence*100:.2f}%
                        </div>
                        <div style="width: 200px; background: #e9ecef; height: 8px; border-radius: 5px; margin-top: 0.3rem;">
                            <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {confidence*100}%; border-radius: 5px;"></div>
                        </div>
                    </div>
                </div>
                <hr>
                <p><b>Input Text:</b></p>
                <p style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px;">{quick_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution if available
            if probabilities is not None:
                try:
                    classes = model.classes_
                    prob_data = pd.DataFrame({
                        "Sentiment": [normalize_prediction(c) for c in classes],
                        "Probability": probabilities
                    })
                    
                    fig = px.bar(
                        prob_data,
                        x="Sentiment",
                        y="Probability",
                        text="Probability",
                        title="Prediction Probability Distribution",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive": "#28a745",
                            "Negative": "#dc3545"
                        }
                    )
                    fig.update_traces(
                        texttemplate="%{text:.2%}",
                        textposition="outside"
                    )
                    fig.update_layout(
                        yaxis_range=[0, 1],
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass


def render_sentiment_distribution() -> None:
    """Render sentiment distribution pie chart"""
    st.subheader("📊 Sentiment Distribution")
    
    positive = st.session_state.positive_predictions
    negative = st.session_state.negative_predictions
    
    if positive + negative > 0:
        df_dist = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive, negative]
        })
        
        fig = px.pie(
            df_dist,
            values="Count",
            names="Sentiment",
            hole=0.4,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#28a745",
                "Negative": "#dc3545"
            },
            title=f"Total: {positive + negative} predictions"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Prediction distribution will appear after analysis.")


def render_features() -> None:
    """Render feature cards"""
    st.markdown("---")
    st.subheader("✨ System Features")
    
    features = [
        ("🌐", "Urdu Support", "Sentiment analysis for Urdu text with proper preprocessing"),
        ("🔤", "Roman Urdu", "Analysis of Roman Urdu social media text"),
        ("🧠", "Three ML Models", "Logistic Regression, Naive Bayes, and Linear SVC"),
        ("📊", "Batch Analysis", "Analyze multiple comments using CSV upload"),
        ("📈", "Visual Analytics", "Interactive charts and performance metrics"),
        ("💾", "Export Results", "Download predictions and history as CSV")
    ]
    
    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_single_analysis(model: Any, vectorizer: Any, language: str, model_display: str) -> None:
    """Render single text analysis tab"""
    st.subheader("📝 Single Text Analysis")
    
    st.info(f"💡 Analyzing **{language}** text using **{model_display}**")
    
    user_input = st.text_area(
        "Enter your text:",
        height=180,
        placeholder=f"Write your {language} text here...",
        key="analysis_text"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.analysis_text = ""
            st.rerun()
    
    if analyze and user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence, probabilities = predict_sentiment(
                user_input, model, vectorizer
            )
        
        if sentiment != "Error":
            # Update statistics
            update_statistics(sentiment)
            add_to_history(user_input, sentiment, confidence, language, model_display)
            
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == "Positive":
                    st.success(f"😊 **POSITIVE**")
                    st.markdown('<span class="badge-positive">Positive Sentiment</span>', unsafe_allow_html=True)
                else:
                    st.error(f"😞 **NEGATIVE**")
                    st.markdown('<span class="badge-negative">Negative Sentiment</span>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
            
            st.markdown("---")
            
            # Display text
            with st.expander("📝 View Input Text", expanded=False):
                st.write(user_input)
            
            # Probability distribution
            if probabilities is not None:
                try:
                    classes = model.classes_
                    prob_df = pd.DataFrame({
                        "Sentiment": [normalize_prediction(c) for c in classes],
                        "Probability": probabilities
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x="Sentiment",
                        y="Probability",
                        text="Probability",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive": "#28a745",
                            "Negative": "#dc3545"
                        }
                    )
                    fig.update_traces(
                        texttemplate="%{text:.2%}",
                        textposition="outside"
                    )
                    fig.update_layout(
                        yaxis_range=[0, 1],
                        height=350,
                        showlegend=False,
                        title="Prediction Confidence Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass


def render_batch_analysis(model: Any, vectorizer: Any, language: str, model_display: str) -> None:
    """Render batch analysis tab"""
    st.subheader("📤 Batch Sentiment Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with text column",
        type=["csv"],
        help="CSV file should contain a column with text to analyze"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows")
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            text_column = st.selectbox(
                "Select text column:",
                df.columns.tolist()
            )
            
            if st.button("🚀 Analyze All", type="primary", use_container_width=True):
                sentiments = []
                confidences = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, text in enumerate(df[text_column].astype(str)):
                    status_text.text(f"Analyzing {idx+1}/{len(df)}...")
                    
                    sentiment, confidence, _ = predict_sentiment(
                        text, model, vectorizer
                    )
                    
                    sentiments.append(sentiment if sentiment != "Error" else "Unknown")
                    confidences.append(confidence * 100 if confidence else 0)
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                status_text.empty()
                progress_bar.empty()
                
                # Add results to dataframe
                df["Sentiment"] = sentiments
                df["Confidence (%)"] = confidences
                
                # Update statistics
                positive_count = sum(df["Sentiment"] == "Positive")
                negative_count = sum(df["Sentiment"] == "Negative")
                
                st.session_state.total_predictions += len(df)
                st.session_state.positive_predictions += positive_count
                st.session_state.negative_predictions += negative_count
                
                # Add to history
                for _, row in df.iterrows():
                    add_to_history(
                        row[text_column],
                        row["Sentiment"],
                        row["Confidence (%)"] / 100,
                        language,
                        model_display
                    )
                
                st.success("✅ Analysis completed!")
                
                st.subheader("📊 Batch Results")
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv_data,
                    f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("📈 Results Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    counts = df["Sentiment"].value_counts()
                    fig = px.pie(
                        values=counts.values,
                        names=counts.index,
                        hole=0.4,
                        color=counts.index,
                        color_discrete_map={
                            "Positive": "#28a745",
                            "Negative": "#dc3545",
                            "Unknown": "#6c757d"
                        },
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        df,
                        x="Confidence (%)",
                        nbins=20,
                        title="Confidence Distribution",
                        color_discrete_sequence=["#667eea"]
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")


def render_analytics() -> None:
    """Render analytics tab"""
    st.subheader("📊 Sentiment Analytics Dashboard")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    total = st.session_state.total_predictions
    pos = st.session_state.positive_predictions
    neg = st.session_state.negative_predictions
    
    with col1:
        st.metric(
            "Total Predictions",
            f"{total:,}",
            delta=f"{len(st.session_state.history)} in this session"
        )
    
    with col2:
        pos_rate = pos / total * 100 if total > 0 else 0
        st.metric(
            "Positive Rate",
            f"{pos_rate:.1f}%",
            delta=f"{pos:,} total"
        )
    
    with col3:
        neg_rate = neg / total * 100 if total > 0 else 0
        st.metric(
            "Negative Rate",
            f"{neg_rate:.1f}%",
            delta=f"{neg:,} total"
        )
    
    st.markdown("---")
    
    # Model performance comparison
    st.subheader("🎯 Model Performance Comparison")
    
    performance_data = []
    for (lang, model), accuracy in PERFORMANCE_DATA.items():
        performance_data.append({
            "Language": lang,
            "Model": model,
            "Accuracy (%)": round(accuracy * 100, 2)
        })
    
    df_performance = pd.DataFrame(performance_data)
    
    fig = px.bar(
        df_performance,
        x="Model",
        y="Accuracy (%)",
        color="Language",
        barmode="group",
        text="Accuracy (%)",
        title="Urdu vs Roman Urdu Model Accuracy Comparison",
        color_discrete_sequence=["#667eea", "#764ba2"]
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )
    fig.update_layout(
        yaxis_range=[0, 100],
        height=450,
        xaxis_title="Model",
        yaxis_title="Accuracy (%)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.subheader("📋 Performance Summary Table")
    st.dataframe(
        df_performance,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Prediction history
    st.subheader("📜 Prediction History")
    
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        
        # Add color highlighting
        def color_sentiment(val):
            if val == "Positive":
                return "color: #28a745; font-weight: bold"
            elif val == "Negative":
                return "color: #dc3545; font-weight: bold"
            return ""
        
        # Try both methods for compatibility
        try:
            styled_df = history_df.style.map(color_sentiment, subset=["Sentiment"])
        except AttributeError:
            styled_df = history_df.style.applymap(color_sentiment, subset=["Sentiment"])
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", width="small"),
                "Text": st.column_config.TextColumn("Text", width="large"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.2f%%")
            }
        )
        
        # Download history
        history_csv = history_df.to_csv(index=False)
        st.download_button(
            "📥 Download Prediction History",
            history_csv,
            f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("No predictions have been made in this session.")


def render_settings(
    language: str,
    language_code: str,
    model_display: str,
    model_code: str,
    model_loaded: bool
) -> None:
    """Render settings tab"""
    st.subheader("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h4>🧠 Current Model Configuration</h4>
            <div style="margin-top: 1rem;">
                <p><strong>Model:</strong> {}</p>
                <p><strong>Language:</strong> {}</p>
                <p><strong>Status:</strong> <span style="color: {}; font-weight: 600;">{}</span></p>
                <p><strong>Model Code:</strong> <code>{}</code></p>
                <p><strong>Language Code:</strong> <code>{}</code></p>
            </div>
        </div>
        """.format(
            model_display,
            language,
            "#28a745" if model_loaded else "#dc3545",
            "✅ Loaded" if model_loaded else "❌ Not Loaded",
            model_code,
            language_code
        ), unsafe_allow_html=True)
        
        # Model description
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-top: 1rem;">
            <h4>📖 Model Description</h4>
            <p style="margin-top: 0.5rem;">{}</p>
        </div>
        """.format(MODEL_DESCRIPTIONS.get(model_display, "No description available.")), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h4>📁 Project Structure</h4>
            <div style="margin-top: 1rem;">
                <p><strong>Models Directory:</strong><br><code>{}</code></p>
                <p><strong>Vectorizers Directory:</strong><br><code>{}</code></p>
            </div>
        </div>
        """.format(MODELS_DIR, VECTORIZERS_DIR), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-top: 1rem;">
            <h4>📊 Research Configuration</h4>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                <p><strong>Task:</strong> Binary Sentiment Classification</p>
                <p><strong>Classes:</strong> Positive, Negative</p>
                <p><strong>Feature:</strong> TF-IDF Vectorization</p>
                <p><strong>Models:</strong> Logistic Regression, Naive Bayes, Linear SVC</p>
                <p><strong>Languages:</strong> Urdu, Roman Urdu</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Clear data section
    st.markdown("---")
    st.subheader("🗑️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear All Predictions", use_container_width=True):
            st.session_state.total_predictions = 0
            st.session_state.positive_predictions = 0
            st.session_state.negative_predictions = 0
            st.session_state.history = []
            st.success("✅ All predictions cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_resource.clear()
            st.success("✅ Session reset! Please refresh.")
            st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    
    # Render sidebar and get configurations
    language, language_code, model_display, model_code, input_method = render_sidebar()
    
    # Store current selections in session state
    st.session_state.current_language = language
    st.session_state.current_model_display = model_display
    
    # Load model
    model, vectorizer, model_path, vectorizer_path = load_selected_model(
        language_code, model_code
    )
    
    model_loaded = model is not None and vectorizer is not None
    st.session_state.model_loaded = model_loaded
    
    # Render header
    render_header(language, model_display)
    
    # Display model status
    if model_loaded:
        st.success(f"✅ {model_display} and {language} TF-IDF vectorizer loaded successfully.")
        if model_path:
            st.caption(f"📁 Model: {model_path.name}")
    else:
        st.error("❌ Model or vectorizer could not be loaded.")
        st.info(f"""
        **Expected files:**
        - Model: `models/{language_code}_{model_code}.pkl`
        - Vectorizer: `vectorizers/{language_code}_tfidf.pkl`
        
        **Troubleshooting:**
        1. Make sure files exist in the correct directories
        2. Check file permissions
        3. Try refreshing models from sidebar
        """)
    
    # Render statistics
    render_statistics()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home",
        "📝 Analyze",
        "📊 Analytics",
        "⚙️ Settings"
    ])
    
    # Home tab
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_quick_analysis(model, vectorizer, language, model_display)
        
        with col2:
            render_sentiment_distribution()
        
        render_features()
    
    # Analyze tab
    with tab2:
        if not model_loaded:
            st.warning("⚠️ Model not loaded. Please check your model files.")
        else:
            if input_method == "Single Text":
                render_single_analysis(model, vectorizer, language, model_display)
            else:
                render_batch_analysis(model, vectorizer, language, model_display)
    
    # Analytics tab
    with tab3:
        render_analytics()
    
    # Settings tab
    with tab4:
        render_settings(
            language, language_code, model_display, model_code, model_loaded
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p style="font-size: 0.95rem;">💬 Urdu & Roman Urdu Sentiment Analysis</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">
            Final Year Project | Machine Learning & Natural Language Processing
        </p>
        <p style="font-size: 0.8rem; opacity: 0.6;">
            © 2026 | Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()