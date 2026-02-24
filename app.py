"""
Financial Sentiment Analysis - Interactive Demo
Real-time sentiment prediction using FinBERT transformer model
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .sentiment-positive {
        color: #00cc00;
        font-weight: bold;
        font-size: 24px;
    }
    .sentiment-negative {
        color: #ff3333;
        font-weight: bold;
        font-size: 24px;
    }
    .sentiment-neutral {
        color: #ff9900;
        font-weight: bold;
        font-size: 24px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load FinBERT model and tokenizer"""
    with st.spinner("Loading FinBERT model... (this may take a minute on first run)"):
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
    return tokenizer, model

# Prediction function
def predict_sentiment(text, tokenizer, model):
    """Predict sentiment for given text"""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get results
    probabilities = predictions[0].cpu().numpy()
    labels = ["Negative", "Neutral", "Positive"]
    predicted_class = probabilities.argmax()

    return {
        "sentiment": labels[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            "Positive": float(probabilities[2]),
            "Neutral": float(probabilities[1]),
            "Negative": float(probabilities[0])
        }
    }

# Visualization functions
def create_gauge_chart(confidence, sentiment):
    """Create a gauge chart for confidence"""
    color = {
        "Positive": "#00cc00",
        "Negative": "#ff3333",
        "Neutral": "#ff9900"
    }[sentiment]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_probability_chart(probabilities):
    """Create a bar chart for probabilities"""
    df = pd.DataFrame({
        'Sentiment': list(probabilities.keys()),
        'Probability': [v * 100 for v in probabilities.values()]
    })

    colors = {
        'Positive': '#00cc00',
        'Neutral': '#ff9900',
        'Negative': '#ff3333'
    }

    fig = go.Figure(data=[
        go.Bar(
            x=df['Sentiment'],
            y=df['Probability'],
            marker_color=[colors[s] for s in df['Sentiment']],
            text=[f"{p:.1f}%" for p in df['Probability']],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Sentiment Probability Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 100])
    )

    return fig

# Example texts
EXAMPLES = {
    "Positive News": "Apple reports record quarterly earnings, beating analyst expectations with strong iPhone sales and services growth.",
    "Negative News": "Tesla stock plunges 15% after missing production targets and CEO announces major layoffs amid economic uncertainty.",
    "Neutral News": "The Federal Reserve will hold its next meeting on monetary policy next week, with markets expecting no change in interest rates.",
    "Earnings Beat": "Microsoft exceeded Wall Street expectations with cloud revenue surging 30% year-over-year, driving stock to all-time highs.",
    "Market Crash": "Global markets tumble as recession fears intensify, with major indices posting worst weekly performance since 2020.",
    "IPO Announcement": "Tech startup files for IPO, seeking to raise $500 million in one of the year's most anticipated public offerings.",
    "Merger News": "Two pharmaceutical giants announce $50 billion merger, creating industry leader in biotech innovation.",
    "Reddit WSB Style": "TSLA to the moon! 🚀🚀🚀 Diamond hands baby, we ain't selling! Bulls are back in town! 💎🙌"
}

# Main app
def main():
    # Header
    st.title("📈 Financial Sentiment Analysis")
    st.markdown("### Enterprise-Grade MLOps Pipeline for Real-Time Sentiment Detection")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/financial-growth-analysis.png", width=100)
        st.markdown("## About")
        st.info(
            """
            This demo showcases an enterprise-grade **Financial Sentiment Analysis** system
            built with MLOps best practices.

            **Features:**
            - FinBERT transformer model
            - Real-time predictions
            - Confidence scoring
            - Multi-source text support

            **Tech Stack:**
            - PyTorch & Transformers
            - MLflow for tracking
            - FastAPI for production
            - Docker & Kubernetes
            - Prometheus monitoring
            """
        )

        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/priyankabolem/financial-sentiment-mlops)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://linkedin.com/in/priyankabolem)")

        st.markdown("---")
        st.markdown("### ⚡ Model Info")
        st.markdown("**Model:** FinBERT")
        st.markdown("**Provider:** ProsusAI")
        st.markdown("**Classes:** 3 (Pos/Neg/Neu)")
        st.markdown("**Max Length:** 512 tokens")

    # Load model
    try:
        tokenizer, model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 📝 Quick Examples")
        example_choice = st.selectbox(
            "Select an example:",
            ["Custom Input"] + list(EXAMPLES.keys())
        )

    with col1:
        st.markdown("### 💬 Enter Financial Text")

        # Text input
        if example_choice == "Custom Input":
            text_input = st.text_area(
                "Enter financial news, tweet, or any market-related text:",
                height=150,
                placeholder="e.g., 'Stock market rallies on strong economic data...'"
            )
        else:
            text_input = st.text_area(
                "Enter financial news, tweet, or any market-related text:",
                value=EXAMPLES[example_choice],
                height=150
            )

        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            predict_button = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    # Prediction
    if predict_button:
        if not text_input or len(text_input.strip()) < 10:
            st.warning("⚠️ Please enter at least 10 characters of text.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Simulate processing time for effect
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress_bar.progress(i + 1)

                # Get prediction
                result = predict_sentiment(text_input, tokenizer, model)

                # Clear progress bar
                progress_bar.empty()

            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # Sentiment with colored styling
            sentiment = result['sentiment']
            confidence = result['confidence']

            sentiment_class = f"sentiment-{sentiment.lower()}"

            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                st.markdown("### Detected Sentiment")
                st.markdown(f'<p class="{sentiment_class}">{sentiment}</p>', unsafe_allow_html=True)

            with col_res2:
                st.markdown("### Confidence")
                st.metric("", f"{confidence*100:.1f}%")

            with col_res3:
                st.markdown("### Timestamp")
                st.metric("", datetime.now().strftime("%H:%M:%S"))

            # Visualizations
            st.markdown("---")
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                st.plotly_chart(
                    create_gauge_chart(confidence, sentiment),
                    use_container_width=True
                )

            with col_viz2:
                st.plotly_chart(
                    create_probability_chart(result['probabilities']),
                    use_container_width=True
                )

            # Detailed probabilities
            st.markdown("---")
            st.markdown("### 📈 Detailed Breakdown")

            prob_df = pd.DataFrame({
                'Sentiment': list(result['probabilities'].keys()),
                'Probability': [f"{v*100:.2f}%" for v in result['probabilities'].values()],
                'Confidence Level': [
                    '🔴 Low' if v < 0.5 else '🟡 Medium' if v < 0.75 else '🟢 High'
                    for v in result['probabilities'].values()
                ]
            })

            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Interpretation
            st.markdown("---")
            st.markdown("### 🎯 Interpretation")

            if confidence > 0.8:
                interpretation = f"The model is **highly confident** ({confidence*100:.1f}%) that this text expresses **{sentiment.lower()}** sentiment."
            elif confidence > 0.6:
                interpretation = f"The model is **moderately confident** ({confidence*100:.1f}%) that this text expresses **{sentiment.lower()}** sentiment."
            else:
                interpretation = f"The model has **low confidence** ({confidence*100:.1f}%) in this prediction. The sentiment may be mixed or unclear."

            st.info(interpretation)

            # Trading signal (fun addition)
            st.markdown("---")
            st.markdown("### 📊 Hypothetical Trading Signal")

            if sentiment == "Positive" and confidence > 0.75:
                st.success("🟢 **BULLISH** - Strong positive sentiment detected")
            elif sentiment == "Negative" and confidence > 0.75:
                st.error("🔴 **BEARISH** - Strong negative sentiment detected")
            else:
                st.warning("🟡 **NEUTRAL** - Hold position or wait for clearer signals")

            st.caption("⚠️ This is for demonstration only. Not financial advice!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Financial Sentiment Analysis Platform</p>
            <p><strong>Priyanka Bolem</strong> | ML Engineer</p>
            <p><a href='https://github.com/priyankabolem/financial-sentiment-mlops'>GitHub</a> |
            <a href='https://linkedin.com/in/priyankabolem'>LinkedIn</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
