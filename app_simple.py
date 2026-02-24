"""
Simplified Streamlit Demo - Use this if main app.py has issues
This version has minimal dependencies and loads faster
"""

import streamlit as st

# Test if streamlit works
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Sentiment Analysis - MLOps Demo")
st.markdown("### Loading model... Please wait 1-2 minutes on first run")

# Try to import required packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import pandas as pd

    st.success("✅ All packages loaded successfully!")

    # Cache model loading
    @st.cache_resource
    def load_model():
        model_name = "ProsusAI/finbert"
        with st.spinner("Loading FinBERT model (this may take a minute on first run)..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
        return tokenizer, model

    # Load model
    try:
        tokenizer, model = load_model()
        st.success("✅ Model loaded successfully!")

        # Input
        st.markdown("---")
        text_input = st.text_area(
            "Enter financial text to analyze:",
            value="Apple reports record quarterly earnings, beating analyst expectations.",
            height=100
        )

        if st.button("🔮 Analyze Sentiment", type="primary"):
            if text_input:
                with st.spinner("Analyzing..."):
                    # Tokenize
                    inputs = tokenizer(
                        text_input,
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

                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", labels[predicted_class])
                    with col2:
                        st.metric("Confidence", f"{probabilities[predicted_class]*100:.1f}%")
                    with col3:
                        sentiment_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                        st.metric("Signal", sentiment_emoji[labels[predicted_class]])

                    # Detailed probabilities
                    st.markdown("### Probability Breakdown")
                    for label, prob in zip(labels, [probabilities[0], probabilities[1], probabilities[2]]):
                        st.progress(float(prob), text=f"{label}: {prob*100:.2f}%")
            else:
                st.warning("Please enter some text to analyze.")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("This is normal on first run. Please refresh the page in 1-2 minutes.")

except ImportError as e:
    st.error(f"Missing package: {str(e)}")
    st.info("Please check requirements.txt file.")

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.info("""
    This is a **Financial Sentiment Analysis** demo using FinBERT.

    **Features:**
    - Real-time predictions
    - FinBERT transformer model
    - Confidence scoring

    **Tech Stack:**
    - PyTorch
    - Transformers
    - Streamlit
    """)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repo](https://github.com/priyankabolem/financial-sentiment-mlops)")
    st.markdown("[LinkedIn](https://linkedin.com/in/priyankabolem)")

    st.markdown("---")
    st.markdown("### 📝 Examples")
    examples = {
        "Positive": "Stock market rallies on strong economic data",
        "Negative": "Markets crash amid recession fears",
        "Neutral": "Fed maintains interest rates at current levels"
    }

    if st.button("Load Positive Example"):
        st.session_state.example = examples["Positive"]
    if st.button("Load Negative Example"):
        st.session_state.example = examples["Negative"]
    if st.button("Load Neutral Example"):
        st.session_state.example = examples["Neutral"]

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit, PyTorch, and Transformers</p>
    <p>Part of an Enterprise MLOps Portfolio Project by <strong>Priyanka Bolem</strong></p>
</div>
""", unsafe_allow_html=True)
