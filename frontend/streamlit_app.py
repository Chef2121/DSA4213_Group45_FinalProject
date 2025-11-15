"""
Streamlit application for article bias detection.
Uses the fine-tuned LoRA BERT model to classify political bias in articles.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import from backend
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import from backend
from backend.model_loader import BiasClassifier
from backend.article_extractor import ArticleExtractor
from backend.perspective_generator import PerspectiveGenerator
from backend.article_fetcher import ArticleFetcher

# Page configuration
st.set_page_config(
    page_title="Separating Signal from Spin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .left-bias {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
    }
    .centre-bias {
        background-color: #f3e5f5;
        border: 2px solid #9c27b0;
    }
    .right-bias {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .confidence-text {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .label-text {
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the bias classification models (cached)."""
    # Load both short and long models
    # Paths are relative to the frontend directory
    return BiasClassifier(
        short_model_path="../models/deberta_model",
        long_model_path="../models/longformer-finetuned-model",
        token_threshold=512
    )


def get_bias_color(label: str) -> str:
    """Get CSS class for bias label."""
    label_lower = label.lower()
    color_map = {
        "left": "left-bias",
        "centre": "centre-bias",
        "center": "centre-bias",
        "right": "right-bias"
    }
    return color_map.get(label_lower, "centre-bias")


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">Separating Signal from Spin</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Analyze political bias in news articles using ModernBERT</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses **two fine-tuned models** to classify 
        the political bias of news articles.
        
        **Bias Categories:**
        - **Left**: Left-leaning bias
        - **Centre**: Neutral or centrist
        - **Right**: Right-leaning bias
        
        **How to use:**
        1. Paste or type your article text, or extract from URL
        2. Click "Analyze Bias"
        3. View the classification and confidence scores
        
        **Model Details:**
        - **Short articles (≤512 tokens)**: DeBERTa with LoRA adapters
        - **Long articles (>512 tokens)**: Longformer (up to 4096 tokens)
        - Automatic model selection based on article length
        """)
        
        st.header("Settings")
        max_length = st.slider(
            "Max Token Length",
            min_value=128,
            max_value=512,
            value=512,
            step=64,
            help="Maximum number of tokens to process"
        )
        
        show_probabilities = st.checkbox(
            "Show All Probabilities",
            value=True,
            help="Display probability scores for all classes"
        )
        
        st.markdown("---")
        
        st.header("AI Perspectives")
        enable_perspectives = st.checkbox(
            "Generate Opposing Viewpoints",
            value=False,
            help="Use AI to generate alternative perspectives on the article"
        )
        
        if enable_perspectives:
            # Check if API key is available in environment
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            if not os.getenv("GEMINI_API_KEY"):
                st.warning("GEMINI_API_KEY not found in .env file. Please add it to enable this feature.")
                st.info("Get a free API key at: https://makersuite.google.com/app/apikey")
            else:
                st.success("API key loaded from .env file")
    
    # Main content
    st.markdown("---")
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            classifier = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Input section
    st.header("Input Article")
    
    # Add tabs for different input methods
    input_tab1, input_tab2 = st.tabs(["Paste Text", "Extract from URL"])
    
    article_text = ""
    article_title = None
    article_metadata = None
    
    with input_tab1:
        article_text = st.text_area(
            "Paste your article text here:",
            value="",
            height=300,
            placeholder="Enter the article text you want to analyze for political bias...",
            key="text_input"
        )
    
    with input_tab2:
        st.markdown("Enter a URL to automatically extract and analyze the article content.")
        
        # URL input
        article_url = st.text_input(
            "Article URL:",
            placeholder="https://example.com/article",
            help="Enter the full URL of the news article"
        )
        
        # Extract button
        if st.button("Extract Article", type="secondary"):
            if not article_url.strip():
                st.warning("Please enter a URL.")
            else:
                with st.spinner("Extracting article from URL..."):
                    try:
                        extractor = ArticleExtractor()
                        extraction_result = extractor.extract_from_url(article_url)
                        
                        if extraction_result["success"]:
                            article_text = extraction_result["text"]
                            article_title = extraction_result["title"]
                            article_metadata = extraction_result
                            
                            st.success(f"Successfully extracted: **{article_title}**")
                            
                            with st.expander("Article Metadata"):
                                if extraction_result.get("authors"):
                                    st.write(f"**Authors:** {', '.join(extraction_result['authors'])}")
                                if extraction_result.get("publish_date"):
                                    st.write(f"**Published:** {extraction_result['publish_date']}")
                                st.write(f"**URL:** {extraction_result['url']}")
                                if extraction_result.get("top_image"):
                                    st.image(extraction_result['top_image'], caption="Article Image", width='stretch')
                            
                            # Show preview
                            preview = extractor.get_article_preview(article_text, max_words=100)
                            st.text_area(
                                "Extracted Article Preview:",
                                value=preview,
                                height=150,
                                disabled=True,
                                key="url_preview"
                            )
                            
                            st.session_state['extracted_text'] = article_text
                            st.session_state['extracted_title'] = article_title
                            st.session_state['extracted_metadata'] = article_metadata
                            
                        else:
                            st.error(f"{extraction_result['error']}")
                            
                    except Exception as e:
                        st.error(f"Error extracting article: {str(e)}")
        
        if 'extracted_text' in st.session_state and st.session_state.get('extracted_text'):
            article_text = st.session_state['extracted_text']
            article_title = st.session_state.get('extracted_title')
            article_metadata = st.session_state.get('extracted_metadata')
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("Analyze Bias", type="primary", width='stretch')
    
    if analyze_button:
        if not article_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing article bias..."):
                try:
                    result = classifier.predict(article_text, max_length=max_length, get_attributions=False)
                    
                    st.markdown("---")
                    
                    if article_title:
                        st.header(f"{article_title}")
                        if article_metadata and article_metadata.get("url"):
                            st.markdown(f"**Source:** [{article_metadata['url']}]({article_metadata['url']})")
                        st.markdown("---")
                    
                    st.header("Analysis Results")
                    
                    model_used = result.get("model_used", "unknown")
                    num_tokens = result.get("num_tokens", 0)
                    
                    if model_used == "short":
                        st.info(f"**Model Used:** DeBERTa with LoRA ({num_tokens} tokens)")
                    elif model_used == "long":
                        st.success(f"**Model Used:** Longformer ({num_tokens} tokens - switched to long model for better accuracy)")
                        
                        if "raw_5_label_probs" in result:
                            with st.expander("View Detailed 5-Label Breakdown (Longformer Output)"):
                                raw_probs = result["raw_5_label_probs"]
                                st.markdown("""
                                The Longformer model provides more granular predictions with 5 labels, 
                                which are then mapped to the 3-label system:
                                """)
                                
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("Left", f"{raw_probs['left']*100:.1f}%")
                                with col2:
                                    st.metric("Lean Left", f"{raw_probs['lean left']*100:.1f}%")
                                with col3:
                                    st.metric("Center", f"{raw_probs['center']*100:.1f}%")
                                with col4:
                                    st.metric("Lean Right", f"{raw_probs['lean right']*100:.1f}%")
                                with col5:
                                    st.metric("Right", f"{raw_probs['right']*100:.1f}%")
                                
                                st.markdown("""
                                **Mapping to 3-label system:**
                                - **Left** = Left + Lean Left
                                - **Centre** = Center
                                - **Right** = Lean Right + Right
                                """)
                    
                    st.subheader("Bias Classification")
                    
                    left_prob = result['probabilities']['Left']
                    centre_prob = result['probabilities']['Centre']
                    right_prob = result['probabilities']['Right']
                    
                    spectrum_position = (left_prob * 0 + centre_prob * 50 + right_prob * 100)
                    
                    st.markdown(f"""
                        <div style="margin: 2rem 0;">
                            <div style="background: linear-gradient(to right, #1976d2 0%, #9c27b0 50%, #d32f2f 100%); 
                                        height: 60px; 
                                        border-radius: 30px; 
                                        position: relative;
                                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                                <div style="position: absolute; 
                                            left: {spectrum_position}%; 
                                            top: 50%; 
                                            transform: translate(-50%, -50%);
                                            width: 80px;
                                            height: 80px;
                                            background: white;
                                            border: 4px solid #333;
                                            border-radius: 50%;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            font-weight: bold;
                                            font-size: 0.9rem;
                                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                                            z-index: 10;">
                                    {result["label"]}
                                </div>
                            </div>
                            <div style="display: flex; 
                                        justify-content: space-between; 
                                        margin-top: 1rem;
                                        font-weight: bold;
                                        color: #666;">
                                <span style="color: #1976d2;">Left</span>
                                <span style="color: #9c27b0;">Centre</span>
                                <span style="color: #d32f2f;">Right</span>
                            </div>
                            <div style="text-align: center; margin-top: 1.5rem; font-size: 1.2rem;">
                                <strong>Confidence:</strong> {result["confidence"]*100:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if show_probabilities:
                        st.subheader("Probability Distribution")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Left",
                                f"{result['probabilities']['Left']*100:.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Centre",
                                f"{result['probabilities']['Centre']*100:.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Right",
                                f"{result['probabilities']['Right']*100:.1f}%"
                            )
                        
                        import pandas as pd
                        prob_df = pd.DataFrame({
                            'Bias': ['Left', 'Centre', 'Right'],
                            'Probability': [
                                result['probabilities']['Left'],
                                result['probabilities']['Centre'],
                                result['probabilities']['Right']
                            ]
                        })
                        st.bar_chart(prob_df.set_index('Bias'))
                    
                    if result.get("word_importance"):
                        st.markdown("---")
                        st.header("What the Model Looks For")
                        
                        st.markdown("""
                            <div style="font-size: 0.95em; color: #666; margin-bottom: 1rem;">
                                The model analyzes the text differently for each bias category. 
                                Words are highlighted based on their importance to each prediction.
                                <strong>Brighter colors = more important to that specific bias classification.</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        highlighted_versions = classifier.highlight_text_with_attributions(
                            article_text, 
                            result["word_importance"]
                        )
                        
                        if highlighted_versions:
                            tab1, tab2, tab3 = st.tabs([
                                "Left View", 
                                "Centre View",
                                "Right View"
                            ])
                            
                            with tab1:
                                st.markdown(f"""
                                    <div style="padding: 0.5rem; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 1rem;">
                                        <strong>Left Bias Indicators (Score: {result['probabilities']['Left']*100:.1f}%)</strong><br>
                                        <small>Highlighted words contributed to the LEFT bias score</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div style="background-color: #f5f5f5; padding: 1.5rem; border-radius: 10px; 
                                                border: 2px solid #1976d2; line-height: 1.8; font-size: 1rem;">
                                        {highlighted_versions.get("Left", article_text)}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if result["word_importance"].get("Left"):
                                    top_words = sorted(result["word_importance"]["Left"], key=lambda x: x[1], reverse=True)[:10]
                                    if top_words:
                                        st.markdown("**Top 10 words influencing Left score:**")
                                        words_str = ", ".join([f"{word} ({score:.2f})" for word, score in top_words])
                                        st.info(words_str)
                            
                            with tab2:
                                st.markdown(f"""
                                    <div style="padding: 0.5rem; background-color: #f3e5f5; border-radius: 5px; margin-bottom: 1rem;">
                                        <strong>Centre Bias Indicators (Score: {result['probabilities']['Centre']*100:.1f}%)</strong><br>
                                        <small>Highlighted words contributed to the CENTRE bias score</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div style="background-color: #f5f5f5; padding: 1.5rem; border-radius: 10px; 
                                                border: 2px solid #9c27b0; line-height: 1.8; font-size: 1rem;">
                                        {highlighted_versions.get("Centre", article_text)}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if result["word_importance"].get("Centre"):
                                    top_words = sorted(result["word_importance"]["Centre"], key=lambda x: x[1], reverse=True)[:10]
                                    if top_words:
                                        st.markdown("**Top 10 words influencing Centre score:**")
                                        words_str = ", ".join([f"{word} ({score:.2f})" for word, score in top_words])
                                        st.info(words_str)
                            
                            with tab3:
                                st.markdown(f"""
                                    <div style="padding: 0.5rem; background-color: #ffebee; border-radius: 5px; margin-bottom: 1rem;">
                                        <strong>Right Bias Indicators (Score: {result['probabilities']['Right']*100:.1f}%)</strong><br>
                                        <small>Highlighted words contributed to the RIGHT bias score</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div style="background-color: #f5f5f5; padding: 1.5rem; border-radius: 10px; 
                                                border: 2px solid #d32f2f; line-height: 1.8; font-size: 1rem;">
                                        {highlighted_versions.get("Right", article_text)}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if result["word_importance"].get("Right"):
                                    top_words = sorted(result["word_importance"]["Right"], key=lambda x: x[1], reverse=True)[:10]
                                    if top_words:
                                        st.markdown("**Top 10 words influencing Right score:**")
                                        words_str = ", ".join([f"{word} ({score:.2f})" for word, score in top_words])
                                        st.info(words_str)
                    
                    st.markdown("---")
                    st.subheader("Interpretation")
                    
                    confidence_pct = result["confidence"] * 100
                    
                    if confidence_pct >= 70:
                        confidence_desc = "high confidence"
                    elif confidence_pct >= 50:
                        confidence_desc = "moderate confidence"
                    else:
                        confidence_desc = "low confidence"
                    
                    st.info(f"""
                    The model predicts this article has a **{result["label"]}** bias with 
                    **{confidence_desc}** ({confidence_pct:.1f}%).
                    
                    Please note that this is an automated analysis and should be considered 
                    as one tool among many for evaluating media bias.
                    """)
                    
                    if enable_perspectives:
                        import os
                        from dotenv import load_dotenv
                        load_dotenv()
                        
                        gemini_api_key = os.getenv("GEMINI_API_KEY")
                        
                        if gemini_api_key:
                            st.markdown("---")
                            st.header("Alternative Perspectives")
                            
                            with st.spinner("Generating opposing viewpoints using AI..."):
                                try:
                                    perspective_gen = PerspectiveGenerator(api_key=gemini_api_key)
                                    perspectives_result = perspective_gen.generate_opposing_perspectives(
                                        article_text,
                                        result["label"],
                                        max_length=500
                                    )
                                    
                                    if perspectives_result["success"]:
                                        perspectives = perspectives_result["perspectives"]
                                        
                                        st.subheader("Article Summary")
                                        st.write(perspectives.get("summary", ""))
                                        
                                        st.markdown("---")
                                        
                                        if result["label"].lower() == "left":
                                            opposing_views = [
                                                ("Right-Leaning Perspective", "right", "#d32f2f"),
                                                ("Centrist Perspective", "centre", "#9c27b0")
                                            ]
                                        elif result["label"].lower() == "right":
                                            opposing_views = [
                                                ("Left-Leaning Perspective", "left", "#1976d2"),
                                                ("Centrist Perspective", "centre", "#9c27b0")
                                            ]
                                        else:  # Centre
                                            opposing_views = [
                                                ("Left-Leaning Perspective", "left", "#1976d2"),
                                                ("Right-Leaning Perspective", "right", "#d32f2f")
                                            ]
                                        
                                        for title, key, color in opposing_views:
                                            with st.expander(title, expanded=True):
                                                st.markdown(f"""
                                                    <div style="padding: 1rem; background-color: {color}15; 
                                                                border-left: 4px solid {color}; border-radius: 5px;">
                                                        {perspectives.get(key, "Not available")}
                                                    </div>
                                                """, unsafe_allow_html=True)
                                        
                                        st.markdown("---")
                                        st.subheader("Balanced Synthesis")
                                        st.markdown(f"""
                                            <div style="padding: 1.5rem; background-color: #e8f5e9; 
                                                        border: 2px solid #4caf50; border-radius: 10px;">
                                                {perspectives.get("balanced", "Not available")}
                                            </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown("---")
                                        st.subheader("Discussion Questions")
                                        
                                        with st.spinner("Generating discussion questions..."):
                                            questions = perspective_gen.generate_discussion_questions(
                                                article_text,
                                                result["label"]
                                            )
                                            
                                            if questions:
                                                for i, question in enumerate(questions, 1):
                                                    st.markdown(f"**{i}.** {question}")
                                            else:
                                                st.write("No questions generated.")
                                    
                                    else:
                                        st.error(f"{perspectives_result['error']}")
                                        
                                except Exception as e:
                                    st.error(f"Error generating perspectives: {str(e)}")
                        else:
                            st.warning("GEMINI_API_KEY not found in .env file")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    st.markdown("---")
    st.header("Find Related Articles")
    st.markdown("""
        Discover how different news sources cover the same topic. Enter a seed article URL 
        to find related articles and analyze their political bias.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        seed_url = st.text_input(
            "Seed Article URL:",
            value=article_metadata.get("url") if article_metadata else "",
            placeholder="https://example.com/article",
            help="URL of the article to find similar articles about",
            key="seed_url_input"
        )
    with col2:
        max_related = st.number_input(
            "Max Results:",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Maximum number of related articles to fetch"
        )
    
    find_related_btn = st.button("Find Related Articles", type="secondary")
    
    if find_related_btn:
        if not seed_url.strip():
            st.warning("Please enter a seed article URL.")
        else:
            with st.spinner("Searching for related articles..."):
                try:
                    import os
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    fetcher = ArticleFetcher()
                    related_articles = fetcher.find_related_articles(seed_url, max_results=max_related)
                    
                    if not related_articles:
                        st.warning("No related articles found. Try a different URL or check your NewsAPI key.")
                    else:
                        st.success(f"Found {len(related_articles)} related articles!")
                        
                        st.markdown("---")
                        st.subheader(f"Related Articles ({len(related_articles)})")
                        
                        progress_bar = st.progress(0)
                        for idx, article in enumerate(related_articles):
                            with st.container():
                                col_a, col_b = st.columns([3, 1])
                                
                                with col_a:
                                    st.markdown(f"### {article.get('title', '(No title)')}")
                                    
                                    source_info = []
                                    if article.get('source'):
                                        source_info.append(f"**Source:** {article['source']}")
                                    if article.get('publishedAt'):
                                        source_info.append(f"**Published:** {article['publishedAt'].strftime('%Y-%m-%d %H:%M')}")
                                    if article.get('length_words'):
                                        source_info.append(f"**Length:** {article['length_words']} words")
                                    
                                    if source_info:
                                        st.markdown(" • ".join(source_info))
                                    
                                    preview_text = article.get('description') or (
                                        article.get('text')[:300] + "..." if article.get('text') else ""
                                    )
                                    if preview_text:
                                        st.markdown(f"_{preview_text}_")
                                    
                                    if article.get('url'):
                                        st.markdown(f"[Read full article]({article['url']})")
                                
                                with col_b:
                                    if article.get('text') and len(article['text'].strip()) > 50:
                                        try:
                                            bias_result = classifier.predict(article['text'], max_length=max_length, get_attributions=False)
                                            
                                            label = bias_result['label']
                                            confidence = bias_result['confidence']
                                            
                                            if label.lower() == 'left':
                                                color = "#1976d2"
                                            elif label.lower() == 'right':
                                                color = "#d32f2f"
                                            else:
                                                color = "#9c27b0"
                                            
                                            st.markdown(f"""
                                                <div style="background-color: {color}15; 
                                                            border: 2px solid {color}; 
                                                            border-radius: 10px; 
                                                            padding: 1rem; 
                                                            text-align: center;">
                                                    <div style="font-weight: bold; color: {color}; font-size: 1.2rem;">
                                                        {label}
                                                    </div>
                                                    <div style="color: #666; margin-top: 0.5rem;">
                                                        {confidence*100:.1f}% confidence
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                        except Exception as e:
                                            st.error(f"Error: {str(e)[:50]}")
                                    else:
                                        st.info("Text too short to analyze")
                                
                                st.markdown("---")
                            
                            progress_bar.progress((idx + 1) / len(related_articles))
                        
                        progress_bar.empty()
                        
                except Exception as e:
                    st.error(f"Error finding related articles: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <small>
                Built with Streamlit | 
                Models: DeBERTa + LoRA (short) & Longformer (long) | 
                DSA4213 Group 45
            </small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
