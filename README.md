# DSA 4213 Group 45 Political Bias Classifier

A web application for analyzing political bias in news articles using fine-tuned transformer models (DeBERTa and Longformer).

## Features

- **Dual Model System**: Automatically switches between DeBERTa (short articles ≤512 tokens) and Longformer (long articles up to 4096 tokens)
- **3-Label Classification**: Classifies articles as Left, Centre, or Right
- **5-Label Classification**: Classifies articles as Left, Left Leaning, Centre, Right Leaning, or Right
- **URL Extraction**: Automatically extract article content from URLs
- **AI Perspectives**: Generate alternative viewpoints using Google Gemini AI (optional)
- **Related Articles**: Find and analyze related articles on the same topic
- **Interactive Visualizations**: Spectrum visualization and probability distributions
- Shapley (SHAP) Feature Importance Highlighting (Currently disabled as requires high computational power to analyze)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

Download the project to your local machine.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Streamlit
- PyTorch
- Transformers
- PEFT (for LoRA adapters)
- newspaper3k (for article extraction)
- newsapi-python (for finding related articles)
- google-generativeai (for AI perspectives)
- And other dependencies

### Step 3: Set Up Environment Variables (Optional)

Create a `.env` file in the project root directory:

```bash
# For AI Perspectives feature
GEMINI_API_KEY=your_gemini_api_key_here

# For Related Articles feature
NEWSAPI_KEY=your_newsapi_key_here
```

**Get API Keys:**
- **Gemini API**: https://makersuite.google.com/app/apikey
- **NewsAPI**: https://newsapi.org/register

Note: The app will work without these keys, but some features will be disabled.

### Step 4: Verify Model Files

Ensure the following model directories exist:
Either download the models pre-fine-tuned by us or rerun the fine-tuning notebooks to get them

Run in terminal to download model after creating venv
```
.\download_models.ps1
```

```
models/
├── deberta_model/          # DeBERTa with LoRA adapters
└── longformer-finetuned-model/  # Longformer model
```

## Running the Application

### From the frontend directory:

```bash
cd frontend
streamlit run streamlit_app.py
```

### From the project root:

```bash
streamlit run frontend/streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Analyze Article Text

**Method A: Paste Text**
1. Click on the "Paste Text" tab
2. Paste your article text into the text area
3. Click "Analyze Bias"
4. View the results including bias label, confidence, and probability distribution

**Method B: Extract from URL**
1. Click on the "Extract from URL" tab
2. Enter the article URL
3. Click "Extract Article"
4. Review the extracted content
5. Click "Analyze Bias"

### 2. Understanding Results

**Bias Classification:**
- **Left**: Left-leaning or progressive bias
- **Centre**: Neutral or centrist perspective
- **Right**: Right-leaning or conservative bias

**Model Information:**
- Short articles (≤512 tokens): Uses DeBERTa with LoRA
- Long articles (>512 tokens): Automatically switches to Longformer
- The model used is displayed in the results

**5-Label Breakdown (Longformer only):**
- For long articles, you can view the detailed 5-label output
- Labels: left, lean left, center, lean right, right
- These are mapped to the 3-label system

### 3. AI Perspectives (Optional)

Enable this feature in the sidebar to:
- Generate opposing viewpoints on the article
- View a balanced synthesis
- Get discussion questions for deeper analysis

Requirements:
- GEMINI_API_KEY must be set in .env file

### 4. Find Related Articles (Optional)

1. Scroll to the "Find Related Articles" section
2. Enter a seed article URL
3. Set the maximum number of results
4. Click "Find Related Articles"
5. View bias analysis for each related article

Requirements:
- NEWSAPI_KEY must be set in .env file
- 
**Enable Word Importance Analysis**
```
# set get_attributions to True to enable, line 150 in model_loader.py
def predict(self, text: str, max_length: int = 512, get_attributions: bool = False) -> Dict[str, any]:
```
- Toggle SHAP-based word highlighting to see which words influenced the prediction
- Shows highlighted text with important words marked
- Displays top 10 most influential words per bias category
- Much Slower processing (~30-60 seconds) due to SHAP computation
- Disabled by default for faster performance
- Only available for short model (DeBERTa) - not supported for Longformer
