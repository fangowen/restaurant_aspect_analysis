# Restaurant Aspect Analysis

This project provides an interactive dashboard for analyzing restaurant reviews using aspect-based sentiment analysis. It leverages NLP and data visualization tools to help users explore review data, compare restaurants, and gain actionable insights.

## Features
- **Streamlit Dashboard**: Interactive web app for visualizing sentiment and aspect analysis
- **Aspect Extraction & Sentiment Analysis**: NLP modules for extracting aspects and classifying sentiment
- **Word Clouds**: Visualize frequent words by sentiment
- **Restaurant Comparison**: Compare restaurants across multiple aspects
- **Customizable Filters**: Filter by restaurant, aspect, sentiment, and confidence

## File Structure
- `src/dashboard.py`: Main Streamlit dashboard
- `src/aspect_extraction.py`: Aspect extraction logic
- `src/aspect_sentiment_analysis.py`: Sentiment analysis logic
- `src/restaurant_analyzer_setup.py`: Setup and configuration
- `src/test_extraction.py`: Test scripts
- `src/yelp_academic_dataset_business.json` & `src/yelp_academic_dataset_review.json`: (Not tracked in GitHub) Large dataset files

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/fangowen/restaurant_aspect_analysis.git
   cd restaurant_aspect_analysis
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt`, install manually:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly wordcloud scikit-learn textblob spacy transformers
   ```
4. **(Optional) Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Dashboard
```bash
streamlit run src/dashboard.py
```

## Notes
- Large dataset files are not tracked in GitHub due to size limits. Add them locally if needed.
- For best performance, install the Watchdog module:
  ```bash
  pip install watchdog
  ```

## License
MIT
