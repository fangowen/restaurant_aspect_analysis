import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Import your analysis modules
from restaurant_analyzer_setup import RestaurantReviewAnalyzer
from aspect_extraction import AspectExtractor
from aspect_sentiment_analysis import AspectSentimentAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="Restaurant Review Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .aspect-score-positive {
        color: #28a745;
        font-weight: bold;
    }
    .aspect-score-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .aspect-score-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class RestaurantDashboard:
    def __init__(self):
        self.sentiment_results = None
        self.aspect_summary = None
        self.restaurant_data = None
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Create sample sentiment results
        np.random.seed(42)
        
        restaurants = ['Bella Vista Italian', 'Golden Dragon Chinese', 'Downtown Burger', 'Le Bernardin', 'Central Cafe']
        aspects = ['food', 'service', 'ambiance', 'price', 'location']
        sentiments = ['positive', 'negative', 'neutral']
        
        sample_data = []
        for i in range(1000):
            restaurant = np.random.choice(restaurants)
            aspect = np.random.choice(aspects)
            sentiment = np.random.choice(sentiments, p=[0.5, 0.3, 0.2])  # More positive reviews
            confidence = np.random.uniform(0.6, 0.95)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.4, 0.2])
            
            # Create realistic sentences based on aspect and sentiment
            sentence_templates = {
                'food': {
                    'positive': ["The food was absolutely delicious!", "Amazing flavors and fresh ingredients!", "Best pasta I've ever had!"],
                    'negative': ["Food was terrible and cold", "Bland and tasteless dishes", "Overcooked and dry"],
                    'neutral': ["Food was decent, nothing special", "Average taste and presentation", "Standard menu options"]
                },
                'service': {
                    'positive': ["Excellent and attentive service!", "Staff was incredibly friendly!", "Quick and professional service"],
                    'negative': ["Terrible service, very rude staff", "Slow and inattentive waiters", "Ignored us the entire night"],
                    'neutral': ["Service was okay", "Average wait time", "Staff was polite but not exceptional"]
                },
                'ambiance': {
                    'positive': ["Beautiful atmosphere and decor!", "Cozy and romantic setting!", "Perfect ambiance for dining"],
                    'negative': ["Loud and uncomfortable environment", "Dirty and outdated interior", "Poor lighting and music"],
                    'neutral': ["Standard restaurant atmosphere", "Nothing special about the decor", "Average dining environment"]
                },
                'price': {
                    'positive': ["Great value for money!", "Very reasonable prices!", "Worth every penny!"],
                    'negative': ["Extremely overpriced!", "Too expensive for the quality", "Not worth the high cost"],
                    'neutral': ["Fair pricing", "Standard costs for the area", "Average price range"]
                },
                'location': {
                    'positive': ["Perfect location, easy to find!", "Great parking and accessibility!", "Convenient downtown location"],
                    'negative': ["Hard to find and no parking", "Terrible location", "Difficult to access"],
                    'neutral': ["Decent location", "Average accessibility", "Standard parking options"]
                }
            }
            
            sentence = np.random.choice(sentence_templates[aspect][sentiment])
            
            sample_data.append({
                'business_id': f"rest_{restaurants.index(restaurant)+1:03d}",
                'restaurant_name': restaurant,
                'sentence': sentence,
                'aspect': aspect,
                'sentiment': sentiment,
                'confidence': confidence,
                'original_rating': rating
            })
        
        self.sentiment_results = pd.DataFrame(sample_data)
        
        # Create restaurant summary
        self.aspect_summary = self.sentiment_results.groupby(['business_id', 'restaurant_name', 'aspect']).agg({
            'sentiment': lambda x: x.value_counts().index[0],
            'confidence': 'mean',
            'original_rating': 'mean'
        }).reset_index()
        
        # Create overall restaurant data
        self.restaurant_data = self.sentiment_results.groupby(['business_id', 'restaurant_name']).agg({
            'original_rating': 'mean',
            'sentiment': lambda x: (x == 'positive').sum() / len(x),  # Positive sentiment ratio
            'confidence': 'mean'
        }).reset_index()
        
        return True
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("🍽️ Restaurant Analyzer")
        
        # Data source selection
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.radio(
            "Choose data source:",
            ["Sample Data", "Upload CSV", "Load from Database"]
        )
        
        if data_source == "Sample Data":
            if st.sidebar.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    self.load_sample_data()
                st.sidebar.success("Sample data loaded!")
        
        elif data_source == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader(
                "Upload sentiment results CSV",
                type=['csv'],
                help="Upload your processed sentiment analysis results"
            )
            
            if uploaded_file is not None:
                self.sentiment_results = pd.read_csv(uploaded_file)
                st.sidebar.success("Data uploaded successfully!")
        
        # Analysis filters
        if self.sentiment_results is not None:
            st.sidebar.subheader("Filters")
            
            # Restaurant filter
            restaurants = ['All'] + list(self.sentiment_results['restaurant_name'].unique())
            selected_restaurant = st.sidebar.selectbox("Select Restaurant:", restaurants)
            
            # Aspect filter
            aspects = ['All'] + list(self.sentiment_results['aspect'].unique())
            selected_aspect = st.sidebar.multiselect("Select Aspects:", aspects, default='All')
            
            # Sentiment filter
            sentiments = ['All'] + list(self.sentiment_results['sentiment'].unique())
            selected_sentiment = st.sidebar.multiselect("Select Sentiments:", sentiments, default='All')
            
            # Confidence threshold
            min_confidence = st.sidebar.slider("Minimum Confidence:", 0.0, 1.0, 0.0, 0.1)
            
            return {
                'restaurant': selected_restaurant,
                'aspects': selected_aspect,
                'sentiments': selected_sentiment,
                'min_confidence': min_confidence
            }
        
        return None
    
    def filter_data(self, filters):
        """Apply filters to the data"""
        if self.sentiment_results is None:
            return None
            
        filtered_data = self.sentiment_results.copy()
        
        # Apply filters
        if filters['restaurant'] != 'All':
            filtered_data = filtered_data[filtered_data['restaurant_name'] == filters['restaurant']]
        
        if 'All' not in filters['aspects']:
            filtered_data = filtered_data[filtered_data['aspect'].isin(filters['aspects'])]
        
        if 'All' not in filters['sentiments']:
            filtered_data = filtered_data[filtered_data['sentiment'].isin(filters['sentiments'])]
        
        filtered_data = filtered_data[filtered_data['confidence'] >= filters['min_confidence']]
        
        return filtered_data
    
    def render_overview_metrics(self, data):
        """Render overview metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Reviews", len(data))
        
        with col2:
            unique_restaurants = data['business_id'].nunique()
            st.metric("Restaurants", unique_restaurants)
        
        with col3:
            avg_rating = data['original_rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}⭐")
        
        with col4:
            positive_pct = (data['sentiment'] == 'positive').mean() * 100
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        
        with col5:
            avg_confidence = data['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    def render_sentiment_overview(self, data):
        """Render sentiment analysis overview"""
        st.subheader("📊 Sentiment Analysis Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = data['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545', 
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Sentiment by aspect
            aspect_sentiment = data.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
            fig_bar = px.bar(
                aspect_sentiment,
                title="Sentiment Distribution by Aspect",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            fig_bar.update_layout(xaxis_title="Aspect", yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def render_aspect_analysis(self, data):
        """Render detailed aspect analysis"""
        st.subheader("🎯 Aspect-Based Analysis")
        
        # Calculate aspect scores
        aspect_scores = data.groupby('aspect').apply(
            lambda x: ((x['sentiment'] == 'positive').sum() - (x['sentiment'] == 'negative').sum()) / len(x)
        ).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Aspect scores chart
            colors = ['#28a745' if score > 0 else '#dc3545' if score < -0.1 else '#6c757d' for score in aspect_scores.values]
            
            fig_aspect = go.Figure(data=[
                go.Bar(x=aspect_scores.index, y=aspect_scores.values, marker_color=colors)
            ])
            fig_aspect.update_layout(
                title="Aspect Sentiment Scores",
                xaxis_title="Aspect",
                yaxis_title="Sentiment Score (-1 to 1)",
                yaxis=dict(range=[-1, 1])
            )
            st.plotly_chart(fig_aspect, use_container_width=True)
        
        with col2:
            # Confidence by aspect
            avg_confidence = data.groupby('aspect')['confidence'].mean().sort_values(ascending=False)
            
            fig_conf = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                title="Average Confidence by Aspect",
                color=avg_confidence.values,
                color_continuous_scale='Blues'
            )
            fig_conf.update_layout(xaxis_title="Aspect", yaxis_title="Confidence")
            st.plotly_chart(fig_conf, use_container_width=True)
    
    def render_restaurant_comparison(self, data):
        """Render restaurant comparison"""
        if data['business_id'].nunique() < 2:
            st.info("Need at least 2 restaurants for comparison")
            return
        
        st.subheader("🏪 Restaurant Comparison")
        
        # Restaurant sentiment scores
        restaurant_scores = data.groupby(['restaurant_name', 'aspect']).apply(
            lambda x: ((x['sentiment'] == 'positive').sum() - (x['sentiment'] == 'negative').sum()) / len(x)
        ).unstack(fill_value=0)
        
        # Heatmap
        fig_heatmap = px.imshow(
            restaurant_scores.values,
            x=restaurant_scores.columns,
            y=restaurant_scores.index,
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="Restaurant Aspect Sentiment Heatmap",
            aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis_title="Aspect",
            yaxis_title="Restaurant",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top restaurants by overall sentiment
        overall_scores = data.groupby('restaurant_name').apply(
            lambda x: ((x['sentiment'] == 'positive').sum() - (x['sentiment'] == 'negative').sum()) / len(x)
        ).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Rated Restaurants (by sentiment):**")
            for restaurant, score in overall_scores.head().items():
                if score > 0:
                    st.markdown(f"✅ **{restaurant}**: {score:.2f}")
                else:
                    st.markdown(f"❌ **{restaurant}**: {score:.2f}")
        
        with col2:
            # Restaurant ratings vs sentiment
            restaurant_data = data.groupby('restaurant_name').agg({
                'original_rating': 'mean',
                'sentiment': lambda x: (x == 'positive').sum() / len(x)
            }).reset_index()
            
            fig_scatter = px.scatter(
                restaurant_data,
                x='original_rating',
                y='sentiment',
                text='restaurant_name',
                title="Rating vs Sentiment Correlation",
                labels={'original_rating': 'Average Rating', 'sentiment': 'Positive Sentiment Ratio'}
            )
            fig_scatter.update_traces(textposition="top center")
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def render_detailed_reviews(self, data):
        """Render detailed review analysis"""
        st.subheader("📝 Detailed Review Analysis")
        
        # Sample reviews by sentiment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Most Positive Reviews:**")
            positive_reviews = data[
                (data['sentiment'] == 'positive') & (data['confidence'] > 0.8)
            ].nlargest(5, 'confidence')
            
            for _, review in positive_reviews.iterrows():
                st.success(f"**{review['aspect'].title()}**: {review['sentence']} (Confidence: {review['confidence']:.2f})")
        
        with col2:
            st.write("**Most Negative Reviews:**")
            negative_reviews = data[
                (data['sentiment'] == 'negative') & (data['confidence'] > 0.8)
            ].nlargest(5, 'confidence')
            
            for _, review in negative_reviews.iterrows():
                st.error(f"**{review['aspect'].title()}**: {review['sentence']} (Confidence: {review['confidence']:.2f})")
        
        with col3:
            st.write("**Neutral/Mixed Reviews:**")
            neutral_reviews = data[
                data['sentiment'] == 'neutral'
            ].sample(min(5, len(data[data['sentiment'] == 'neutral'])))
            
            for _, review in neutral_reviews.iterrows():
                st.info(f"**{review['aspect'].title()}**: {review['sentence']} (Confidence: {review['confidence']:.2f})")
    
    def render_word_clouds(self, data):
        """Render word clouds for different sentiments"""
        st.subheader("☁️ Word Clouds")
        
        col1, col2, col3 = st.columns(3)
        
        sentiments = ['positive', 'negative', 'neutral']
        colors = ['Greens', 'Reds', 'Blues']
        
        for col, sentiment, colormap in zip([col1, col2, col3], sentiments, colors):
            with col:
                st.write(f"**{sentiment.title()} Reviews**")
                
                # Get text for this sentiment
                sentiment_text = ' '.join(
                    data[data['sentiment'] == sentiment]['sentence'].tolist()
                )
                
                if sentiment_text.strip():
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=300, 
                        height=200, 
                        background_color='white',
                        colormap=colormap,
                        max_words=50
                    ).generate(sentiment_text)
                    
                    # Display using matplotlib
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.write("No data for word cloud")
    
    def render_analytics_insights(self, data):
        """Render analytics and insights"""
        st.subheader("💡 Key Insights")
        
        insights = []
        
        # Overall sentiment
        positive_pct = (data['sentiment'] == 'positive').mean() * 100
        if positive_pct > 60:
            insights.append(f"✅ **Strong positive sentiment** - {positive_pct:.1f}% of reviews are positive")
        elif positive_pct < 40:
            insights.append(f"⚠️ **Areas for improvement** - Only {positive_pct:.1f}% of reviews are positive")
        
        # Best and worst aspects
        aspect_scores = data.groupby('aspect').apply(
            lambda x: ((x['sentiment'] == 'positive').sum() - (x['sentiment'] == 'negative').sum()) / len(x)
        )
        
        best_aspect = aspect_scores.idxmax()
        worst_aspect = aspect_scores.idxmin()
        
        insights.append(f"🏆 **Strongest aspect**: {best_aspect.title()} (Score: {aspect_scores[best_aspect]:.2f})")
        insights.append(f"🔧 **Needs improvement**: {worst_aspect.title()} (Score: {aspect_scores[worst_aspect]:.2f})")
        
        # Confidence analysis
        high_conf_pct = (data['confidence'] > 0.8).mean() * 100
        insights.append(f"🎯 **Prediction confidence**: {high_conf_pct:.1f}% of predictions have high confidence (>0.8)")
        
        # Rating vs sentiment correlation
        if 'original_rating' in data.columns:
            rating_sentiment = data.groupby('original_rating').apply(
                lambda x: (x['sentiment'] == 'positive').mean()
            )
            
            if rating_sentiment.corr(rating_sentiment.index) > 0.7:
                insights.append("✅ **Good alignment** between star ratings and sentiment analysis")
            else:
                insights.append("⚠️ **Misalignment detected** between ratings and sentiment - may indicate review authenticity issues")
        
        for insight in insights:
            st.markdown(insight)
    
    def run(self):
        """Run the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🍽️ Restaurant Review Analyzer</h1>', unsafe_allow_html=True)
        
        # Sidebar
        filters = self.render_sidebar()
        
        if self.sentiment_results is None:
            st.info("👆 Please load data using the sidebar to get started!")
            
            # Show sample screenshots or demo
            st.subheader("✨ What you'll get:")
            st.write("""
            - **Comprehensive sentiment analysis** across multiple aspects (food, service, ambiance, price, location)
            - **Interactive visualizations** with filtering and comparison tools
            - **Restaurant comparison** and ranking capabilities
            - **Detailed insights** and actionable recommendations
            - **Word clouds** and review exploration tools
            """)
            
            return
        
        # Apply filters
        if filters:
            filtered_data = self.filter_data(filters)
            if filtered_data.empty:
                st.warning("No data matches the selected filters. Please adjust your selection.")
                return
        else:
            filtered_data = self.sentiment_results
        
        # Main dashboard content
        self.render_overview_metrics(filtered_data)
        
        st.divider()
        
        # Sentiment analysis
        self.render_sentiment_overview(filtered_data)
        
        st.divider()
        
        # Aspect analysis
        self.render_aspect_analysis(filtered_data)
        
        st.divider()
        
        # Restaurant comparison
        self.render_restaurant_comparison(filtered_data)
        
        st.divider()
        
        # Detailed reviews
        self.render_detailed_reviews(filtered_data)
        
        st.divider()
        
        # Word clouds
        try:
            self.render_word_clouds(filtered_data)
            st.divider()
        except Exception as e:
            st.warning(f"Could not generate word clouds: {e}")
        
        # Insights
        self.render_analytics_insights(filtered_data)

# Main app
def main():
    """Main function to run the Streamlit app"""
    dashboard = RestaurantDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
