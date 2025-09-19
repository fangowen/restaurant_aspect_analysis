import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class RestaurantReviewAnalyzer:
    def __init__(self):
        self.business_data = None
        self.review_data = None
        self.restaurant_reviews = None
        
    def load_yelp_data(self, business_file_path, review_file_path):
        """
        Load Yelp Academic Dataset files
        Download from: https://www.yelp.com/dataset
        """
        print("Loading business data...")
        # Load business data (JSON lines format)
        businesses = []
        with open(business_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                businesses.append(json.loads(line))
        
        self.business_data = pd.DataFrame(businesses)
        print(f"Loaded {len(self.business_data)} businesses")
        
        # Filter for restaurants only
        restaurant_mask = self.business_data['categories'].str.contains('Restaurant', na=False, case=False)
        restaurants = self.business_data[restaurant_mask].copy()
        restaurant_ids = set(restaurants['business_id'])
        
        print(f"Found {len(restaurants)} restaurants")
        
        print("Loading review data...")
        # Load reviews (this might take a while - the file is large)
        reviews = []
        count = 0
        with open(review_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line)
                # Only keep reviews for restaurants
                if review['business_id'] in restaurant_ids:
                    reviews.append(review)
                    count += 1
                    if count % 10000 == 0:
                        print(f"Loaded {count} restaurant reviews...")
                # Limit for testing - remove this line for full dataset
                if count >= 50000:  # Start with 50k reviews for development
                    break
        
        self.review_data = pd.DataFrame(reviews)
        print(f"Loaded {len(self.review_data)} restaurant reviews")
        
        # Merge restaurant info with reviews
        self.restaurant_reviews = self.review_data.merge(
            restaurants[['business_id', 'name', 'categories', 'city', 'state']], 
            on='business_id', 
            how='left'
        )
        
        return self.restaurant_reviews
    
    def load_sample_data(self):
        """
        Create sample data for development/testing
        Use this if you don't have the Yelp dataset yet
        """
        print("Creating sample restaurant review data...")
        
        sample_reviews = [
            {
                'business_id': 'rest_001',
                'text': 'Amazing food quality! The pasta was perfectly cooked and the sauce was incredible. Service was a bit slow though, waited 20 minutes for our appetizer. The atmosphere was cozy and romantic, perfect for date night. A bit pricey but worth it.',
                'stars': 4,
                'name': 'Bella Vista Italian',
                'categories': 'Italian, Restaurants',
                'city': 'New York'
            },
            {
                'business_id': 'rest_002', 
                'text': 'Terrible experience. Food was cold and tasteless. The chicken was dry and seemed reheated. Our server was rude and ignored us most of the night. The restaurant was dirty and outdated. Way overpriced for what you get. Never coming back!',
                'stars': 1,
                'name': 'Golden Dragon Chinese',
                'categories': 'Chinese, Restaurants',
                'city': 'Los Angeles'
            },
            {
                'business_id': 'rest_003',
                'text': 'Great casual spot! The burgers are juicy and flavorful. Fast and friendly service. The place gets pretty loud during peak hours but has a fun vibe. Very reasonable prices for the portion sizes. Perfect for families.',
                'stars': 4,
                'name': 'Downtown Burger Co',
                'categories': 'Burgers, American, Restaurants', 
                'city': 'Chicago'
            },
            {
                'business_id': 'rest_004',
                'text': 'Outstanding fine dining experience. Every dish was a work of art and tasted incredible. The service was impeccable - attentive without being intrusive. Beautiful upscale ambiance with dim lighting and elegant decor. Expensive but justified.',
                'stars': 5,
                'name': 'Le Bernardin NYC',
                'categories': 'French, Fine Dining, Restaurants',
                'city': 'New York'
            }
        ]
        
        # Duplicate and modify to create more sample data
        extended_reviews = []
        for i, review in enumerate(sample_reviews * 25):  # 100 sample reviews
            modified_review = review.copy()
            modified_review['review_id'] = f"review_{i:03d}"
            modified_review['user_id'] = f"user_{i % 20:03d}"  # 20 different users
            extended_reviews.append(modified_review)
        
        self.restaurant_reviews = pd.DataFrame(extended_reviews)
        print(f"Created {len(self.restaurant_reviews)} sample reviews")
        return self.restaurant_reviews
    
    def basic_eda(self):
        """
        Perform basic exploratory data analysis
        """
        if self.restaurant_reviews is None:
            print("No data loaded. Please load data first.")
            return
            
        print("=== BASIC STATISTICS ===")
        print(f"Total reviews: {len(self.restaurant_reviews)}")
        print(f"Unique restaurants: {self.restaurant_reviews['business_id'].nunique()}")
        print(f"Average rating: {self.restaurant_reviews['stars'].mean():.2f}")
        print(f"Rating distribution:")
        print(self.restaurant_reviews['stars'].value_counts().sort_index())
        
        print("\n=== REVIEW LENGTH ANALYSIS ===")
        self.restaurant_reviews['text_length'] = self.restaurant_reviews['text'].str.len()
        self.restaurant_reviews['word_count'] = self.restaurant_reviews['text'].str.split().str.len()
        
        print(f"Average review length: {self.restaurant_reviews['text_length'].mean():.0f} characters")
        print(f"Average word count: {self.restaurant_reviews['word_count'].mean():.0f} words")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rating distribution
        self.restaurant_reviews['stars'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Rating Distribution')
        axes[0,0].set_xlabel('Stars')
        axes[0,0].set_ylabel('Number of Reviews')
        
        # Review length distribution
        axes[0,1].hist(self.restaurant_reviews['word_count'], bins=50, edgecolor='black', alpha=0.7)
        axes[0,1].set_title('Review Word Count Distribution')
        axes[0,1].set_xlabel('Number of Words')
        axes[0,1].set_ylabel('Frequency')
        
        # Top cities
        if 'city' in self.restaurant_reviews.columns:
            top_cities = self.restaurant_reviews['city'].value_counts().head(10)
            top_cities.plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Top 10 Cities by Review Count')
            axes[1,0].set_xlabel('City')
            axes[1,0].set_ylabel('Number of Reviews')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Rating vs word count
        axes[1,1].scatter(self.restaurant_reviews['stars'], self.restaurant_reviews['word_count'], alpha=0.5)
        axes[1,1].set_title('Rating vs Review Length')
        axes[1,1].set_xlabel('Stars')
        axes[1,1].set_ylabel('Word Count')
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_text(self, text):
        """
        Basic text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep punctuation that might be important for sentiment
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_processed_dataset(self):
        """
        Create a cleaned dataset ready for ML
        """
        if self.restaurant_reviews is None:
            print("No data loaded. Please load data first.")
            return None
            
        print("Processing text data...")
        
        # Create processed dataset
        processed_df = self.restaurant_reviews.copy()
        
        # Preprocess text
        processed_df['processed_text'] = processed_df['text'].apply(self.preprocess_text)
        
        # Create binary sentiment labels (for initial testing)
        processed_df['sentiment'] = processed_df['stars'].apply(
            lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
        )
        
        processed_df['text_length'] = processed_df['text'].str.len()
        processed_df['word_count'] = processed_df['text'].str.split().str.len()
        processed_df['exclamation_count'] = processed_df['text'].str.count(r'!')
        processed_df['question_count'] = processed_df['text'].str.count(r'\?')
        
        print(f"Processed {len(processed_df)} reviews")
        print(f"Sentiment distribution:")
        print(processed_df['sentiment'].value_counts())
        
        return processed_df

# Usage example:
if __name__ == "__main__":
    analyzer = RestaurantReviewAnalyzer()
    
    # Load real Yelp data from workspace folder
    try:
        reviews_df = analyzer.load_yelp_data(
            'yelp_academic_dataset_business.json',
            'yelp_academic_dataset_review.json'
        )
        print("✅ Successfully loaded Yelp dataset!")
    except FileNotFoundError as e:
        print(f"❌ Yelp files not found: {e}")
        print("Falling back to sample data...")
        reviews_df = analyzer.load_sample_data()
    
    # Run basic analysis
    analyzer.basic_eda()
    
    # Create processed dataset
    processed_reviews = analyzer.create_processed_dataset()
    