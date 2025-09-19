import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download spacy model if needed: python -m spacy download en_core_web_sm

class AspectExtractor:
    def __init__(self):
        self.aspect_keywords = {
            'food': [
                'food', 'dish', 'meal', 'taste', 'flavor', 'delicious', 'tasty', 'bland', 
                'spicy', 'sweet', 'sour', 'fresh', 'stale', 'cooked', 'raw', 'hot', 'cold',
                'appetizer', 'entree', 'dessert', 'soup', 'salad', 'pasta', 'pizza', 'burger',
                'chicken', 'beef', 'fish', 'seafood', 'vegetarian', 'vegan', 'portion', 'size',
                'recipe', 'ingredient', 'sauce', 'seasoning', 'tender', 'crispy', 'juicy'
            ],
            'service': [
                'service', 'server', 'waiter', 'waitress', 'staff', 'employee', 'manager',
                'friendly', 'rude', 'polite', 'attentive', 'helpful', 'slow', 'fast', 'quick',
                'wait', 'waiting', 'order', 'serve', 'served', 'brought', 'took', 'table',
                'reservation', 'seated', 'greeted', 'smiled', 'professional', 'knowledgeable'
            ],
            'ambiance': [
                'atmosphere', 'ambiance', 'ambience', 'vibe', 'mood', 'setting', 'environment',
                'decor', 'decoration', 'interior', 'lighting', 'music', 'loud', 'quiet', 'noisy',
                'cozy', 'romantic', 'casual', 'formal', 'elegant', 'modern', 'traditional',
                'clean', 'dirty', 'crowded', 'spacious', 'comfortable', 'seating', 'tables'
            ],
            'price': [
                'price', 'cost', 'expensive', 'cheap', 'affordable', 'overpriced', 'reasonable',
                'value', 'money', 'worth', 'budget', 'deal', 'bargain', 'pricey', 'costly',
                'dollar', '$', 'bill', 'check', 'pay', 'paid', 'charge', 'fee'
            ],
            'location': [
                'location', 'parking', 'drive', 'walk', 'accessible', 'convenient', 'far',
                'close', 'nearby', 'downtown', 'area', 'neighborhood', 'street', 'address',
                'easy to find', 'hard to find', 'directions', 'GPS'
            ]
        }
        
        self.nlp = None
        self.aspect_classifier = None
        self.tfidf_vectorizer = None
        
    def load_spacy_model(self):
        """Load spaCy model for NLP processing"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            return False
        return True
    
    def extract_sentences(self, text):
        """Split text into sentences using spaCy"""
        if self.nlp is None:
            if not self.load_spacy_model():
                # Fallback to simple sentence splitting
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
        
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    
    def keyword_based_aspect_detection(self, sentence):
        """
        Detect aspects in a sentence using keyword matching
        Returns list of aspects found in the sentence
        """
        sentence_lower = sentence.lower()
        detected_aspects = []
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    detected_aspects.append(aspect)
                    break  # Only count each aspect once per sentence
                    
        return detected_aspects if detected_aspects else ['general']
    
    def create_aspect_dataset(self, reviews_df):
        """
        Create a dataset where each row is a sentence with its detected aspects
        """
        print("Creating aspect-labeled dataset...")
        
        sentences_data = []
        
        for idx, row in reviews_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx} reviews...")
                
            sentences = self.extract_sentences(row['text'])
            
            for sentence in sentences:
                aspects = self.keyword_based_aspect_detection(sentence)
                
                for aspect in aspects:
                    sentences_data.append({
                        'sentence': sentence,
                        'aspect': aspect,
                        'original_rating': row['stars'],
                        'business_id': row['business_id']
                    })
        
        sentences_df = pd.DataFrame(sentences_data)
        print(f"Created {len(sentences_df)} aspect-labeled sentences")
        
        return sentences_df
    
    def analyze_aspect_distribution(self, sentences_df):
        """Analyze the distribution of detected aspects"""
        print("\n=== ASPECT DISTRIBUTION ANALYSIS ===")
        
        aspect_counts = sentences_df['aspect'].value_counts()
        print("Aspect frequencies:")
        for aspect, count in aspect_counts.items():
            percentage = (count / len(sentences_df)) * 100
            print(f"{aspect}: {count} ({percentage:.1f}%)")
        
        # Visualize aspect distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        aspect_counts.plot(kind='bar')
        plt.title('Aspect Distribution')
        plt.xlabel('Aspect')
        plt.ylabel('Number of Sentences')
        plt.xticks(rotation=45)
        
        # Aspect distribution by rating
        plt.subplot(2, 2, 2)
        aspect_rating = sentences_df.groupby(['aspect', 'original_rating']).size().unstack(fill_value=0)
        aspect_rating.plot(kind='bar', stacked=True)
        plt.title('Aspects by Rating')
        plt.xlabel('Aspect')
        plt.ylabel('Number of Sentences')
        plt.xticks(rotation=45)
        plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Average rating by aspect
        plt.subplot(2, 2, 3)
        avg_rating_by_aspect = sentences_df.groupby('aspect')['original_rating'].mean().sort_values(ascending=False)
        avg_rating_by_aspect.plot(kind='bar')
        plt.title('Average Rating by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        
        # Sample sentences for each aspect
        plt.subplot(2, 2, 4)
        plt.axis('off')
        sample_text = "Sample sentences by aspect:\n\n"
        for aspect in ['food', 'service', 'ambiance', 'price']:
            if aspect in sentences_df['aspect'].values:
                sample = sentences_df[sentences_df['aspect'] == aspect]['sentence'].iloc[0]
                sample_text += f"{aspect.upper()}: {sample[:60]}...\n\n"
        
        plt.text(0.1, 0.9, sample_text, transform=plt.gca().transAxes, 
                fontsize=8, verticalalignment='top', wrap=True)
        
        plt.tight_layout()
        plt.show()
        
        return aspect_counts
    
    def train_aspect_classifier(self, sentences_df):
        """
        Train a machine learning classifier for aspect detection
        This can improve upon keyword-based detection
        """
        print("\nTraining ML-based aspect classifier...")
        
        # Prepare data - remove 'general' aspect for cleaner training
        ml_data = sentences_df[sentences_df['aspect'] != 'general'].copy()
        
        if len(ml_data) < 100:
            print("Not enough labeled data for ML training. Using keyword-based approach only.")
            return None
        
        # Features: TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        X = self.tfidf_vectorizer.fit_transform(ml_data['sentence'])
        y = ml_data['aspect']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.aspect_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.aspect_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.aspect_classifier.predict(X_test)
        
        print("\nClassifier Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        importances = self.aspect_classifier.feature_importances_
        
        # Get top features for each aspect
        print("\nTop features for each aspect:")
        for aspect in self.aspect_classifier.classes_:
            aspect_idx = list(self.aspect_classifier.classes_).index(aspect)
            
            # Get feature importances for this aspect (using one tree as example)
            tree_importances = []
            for tree in self.aspect_classifier.estimators_[:10]:  # Use first 10 trees
                tree_importances.append(tree.feature_importances_)
            
            avg_importance = np.mean(tree_importances, axis=0)
            top_features_idx = np.argsort(avg_importance)[-10:][::-1]
            
            top_features = [feature_names[i] for i in top_features_idx]
            print(f"{aspect}: {', '.join(top_features[:5])}")
        
        return self.aspect_classifier
    
    def predict_aspects(self, text):
        """
        Predict aspects for a given text using both keyword and ML approaches
        """
        sentences = self.extract_sentences(text)
        results = []
        
        for sentence in sentences:
            # Keyword-based prediction
            keyword_aspects = self.keyword_based_aspect_detection(sentence)
            
            # ML-based prediction (if model is trained)
            ml_aspects = []
            if self.aspect_classifier and self.tfidf_vectorizer:
                try:
                    sentence_tfidf = self.tfidf_vectorizer.transform([sentence])
                    ml_prediction = self.aspect_classifier.predict(sentence_tfidf)[0]
                    ml_aspects = [ml_prediction]
                except:
                    ml_aspects = []
            
            # Combine predictions (prioritize ML if available)
            final_aspects = ml_aspects if ml_aspects else keyword_aspects
            
            results.append({
                'sentence': sentence,
                'aspects': final_aspects,
                'keyword_aspects': keyword_aspects,
                'ml_aspects': ml_aspects
            })
        
        return results

# Example usage and testing
def test_aspect_extraction(reviews_df):
    """Test the aspect extraction pipeline"""
    print("=== TESTING ASPECT EXTRACTION ===")
    
    extractor = AspectExtractor()
    
    # Create aspect dataset
    sentences_df = extractor.create_aspect_dataset(reviews_df.head(500))  # Start with 500 reviews
    
    # Analyze distribution
    aspect_counts = extractor.analyze_aspect_distribution(sentences_df)
    
    # Train ML classifier
    classifier = extractor.train_aspect_classifier(sentences_df)
    
    # Test on a sample review
    test_review = """
    The food was absolutely amazing! The pasta was perfectly cooked and the sauce was incredible. 
    However, the service was quite slow - we waited 25 minutes for our appetizer. 
    The atmosphere was cozy and romantic, perfect for a date night. 
    A bit expensive but definitely worth the price for the quality.
    """
    
    print(f"\n=== TESTING ON SAMPLE REVIEW ===")
    print(f"Review: {test_review}")
    
    results = extractor.predict_aspects(test_review)
    
    print("\nExtracted aspects:")
    for result in results:
        print(f"Sentence: {result['sentence']}")
        print(f"Detected aspects: {result['aspects']}")
        print("---")
    
    return extractor, sentences_df

if __name__ == "__main__":
    # This would typically be called after loading data from the previous phase
    print("Aspect Extraction Module Ready!")
    print("Call test_aspect_extraction(your_reviews_df) to test the pipeline")
