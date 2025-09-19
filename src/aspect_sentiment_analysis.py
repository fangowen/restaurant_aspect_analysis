import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from collections import defaultdict, Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class AspectSentimentAnalyzer:
    def __init__(self):
        self.sentiment_models = {}
        self.tfidf_vectorizers = {}
        self.label_encoders = {}
        self.bert_sentiment_pipeline = None
        
        # Sentiment lexicons for aspects
        self.aspect_sentiment_words = {
            'food': {
                'positive': ['delicious', 'tasty', 'amazing', 'excellent', 'fresh', 'flavorful', 'perfect', 
                           'wonderful', 'outstanding', 'incredible', 'juicy', 'tender', 'crispy', 'savory'],
                'negative': ['terrible', 'awful', 'disgusting', 'bland', 'tasteless', 'stale', 'overcooked', 
                           'undercooked', 'dry', 'soggy', 'burnt', 'cold', 'horrible', 'inedible']
            },
            'service': {
                'positive': ['excellent', 'friendly', 'attentive', 'helpful', 'professional', 'quick', 'fast',
                           'polite', 'courteous', 'knowledgeable', 'efficient', 'amazing', 'wonderful'],
                'negative': ['terrible', 'rude', 'slow', 'awful', 'unprofessional', 'ignored', 'dismissive',
                           'arrogant', 'incompetent', 'lazy', 'inattentive', 'horrible']
            },
            'ambiance': {
                'positive': ['cozy', 'romantic', 'beautiful', 'elegant', 'comfortable', 'pleasant', 'lovely',
                           'charming', 'inviting', 'relaxing', 'atmospheric', 'stunning'],
                'negative': ['loud', 'noisy', 'dirty', 'uncomfortable', 'cramped', 'dingy', 'boring',
                           'tacky', 'outdated', 'chaotic', 'unpleasant']
            },
            'price': {
                'positive': ['reasonable', 'affordable', 'worth', 'value', 'bargain', 'cheap', 'fair',
                           'good deal', 'inexpensive', 'budget-friendly'],
                'negative': ['expensive', 'overpriced', 'costly', 'pricey', 'rip-off', 'outrageous',
                           'unreasonable', 'not worth', 'too much']
            }
        }
    
    def initialize_bert_model(self):
        """Initialize BERT model for sentiment analysis"""
    
    def textblob_sentiment(self, text):
        """Get sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def lexicon_based_sentiment(self, text, aspect):
        """Get sentiment using aspect-specific lexicons"""
        text_lower = text.lower()
        
        if aspect not in self.aspect_sentiment_words:
            return self.textblob_sentiment(text)
        
        positive_words = self.aspect_sentiment_words[aspect]['positive']
        negative_words = self.aspect_sentiment_words[aspect]['negative']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            confidence = (pos_count - neg_count) / (pos_count + neg_count + 1)
            return 'positive', confidence
        elif neg_count > pos_count:
            confidence = (neg_count - pos_count) / (pos_count + neg_count + 1)
            return 'negative', confidence
        else:
            # Fall back to TextBlob for neutral cases
            return self.textblob_sentiment(text)
    
    def bert_sentiment(self, text):
        """Get sentiment using BERT model"""
        if self.bert_sentiment_pipeline is None:
            return self.textblob_sentiment(text)
        
        try:
            results = self.bert_sentiment_pipeline(text)[0]
            
            # Convert to our format
            sentiment_scores = {result['label'].lower(): result['score'] for result in results}
            
            # Map labels (RoBERTa model uses LABEL_0, LABEL_1, LABEL_2)
            if 'label_0' in sentiment_scores:  # Negative
                negative_score = sentiment_scores['label_0']
            elif 'negative' in sentiment_scores:
                negative_score = sentiment_scores['negative']
            else:
                negative_score = 0
            
            if 'label_2' in sentiment_scores:  # Positive  
                positive_score = sentiment_scores['label_2']
            elif 'positive' in sentiment_scores:
                positive_score = sentiment_scores['positive']
            else:
                positive_score = 0
            
            if positive_score > negative_score and positive_score > 0.6:
                return 'positive', positive_score
            elif negative_score > positive_score and negative_score > 0.6:
                return 'negative', negative_score
            else:
                return 'neutral', max(positive_score, negative_score)
                
        except Exception as e:
            print(f"BERT prediction failed: {e}")
            return self.textblob_sentiment(text)
    
    def analyze_sentence_sentiment(self, sentence, aspect, method='combined'):
        """
        Analyze sentiment of a sentence for a specific aspect
        Methods: 'textblob', 'lexicon', 'bert', 'combined'
        """
        results = {}
        
        # TextBlob sentiment
        textblob_sent, textblob_conf = self.textblob_sentiment(sentence)
        results['textblob'] = {'sentiment': textblob_sent, 'confidence': abs(textblob_conf)}
        
        # Lexicon-based sentiment
        lexicon_sent, lexicon_conf = self.lexicon_based_sentiment(sentence, aspect)
        results['lexicon'] = {'sentiment': lexicon_sent, 'confidence': abs(lexicon_conf)}
        
        bert_sent, bert_conf = self.bert_sentiment(sentence)
        results['bert'] = {'sentiment': bert_sent, 'confidence': bert_conf}
        
        # Combined approach
        if method == 'combined':
            # Weight the different methods
            sentiment_votes = []
            confidence_weights = []
            
            # Add TextBlob vote
            sentiment_votes.append(textblob_sent)
            confidence_weights.append(abs(textblob_conf) * 0.3)  # Lower weight
            
            # Add Lexicon vote (higher weight for domain-specific)
            sentiment_votes.append(lexicon_sent)
            confidence_weights.append(lexicon_conf * 0.5)  # Higher weight
            
            
            if 'bert' in results:
                sentiment_votes.append(bert_sent)
                confidence_weights.append(bert_conf * 0.4)  # Medium weight
            
            # Weighted voting
            sentiment_counts = Counter()
            for sent, weight in zip(sentiment_votes, confidence_weights):
                sentiment_counts[sent] += weight
            
            final_sentiment = sentiment_counts.most_common(1)[0][0]
            final_confidence = sentiment_counts[final_sentiment] / sum(confidence_weights)
            
            results['combined'] = {'sentiment': final_sentiment, 'confidence': final_confidence}
            
            return final_sentiment, final_confidence, results
        else:
            chosen_result = results[method]
            return chosen_result['sentiment'], chosen_result['confidence'], results
    
    def analyze_aspect_sentiments(self, sentences_df, method='combined'):
        """
        Analyze sentiment for all aspect-sentence pairs
        """
        print(f"Analyzing sentiment for {len(sentences_df)} sentences using {method} method...")
        
        results = []
        
        for idx, row in sentences_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx} sentences...")
            
            sentence = row['sentence']
            aspect = row['aspect']
            
            # Skip general aspects for now
            if aspect == 'general':
                continue
            
            sentiment, confidence, method_details = self.analyze_sentence_sentiment(
                sentence, aspect, method
            )
            
            results.append({
                'sentence': sentence,
                'aspect': aspect,
                'sentiment': sentiment,
                'confidence': confidence,
                'original_rating': row['original_rating'],
                'business_id': row['business_id'],
                'method_details': method_details
            })
        
        results_df = pd.DataFrame(results)
        print(f"Completed sentiment analysis for {len(results_df)} sentences")
        
        return results_df
    
    def create_restaurant_aspect_summary(self, sentiment_results_df):
        """
        Create restaurant-level aspect sentiment summaries
        """
        print("Creating restaurant-level aspect summaries...")
        
        # Group by restaurant and aspect
        restaurant_aspects = sentiment_results_df.groupby(['business_id', 'aspect']).agg({
            'sentiment': lambda x: x.value_counts().index[0],  # Most common sentiment
            'confidence': 'mean',  # Average confidence
            'original_rating': 'first'  # Original rating (should be same for all)
        }).reset_index()
        
        # Calculate sentiment scores (positive: +1, neutral: 0, negative: -1)
        restaurant_aspects['sentiment_score'] = restaurant_aspects['sentiment'].map({
            'positive': 1,
            'neutral': 0, 
            'negative': -1
        })
        
        # Pivot to get aspects as columns
        aspect_summary = restaurant_aspects.pivot(
            index='business_id', 
            columns='aspect', 
            values='sentiment_score'
        ).fillna(0)
        
        # Add overall metrics
        aspect_summary['overall_aspect_score'] = aspect_summary.mean(axis=1)
        aspect_summary['original_rating'] = restaurant_aspects.groupby('business_id')['original_rating'].first()
        
        return aspect_summary, restaurant_aspects
    
    def visualize_sentiment_analysis(self, sentiment_results_df, restaurant_aspects_df):
        """
        Create comprehensive visualizations for sentiment analysis
        """
        plt.figure(figsize=(20, 15))
        
        # 1. Overall sentiment distribution by aspect
        plt.subplot(3, 4, 1)
        sentiment_by_aspect = sentiment_results_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        sentiment_by_aspect.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Sentiment Distribution by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Number of Sentences')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
        
        # 2. Average confidence by aspect
        plt.subplot(3, 4, 2)
        avg_confidence = sentiment_results_df.groupby('aspect')['confidence'].mean()
        avg_confidence.plot(kind='bar')
        plt.title('Average Confidence by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Confidence')
        plt.xticks(rotation=45)
        
        # 3. Sentiment vs Original Rating
        plt.subplot(3, 4, 3)
        sentiment_rating = sentiment_results_df.groupby(['original_rating', 'sentiment']).size().unstack(fill_value=0)
        sentiment_rating_pct = sentiment_rating.div(sentiment_rating.sum(axis=1), axis=0) * 100
        sentiment_rating_pct.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Sentiment Distribution by Original Rating')
        plt.xlabel('Original Rating')
        plt.ylabel('Percentage')
        plt.legend(title='Sentiment')
        
        # 4. Aspect sentiment correlation heatmap
        plt.subplot(3, 4, 4)
        aspect_pivot = sentiment_results_df.pivot_table(
            index='business_id', 
            columns='aspect', 
            values='sentiment', 
            aggfunc=lambda x: (x == 'positive').sum() - (x == 'negative').sum()
        ).fillna(0)
        
        if len(aspect_pivot.columns) > 1:
            correlation_matrix = aspect_pivot.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=plt.gca())
            plt.title('Aspect Sentiment Correlation')
        
        # 5-8. Individual aspect sentiment distributions
        aspects = sentiment_results_df['aspect'].unique()
        for i, aspect in enumerate(aspects[:4], 5):
            plt.subplot(3, 4, i)
            aspect_data = sentiment_results_df[sentiment_results_df['aspect'] == aspect]
            aspect_data['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title(f'{aspect.title()} Sentiment')
            plt.ylabel('')
        
        # 9. Confidence distribution
        plt.subplot(3, 4, 9)
        sentiment_results_df['confidence'].hist(bins=50, edgecolor='black', alpha=0.7)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        
        # 10. Sentiment by aspect (percentage)
        plt.subplot(3, 4, 10)
        sentiment_pct = sentiment_results_df.groupby('aspect')['sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
        sentiment_pct.plot(kind='bar', ax=plt.gca())
        plt.title('Sentiment Percentage by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
        
        # 11. Top positive and negative sentences
        plt.subplot(3, 4, 11)
        plt.axis('off')
        top_positive = sentiment_results_df[
            (sentiment_results_df['sentiment'] == 'positive') & 
            (sentiment_results_df['confidence'] > 0.8)
        ].nlargest(3, 'confidence')
        
        top_negative = sentiment_results_df[
            (sentiment_results_df['sentiment'] == 'negative') & 
            (sentiment_results_df['confidence'] > 0.8)
        ].nlargest(3, 'confidence')
        
        sample_text = "Top Confident Predictions:\n\n"
        sample_text += "POSITIVE:\n"
        for _, row in top_positive.iterrows():
            sample_text += f"• {row['sentence'][:50]}... ({row['aspect']})\n"
        
        sample_text += "\nNEGATIVE:\n"
        for _, row in top_negative.iterrows():
            sample_text += f"• {row['sentence'][:50]}... ({row['aspect']})\n"
        
        plt.text(0.05, 0.95, sample_text, transform=plt.gca().transAxes, 
                fontsize=8, verticalalignment='top', wrap=True)
        
        # 12. Method comparison (if available)
        plt.subplot(3, 4, 12)
        if 'method_details' in sentiment_results_df.columns:
            # Compare different methods
            method_agreement = []
            for _, row in sentiment_results_df.iterrows():
                methods = row['method_details']
                sentiments = [methods[m]['sentiment'] for m in methods if m != 'combined']
                if len(set(sentiments)) == 1:
                    method_agreement.append('All Agree')
                else:
                    method_agreement.append('Disagree')
            
            agreement_counts = pd.Series(method_agreement).value_counts()
            agreement_counts.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Method Agreement')
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, 'Method comparison\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def generate_insights(self, sentiment_results_df, restaurant_aspects_df):
        """
        Generate textual insights from the sentiment analysis
        """
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS INSIGHTS")
        print("="*50)
        
        # Overall statistics
        total_sentences = len(sentiment_results_df)
        aspects_analyzed = sentiment_results_df['aspect'].nunique()
        restaurants_analyzed = sentiment_results_df['business_id'].nunique()
        
        print(f"📊 Analyzed {total_sentences} sentences across {aspects_analyzed} aspects for {restaurants_analyzed} restaurants")
        
        # Sentiment distribution
        overall_sentiment = sentiment_results_df['sentiment'].value_counts(normalize=True) * 100
        print(f"\n📈 Overall Sentiment Distribution:")
        for sentiment, pct in overall_sentiment.items():
            print(f"   {sentiment.title()}: {pct:.1f}%")
        
        # Most positive and negative aspects
        aspect_scores = sentiment_results_df.groupby('aspect').apply(
            lambda x: ((x['sentiment'] == 'positive').sum() - (x['sentiment'] == 'negative').sum()) / len(x)
        ).sort_values(ascending=False)
        
        print(f"\n🎯 Most Positive Aspects:")
        for aspect, score in aspect_scores.head(3).items():
            print(f"   {aspect.title()}: {score:.2f}")
        
        print(f"\n⚠️ Most Negative Aspects:")
        for aspect, score in aspect_scores.tail(3).items():
            print(f"   {aspect.title()}: {score:.2f}")
        
        # Confidence analysis
        avg_confidence = sentiment_results_df['confidence'].mean()
        high_conf_pct = (sentiment_results_df['confidence'] > 0.7).sum() / total_sentences * 100
        
        print(f"\n🎯 Confidence Metrics:")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   High confidence predictions (>0.7): {high_conf_pct:.1f}%")
        
        # Rating vs sentiment alignment
        rating_sentiment_corr = sentiment_results_df.groupby('original_rating').apply(
            lambda x: (x['sentiment'] == 'positive').sum() / len(x)
        )
        
        print(f"\n⭐ Rating-Sentiment Alignment:")
        for rating, pos_pct in rating_sentiment_corr.items():
            print(f"   {rating} stars: {pos_pct:.1%} positive sentiment")
        
        return {
            'overall_sentiment': overall_sentiment,
            'aspect_scores': aspect_scores,
            'avg_confidence': avg_confidence,
            'high_confidence_pct': high_conf_pct,
            'rating_sentiment_corr': rating_sentiment_corr
        }

def test_sentiment_analysis(sentences_df):
    """
    Test the complete sentiment analysis pipeline
    """
    print("=== TESTING SENTIMENT ANALYSIS PIPELINE ===")
    
    # Initialize analyzer
    analyzer = AspectSentimentAnalyzer()
    
    # Initialize BERT model if available
    analyzer.initialize_bert_model()
    
    # Test individual sentence analysis
    print("\n1. Testing individual sentence analysis:")
    test_sentences = [
        ("The food was absolutely amazing!", "food"),
        ("Our server was incredibly rude and slow", "service"),
        ("Beautiful atmosphere but too noisy", "ambiance"),
        ("Overpriced for what you get", "price")
    ]
    
    for sentence, aspect in test_sentences:
        sentiment, confidence, details = analyzer.analyze_sentence_sentiment(sentence, aspect)
        print(f"Sentence: {sentence}")
        print(f"Aspect: {aspect}, Sentiment: {sentiment}, Confidence: {confidence:.2f}")
        print(f"Method details: {details}")
        print()
    
    # Run full analysis
    print("2. Running full sentiment analysis...")
    sentiment_results = analyzer.analyze_aspect_sentiments(sentences_df)
    
    # Create summaries
    print("3. Creating restaurant summaries...")
    aspect_summary, restaurant_aspects = analyzer.create_restaurant_aspect_summary(sentiment_results)
    
    # Generate insights
    insights = analyzer.generate_insights(sentiment_results, restaurant_aspects)
    
    # Create visualizations
    print("4. Creating visualizations...")
    analyzer.visualize_sentiment_analysis(sentiment_results, restaurant_aspects)
    
    return analyzer, sentiment_results, aspect_summary, insights

if __name__ == "__main__":
    print("🎭 ASPECT-BASED SENTIMENT ANALYSIS MODULE READY! 🎭")
    print("Call test_sentiment_analysis(sentences_df) with your aspect-labeled data")
    print("Make sure you have the output from Phase 2 (sentences_df)")
