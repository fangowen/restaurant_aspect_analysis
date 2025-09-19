# Test script for aspect extraction system
# Run this after you have both Phase 1 and Phase 2 code

import pandas as pd
import numpy as np

# Import your classes (assuming they're in the same file or imported)
from restaurant_analyzer_setup import RestaurantReviewAnalyzer
from aspect_extraction import AspectExtractor, test_aspect_extraction

def quick_test_with_sample_data():
    """
    Quick test using sample data - no external files needed
    """
    print("=== QUICK TEST WITH SAMPLE DATA ===\n")
    
    # Create sample review data
    sample_reviews = [
        {
            'business_id': 'rest_001',
            'text': 'Amazing food quality! The pasta was perfectly cooked and the sauce was incredible. Service was a bit slow though, waited 20 minutes for our appetizer. The atmosphere was cozy and romantic, perfect for date night. A bit pricey but worth it.',
            'stars': 4,
            'name': 'Bella Vista Italian'
        },
        {
            'business_id': 'rest_002', 
            'text': 'Terrible experience. Food was cold and tasteless. The chicken was dry and seemed reheated. Our server was rude and ignored us most of the night. The restaurant was dirty and loud. Way overpriced for what you get.',
            'stars': 1,
            'name': 'Golden Dragon Chinese'
        },
        {
            'business_id': 'rest_003',
            'text': 'Great casual spot! The burgers are juicy and flavorful. Fast and friendly service. The place gets pretty loud during peak hours but has a fun vibe. Very reasonable prices for the portion sizes.',
            'stars': 4,
            'name': 'Downtown Burger Co'
        },
        {
            'business_id': 'rest_004',
            'text': 'Outstanding fine dining experience. Every dish was a work of art and tasted incredible. The service was impeccable - attentive without being intrusive. Beautiful upscale ambiance with dim lighting. Expensive but justified.',
            'stars': 5,
            'name': 'Le Bernardin NYC'
        },
        {
            'business_id': 'rest_005',
            'text': 'The location is perfect, right downtown with easy parking. Food was decent but nothing special. Our waiter was friendly and knowledgeable. The interior design is modern and clean. Fair prices for the area.',
            'stars': 3,
            'name': 'Central Cafe'
        }
    ]
    
    # Create DataFrame
    reviews_df = pd.DataFrame(sample_reviews)
    print(f"Created {len(reviews_df)} sample reviews for testing")
    
    return reviews_df

def test_step_by_step(reviews_df):
    """
    Test the aspect extraction step by step
    """
    print("\n=== STEP-BY-STEP TESTING ===\n")
    
    # Initialize extractor
    extractor = AspectExtractor()
    
    # Test 1: Sentence extraction
    print("1. Testing sentence extraction:")
    sample_text = reviews_df.iloc[0]['text']
    print(f"Original text: {sample_text}")
    
    sentences = extractor.extract_sentences(sample_text)
    print(f"Extracted {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"   {i}. {sentence}")
    
    # Test 2: Keyword-based aspect detection
    print("\n2. Testing keyword-based aspect detection:")
    for sentence in sentences:
        aspects = extractor.keyword_based_aspect_detection(sentence)
        print(f"Sentence: {sentence}")
        print(f"Detected aspects: {aspects}\n")
    
    # Test 3: Create aspect dataset
    print("3. Creating aspect dataset from all reviews...")
    sentences_df = extractor.create_aspect_dataset(reviews_df)
    
    print(f"Created dataset with {len(sentences_df)} labeled sentences")
    print("\nFirst few rows:")
    print(sentences_df[['sentence', 'aspect', 'original_rating']].head(10))
    
    # Test 4: Analyze distribution
    print("\n4. Analyzing aspect distribution:")
    aspect_counts = extractor.analyze_aspect_distribution(sentences_df)
    
    # Test 5: Train classifier (if enough data)
    print("\n5. Training ML classifier:")
    classifier = extractor.train_aspect_classifier(sentences_df)
    
    # Test 6: Test prediction on new review
    print("\n6. Testing prediction on new review:")
    test_review = """
    I had dinner here last night and it was fantastic! The steak was cooked perfectly - medium rare just like I asked. 
    Our server John was extremely attentive and gave great wine recommendations. 
    The restaurant has a warm, intimate atmosphere with soft lighting and jazz music. 
    It's definitely on the expensive side, but the quality justifies the cost. 
    Easy to find with valet parking available.
    """
    
    print(f"Test review: {test_review}")
    
    results = extractor.predict_aspects(test_review)
    
    print(f"\nExtracted {len(results)} sentences with aspects:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Sentence: {result['sentence']}")
        print(f"   Final aspects: {result['aspects']}")
        if result['keyword_aspects'] != result['ml_aspects']:
            print(f"   Keyword aspects: {result['keyword_aspects']}")
            print(f"   ML aspects: {result['ml_aspects']}")
        print()
    
    return extractor, sentences_df

def test_with_your_yelp_data():
    """
    Test with your actual Yelp data
    """
    print("=== TESTING WITH YOUR YELP DATA ===\n")
    
    try:
        # Load your data using Phase 1 code
        analyzer = RestaurantReviewAnalyzer()
        reviews_df = analyzer.load_yelp_data(
            'yelp_academic_dataset_business.json',
            'yelp_academic_dataset_review.json'
        )
        
        print(f"✅ Loaded {len(reviews_df)} reviews")
        
        # Test with subset for faster processing
        test_subset = reviews_df.head(1000)  # Use first 1000 reviews
        print(f"Testing with subset of {len(test_subset)} reviews")
        
        # Run the full test
        extractor, sentences_df = test_aspect_extraction(test_subset)
        
        return extractor, sentences_df
        
    except FileNotFoundError:
        print("❌ Yelp data files not found in workspace")
        print("Using sample data instead...")
        return test_with_sample_data()
    except Exception as e:
        print(f"❌ Error loading Yelp data: {e}")
        print("Using sample data instead...")
        return test_with_sample_data()

def test_with_sample_data():
    """
    Full test using sample data
    """
    print("=== FULL TEST WITH SAMPLE DATA ===\n")
    
    # Get sample data
    reviews_df = quick_test_with_sample_data()
    
    # Run step by step test
    extractor, sentences_df = test_step_by_step(reviews_df)
    
    return extractor, sentences_df

def manual_test_individual_review():
    """
    Test individual review manually
    """
    print("=== MANUAL TESTING ===\n")
    
    extractor = AspectExtractor()
    
    # Test different types of reviews
    test_reviews = [
        "The food was amazing but the service was terrible and it was way too expensive!",
        "Great atmosphere and friendly staff. The pasta was delicious and reasonably priced.",
        "Couldn't find parking and the restaurant was hard to locate. Food was cold when it arrived.",
        "Beautiful decor and romantic lighting. Perfect for date night. The wine selection was excellent."
    ]
    
    print("Testing individual reviews:\n")
    for i, review in enumerate(test_reviews, 1):
        print(f"Review {i}: {review}")
        results = extractor.predict_aspects(review)
        
        for result in results:
            print(f"  Sentence: {result['sentence']}")
            print(f"  Aspects: {result['aspects']}")
        print()

if __name__ == "__main__":
    print("🚀 ASPECT EXTRACTION TESTING SUITE 🚀\n")
    
    # Choose your test method:
    
    print("Choose a testing method:")
    print("1. Quick test with sample data")
    print("2. Step-by-step detailed test")
    print("3. Test with your Yelp data")
    print("4. Manual test individual reviews")
    
    choice = input("\nEnter choice (1-4) or press Enter for option 1: ").strip() or "1"
    
    if choice == "1":
        extractor, sentences_df = test_with_sample_data()
    elif choice == "2":
        reviews_df = quick_test_with_sample_data()
        extractor, sentences_df = test_step_by_step(reviews_df)
    elif choice == "3":
        extractor, sentences_df = test_with_your_yelp_data()
    elif choice == "4":
        manual_test_individual_review()
        extractor, sentences_df = None, None
    else:
        print("Invalid choice, using sample data test")
        extractor, sentences_df = test_with_sample_data()
    
    if extractor and sentences_df is not None:
        print(f"\n✅ Testing complete!")
        print(f"📊 Created {len(sentences_df)} aspect-labeled sentences")
        print(f"🎯 Ready for Phase 3: Sentiment Analysis")
        
        # Show summary
        print(f"\nAspect distribution:")
        print(sentences_df['aspect'].value_counts())