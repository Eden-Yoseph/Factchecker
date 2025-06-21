from flask import Flask, render_template, request, jsonify
import requests
import json
import os
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

# Hugging Face API Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN") 
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Models used
MODELS = {
    "primary": "martin-ha/toxic-comment-model",  # Good for detecting misleading content
    "secondary": "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Sentiment analysis
    "backup": "microsoft/DialoGPT-medium"  # Fallback option
}

def call_huggingface_api(model, text, max_retries=3):
    """Call Hugging Face Inference API with retry logic"""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    headers["Content-Type"] = "application/json"
    
    url = f"{HF_API_URL}{model}"
    payload = {"inputs": text}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Request timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Request error: {str(e)}")
            return None
    
    return None

def analyze_with_multiple_models(text):
    """Analyze text using multiple AI models for better accuracy"""
    results = {}
    
    # Try primary model for toxicity/misleading content detection
    try:
        primary_result = call_huggingface_api(MODELS["primary"], text)
        if primary_result:
            results["toxicity"] = primary_result
    except Exception as e:
        print(f"Primary model error: {e}")
    
    # Try sentiment analysis
    try:
        sentiment_result = call_huggingface_api(MODELS["secondary"], text)
        if sentiment_result:
            results["sentiment"] = sentiment_result
    except Exception as e:
        print(f"Sentiment model error: {e}")
    
    return results

def analyze_text_features(text):
    """Analyze text features that might indicate fake news"""
    features = {
        "length": len(text),
        "word_count": len(text.split()),
        "exclamation_marks": text.count('!'),
        "question_marks": text.count('?'),
        "all_caps_words": len([word for word in text.split() if word.isupper() and len(word) > 2]),
        "numbers": len([char for char in text if char.isdigit()]),
        "has_quotes": '"' in text or "'" in text,
    }
    
    # Suspicious patterns common in fake news
    suspicious_phrases = [
        "you won't believe", "doctors hate this", "secret that", "they don't want you to know",
        "shocking truth", "incredible discovery", "amazing breakthrough", "miracle cure",
        "scientists baffled", "experts shocked", "unbelievable results", "this will blow your mind"
    ]
    
    clickbait_indicators = [
        "click here", "find out", "you'll never guess", "what happens next",
        "number 7 will shock you", "the result will surprise you"
    ]
    
    features["suspicious_phrases"] = sum(1 for phrase in suspicious_phrases if phrase.lower() in text.lower())
    features["clickbait_indicators"] = sum(1 for phrase in clickbait_indicators if phrase.lower() in text.lower())
    
    return features

def calculate_fake_news_score(ai_results, text_features, original_text):
    """Calculate fake news probability based on AI results and text analysis"""
    
    fake_score = 0
    confidence = 50
    explanation_parts = []
    
    # Analyze AI results
    if "toxicity" in ai_results and ai_results["toxicity"]:
        try:
            # Handle different response formats
            toxicity_data = ai_results["toxicity"]
            if isinstance(toxicity_data, list) and len(toxicity_data) > 0:
                if isinstance(toxicity_data[0], dict):
                    # Look for toxic/misleading indicators
                    for item in toxicity_data[0]:
                        if isinstance(item, dict) and "label" in item:
                            label = item["label"].lower()
                            score = item.get("score", 0)
                            if "toxic" in label or "fake" in label or score > 0.7:
                                fake_score += 30
                                explanation_parts.append("AI detected potentially misleading language patterns")
        except Exception as e:
            print(f"Error processing toxicity results: {e}")
    
    if "sentiment" in ai_results and ai_results["sentiment"]:
        try:
            sentiment_data = ai_results["sentiment"]
            if isinstance(sentiment_data, list) and len(sentiment_data) > 0:
                # Extremely positive or negative sentiment can indicate bias
                for item in sentiment_data:
                    if isinstance(item, dict) and "label" in item:
                        label = item["label"].lower()
                        score = item.get("score", 0)
                        if ("negative" in label and score > 0.8) or ("positive" in label and score > 0.9):
                            fake_score += 15
                            explanation_parts.append("Detected unusually strong emotional language")
        except Exception as e:
            print(f"Error processing sentiment results: {e}")
    
    # Analyze text features
    if text_features["suspicious_phrases"] > 0:
        fake_score += text_features["suspicious_phrases"] * 25
        explanation_parts.append("Contains phrases commonly found in misleading content")
    
    if text_features["clickbait_indicators"] > 0:
        fake_score += text_features["clickbait_indicators"] * 20
        explanation_parts.append("Shows clickbait characteristics")
    
    if text_features["exclamation_marks"] > 3:
        fake_score += 10
        explanation_parts.append("Excessive use of exclamation marks")
    
    if text_features["all_caps_words"] > 2:
        fake_score += 15
        explanation_parts.append("Frequent use of all-caps words")
    
    # Text length analysis
    if text_features["word_count"] < 10:
        fake_score += 10
        explanation_parts.append("Very short text may lack context")
    elif text_features["word_count"] > 500:
        fake_score -= 5  # Longer articles are often more credible
        explanation_parts.append("Detailed length suggests thorough reporting")
    
    # Presence of quotes often indicates credible journalism
    if text_features["has_quotes"]:
        fake_score -= 10
        explanation_parts.append("Contains quoted sources")
    
    # Normalize score
    fake_score = max(0, min(100, fake_score))
    
    # Calculate confidence based on how many indicators we found
    indicators_count = len(explanation_parts)
    if indicators_count == 0:
        confidence = 60  
        explanation_parts.append("Analysis shows mixed signals")
    elif indicators_count >= 3:
        confidence = min(95, 70 + indicators_count * 5)
    else:
        confidence = 60 + indicators_count * 10
    
    # Determine if it's fake or real
    is_fake = fake_score > 50
    
    if not explanation_parts:
        explanation_parts.append("Standard analysis completed")
    
    return {
        "is_fake": is_fake,
        "fake_score": fake_score,
        "confidence": confidence,
        "explanation": ". ".join(explanation_parts[:3]) + "."  # Limit to 3 main points
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_news():
    """Analyze news text using Hugging Face AI models"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided for analysis'
            }), 400
        
        news_text = data['text'].strip()
        
        if not news_text:
            return jsonify({
                'error': 'Empty text provided'
            }), 400
        
        if len(news_text) > 2000:  # Limit for free API
            return jsonify({
                'error': 'Text too long. Please limit to 2000 characters for free analysis.'
            }), 400
        
        # Analyze with AI models
        print("Analyzing with Hugging Face AI models...")
        ai_results = analyze_with_multiple_models(news_text)
        
        # Analyze text features
        text_features = analyze_text_features(news_text)
        
        # Calculate final score
        analysis = calculate_fake_news_score(ai_results, text_features, news_text)
        
        # Prepare response
        classification = "FAKE" if analysis["is_fake"] else "REAL"
        
        return jsonify({
            'classification': classification,
            'confidence': analysis["confidence"],
            'explanation': analysis["explanation"],
            'isReal': not analysis["is_fake"],
            'ai_powered': True,
            'models_used': list(MODELS.keys())
        })
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Unable to connect to AI services. Please check your internet connection.'
        }), 500
        
    except Exception as e:
        print(f"Error in check_news: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred during analysis. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'TruthLens AI (Hugging Face)',
        'models_available': list(MODELS.keys())
    })

@app.route('/test')
def test_ai():
    """Test endpoint to verify Hugging Face connection"""
    test_text = "This is a test message to verify AI connectivity."
    
    try:
        result = call_huggingface_api(MODELS["primary"], test_text)
        
        return jsonify({
            'status': 'success' if result else 'failed',
            'message': 'AI connection test completed',
            'result': result,
            'has_token': bool(HF_API_TOKEN)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    print("🚀 Starting TruthLens with Hugging Face AI")
    print("=" * 50)
    
    if HF_API_TOKEN:
        print("✅ Hugging Face API token found")
    else:
        print("⚠️  No Hugging Face API token found")
        print("   You can still use the service with rate limits")
        print("   Get a free token at: https://huggingface.co/settings/tokens")
    
    print(f"📍 Available models: {list(MODELS.keys())}")
    print("🌐 Server starting at: http://127.0.0.1:5000")
    print("🧪 Test AI connection at: http://127.0.0.1:5000/test")
    print("=" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)

