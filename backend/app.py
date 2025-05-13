from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# multilingual support libraries
from langdetect import detect, LangDetectException
from googletrans import Translator

# Ensure required NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)

# Initialize the VADER analyzer and translator
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # try to detect language
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'en'

    # if not English, translate to English
    if lang != 'en':
        try:
            translated = translator.translate(text, src=lang, dest='en')
            text_to_analyze = translated.text
        except Exception:
            # fallback to original
            text_to_analyze = text
    else:
        text_to_analyze = text

    # sentiment analysis on English text
    scores = analyzer.polarity_scores(text_to_analyze)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return jsonify({
        'sentiment': sentiment,
        'original_language': lang,
        'scores': scores
    })

if __name__ == '__main__':
    app.run(debug=True)
