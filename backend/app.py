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

# In-memory comment store: { article_link: [ {user, text, sentiment, emotion, time} ] }
comment_store = {}

# Map compound score to more nuanced emotion categories
def map_emotion(compound_score):
    if compound_score >= 0.6:
        return 'very_happy'
    elif 0.2 <= compound_score < 0.6:
        return 'happy'
    elif -0.2 < compound_score < 0.2:
        return 'neutral'
    elif -0.6 < compound_score <= -0.2:
        return 'disappointed'
    else:
        return 'angry'

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # detect language
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'en'

    # translate if needed
    if lang != 'en':
        try:
            translated = translator.translate(text, src=lang, dest='en')
            text_to_analyze = translated.text
        except Exception:
            text_to_analyze = text
    else:
        text_to_analyze = text

    # sentiment analysis
    scores = analyzer.polarity_scores(text_to_analyze)
    compound = scores['compound']

    # basic sentiment label
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    # more nuanced emotion
    emotion = map_emotion(compound)

    return jsonify({
        'sentiment': sentiment,
        'emotion': emotion,
        'original_language': lang,
        'scores': scores
    })

@app.route('/comment', methods=['POST'])
def post_comment():
    data = request.get_json() or {}
    link = data.get('link')
    user = data.get('user')
    text = data.get('text', '').strip()
    if not link or not user or not text:
        return jsonify({'error': 'Missing link, user, or text'}), 400

    # detect and translate comment
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'en'

    if lang != 'en':
        try:
            translated = translator.translate(text, src=lang, dest='en')
            to_analyze = translated.text
        except Exception:
            to_analyze = text
    else:
        to_analyze = text

    # analyze comment
    scores = analyzer.polarity_scores(to_analyze)
    compound = scores['compound']
    sentiment = 'positive' if compound >= 0.05 else 'negative' if compound <= -0.05 else 'neutral'
    emotion = map_emotion(compound)

    comment = {
        'user': user,
        'text': text,
        'time': request.environ.get('REQUEST_TIME', None),
        'scores': scores,
        'sentiment': sentiment,
        'emotion': emotion
    }

    # store comment
    comment_store.setdefault(link, []).append(comment)

    # return stored comments for this article
    return jsonify({'comments': comment_store[link]})

@app.route('/comments', methods=['GET'])
def get_comments():
    link = request.args.get('link')
    if not link:
        return jsonify({'error': 'Missing link parameter'}), 400
    comments = comment_store.get(link, [])
    return jsonify({'comments': comments})

if __name__ == '__main__':
    app.run(debug=True)