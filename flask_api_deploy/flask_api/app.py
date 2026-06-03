from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

# ── FEATURES ────────────────────────────────────────────────
FEATURES = [
    'incident_type', 'severity', 'frequency',
    'social_isolation', 'emotional_distress', 'school_avoidance',
    'low_self_esteem', 'peer_fear', 'academic_decline',
    'aggressive_behaviour', 'verbal_teasing', 'peer_pressure',
    'lack_of_empathy', 'rule_defiance', 'dominance_seeking'
]

THRESHOLD = 0.5

RISK_KEYWORDS = [
    'scared', 'bully', 'bullied', 'bullying', 'crying', 'cry', 'tease', 'teased',
    'teasing', 'isolated', 'lonely', 'alone', 'hit', 'pushed', 'push', 'fear',
    'afraid', 'anxious', 'sad', 'depressed', 'ignored', 'excluded', 'threaten',
    'threatened', 'hurt', 'pain', 'avoid', 'withdrawn', 'nervous', 'uncomfortable',
    'bruises'
]

VIOLENCE_KEYWORDS = [
    'beat', 'hit', 'punch', 'kicked', 'slap', 'attack', 'hurt', 'assault'
]

# ── LOAD MODELS ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

victim_model = load_model('victim_model.pkl')
bully_model  = load_model('bully_model.pkl')

# ── ROUTES ───────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'NOBULLY API is running', 'endpoints': ['/predict', '/analyze']})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        mode = data.get('mode', 'both')
        features = [[data.get(f, 0) for f in FEATURES]]
        result = {}

        if mode in ('victim', 'both'):
            v_prob = float(victim_model.predict_proba(features)[0][1])
            result['victim_score'] = round(v_prob, 4)
            result['victim_alert'] = v_prob >= THRESHOLD

        if mode in ('bully', 'both'):
            b_prob = float(bully_model.predict_proba(features)[0][1])
            result['bully_score'] = round(b_prob, 4)
            result['bully_alert'] = b_prob >= THRESHOLD

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        text_lower = text.lower()

        found_keywords = [kw for kw in RISK_KEYWORDS
                          if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        violence_found = [kw for kw in VIOLENCE_KEYWORDS
                          if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]

        compound = sentiment['compound']
        keyword_count = len(found_keywords)

        if violence_found:
            risk_level = 'HIGH'
        elif compound <= -0.5 or keyword_count >= 3:
            risk_level = 'HIGH'
        elif compound <= -0.2 or keyword_count >= 1:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        result = {
            'sentiment_score':  round(compound, 3),
            'negative_score':   round(sentiment['neg'], 3),
            'neutral_score':    round(sentiment['neu'], 3),
            'positive_score':   round(sentiment['pos'], 3),
            'risk_keywords':    list(set(found_keywords + violence_found))[:10],
            'risk_level':       risk_level,
            'keyword_count':    len(set(found_keywords + violence_found)),
            'text_length':      len(text)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
