from flask import Flask, request, jsonify
import joblib
from scam_detector import ScamDetector

MODEL_PATH = 'whatsapp_scam_detector.joblib'

app = Flask(__name__)

model_files = joblib.load(MODEL_PATH)
scam_detector = ScamDetector(
    model_files['model'],
    model_files['tfidf_vectorizer'],
    model_files['scam_indicators'],
    model_files['threshold'],
    model_files['feature_columns']
)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    result = scam_detector.predict(message)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)