import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class ScamDetector:
    def __init__(self, model, tfidf_vectorizer, scam_indicators, threshold, feature_columns):
        self.model = model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.scam_indicators = scam_indicators
        self.threshold = threshold
        self.feature_columns = feature_columns
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def extract_features(self, text):
        processed_text = self.preprocess_text(text)
        features = {}
        for kw in self.scam_indicators:
            features[f'has_{kw}'] = int(re.search(rf'\b{re.escape(kw)}\b', processed_text) is not None)
        features['text_length'] = len(text)
        features['word_count'] = len(processed_text.split())
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) else 0
        tfidf_feats = self.tfidf_vectorizer.transform([processed_text])
        tfidf_df = pd.DataFrame(tfidf_feats.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        features_df = pd.DataFrame([features])
        all_feats = pd.concat([features_df, tfidf_df], axis=1)
        for col in self.feature_columns:
            if col not in all_feats:
                all_feats[col] = 0
        all_feats = all_feats[self.feature_columns]
        return all_feats

    def predict(self, text):
        feats = self.extract_features(text)
        proba = self.model.predict_proba(feats)[0, 1]
        is_scam = int(proba >= self.threshold)
        return {
            "is_scam": bool(is_scam),
            "scam_probability": float(proba),
            "risk_level": "High" if proba >= 0.7 else "Medium" if proba >= 0.3 else "Low",
            "threshold": self.threshold
        }