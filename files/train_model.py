import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, f1_score

with open('scam_keywords.txt', 'r', encoding='utf-8') as f:
    scam_keywords = [line.strip().lower() for line in f if line.strip()]

df = pd.read_csv('Output/CSVs/clean_dataset.csv')
X_base = df.drop(['label', 'Message'], axis=1)
y = df['label']

tfidf = TfidfVectorizer(max_features=3000)
tfidf_matrix = tfidf.fit_transform(df['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
X = pd.concat([X_base.reset_index(drop=True).drop('processed_text', axis=1), tfidf_df], axis=1)
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
clf.fit(X_train_bal, y_train_bal)

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_pred_proba))

best_threshold = 0.3  # You may fine-tune this as needed

joblib.dump({
    'model': clf,
    'tfidf_vectorizer': tfidf,
    'scam_indicators': scam_keywords,
    'threshold': best_threshold,
    'feature_columns': feature_columns
}, 'whatsapp_scam_detector.joblib')
print("✅ Model and pipeline saved to whatsapp_scam_detector.joblib")