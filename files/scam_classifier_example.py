import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your labeled data
df = pd.read_csv('scam_data_set.csv')
X = df['Message'].fillna('')
y = (df['Is Scam'] == 'Yes').astype(int)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=3000)
X_vect = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))