import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

with open('scam_keywords.txt', 'r', encoding='utf-8') as f:
    scam_keywords = [line.strip().lower() for line in f if line.strip()]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    input_csv = 'Output/CSVs/bi_modal_dataset.csv'
    output_csv = 'Output/CSVs/clean_dataset.csv'
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    df['processed_text'] = df['Message'].fillna('').apply(preprocess_text)
    df['label'] = df.get('Is Scam', '').map(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'scam'] else 0 if str(x).lower() not in ['', 'nan', 'none'] else np.nan)
    df['label'] = df['label'].fillna(0).astype(int)

    for kw in scam_keywords:
        df[f'has_{kw}'] = df['processed_text'].str.contains(rf'\b{re.escape(kw)}\b', regex=True).astype(int)

    df['text_length'] = df['Message'].fillna('').apply(len)
    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
    df['capital_ratio'] = df['Message'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) else 0)

    out_cols = ['Message', 'processed_text', 'label'] + [f'has_{kw}' for kw in scam_keywords] + ['text_length', 'word_count', 'capital_ratio']
    df_out = df[out_cols]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f'✅ Preprocessing complete. Output: {output_csv}')

if __name__ == '__main__':
    main()