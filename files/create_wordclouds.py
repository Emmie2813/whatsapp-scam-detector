import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv('Output/CSVs/clean_dataset.csv')
scam = df[df['label'] == 1]['Message'].dropna()
non_scam = df[df['label'] == 0]['Message'].dropna()

def clean_and_join(texts):
    words = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words.extend([w for w in text.split() if w not in stop_words and len(w) > 2])
    return ' '.join(words)

scam_text = clean_and_join(scam)
non_scam_text = clean_and_join(non_scam)

scam_wc = WordCloud(width=800, height=400, background_color='white').generate(scam_text)
scam_wc.to_file('Output/Visualizations/scam_wordcloud.png')
plt.imshow(scam_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

non_scam_wc = WordCloud(width=800, height=400, background_color='white').generate(non_scam_text)
non_scam_wc.to_file('Output/Visualizations/non_scam_wordcloud.png')
plt.imshow(non_scam_wc, interpolation='bilinear')
plt.axis('off')
plt.show()