
#Getting the Data from Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Just making sure that the mounting worked
import os
data_dir = '/content/drive/MyDrive/ML Capstone Project/Data/Raw/chest_xray'
print(os.listdir(data_dir))

#Here we make directories easier to access
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

print(os.listdir(train_dir))

#Knowing the numbers of photos in every directory to make sure there isnt any imbalances


#Training
norm_train_dir = os.path.join(train_dir, 'NORMAL')
print('total training normal images:', len(os.listdir(norm_train_dir)))
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')
print('total training pneumonia images:', len(os.listdir(pneumonia_train_dir)))
print("\n")

#Val
norm_val_dir = os.path.join(val_dir, 'NORMAL')
print('total val normal images:', len(os.listdir(norm_val_dir)))
pneumonia_val_dir = os.path.join(val_dir, 'PNEUMONIA')
print('total val pneumonia images:', len(os.listdir(pneumonia_val_dir)))
print("\n")

#Test
norm_test_dir = os.path.join(test_dir, 'NORMAL')
print('total test normal images:', len(os.listdir(norm_test_dir)))
pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')
print('total test pneumonia images:', len(os.listdir(pneumonia_test_dir)))

import random

def generate_caption(label):
    normal_captions = [
        "chest xray showing normal lung tissue",
        "clear lung fields with no abnormalities",
        "normal chest xray with no signs of infection",
        "healthy lung appearance with clear airways",
        "no evidence of pneumonia in chest xray"
    ]
    pneumonia_captions = [
        "chest xray showing signs of pneumonia infection",
        "lung opacity visible indicating possible pneumonia",
        "abnormal chest xray with infiltrates present",
        "chest xray showing consolidation in lung fields",
        "evidence of pneumonia with cloudy lung appearance"
    ]
    if label == 0:
        return random.choice(normal_captions)
    else:
        return random.choice(pneumonia_captions)

import numpy as np

# Generate captions for train set
normal_count = 1342
pneumonia_count = 3876

captions = []
labels = []

for i in range(normal_count):
    captions.append(generate_caption(0))
    labels.append(0)

for i in range(pneumonia_count):
    captions.append(generate_caption(1))
    labels.append(1)

import pandas as pd
df = pd.DataFrame({'caption': captions, 'label': labels})
print(df.shape)
print(df.head(10))

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenize
    tokens = text.split()
    # remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['cleaned'] = df['caption'].apply(preprocess_text)
print(df[['caption', 'cleaned']].head(5))

from collections import Counter
import matplotlib.pyplot as plt

all_words = ' '.join(df['cleaned']).split()
word_counts = Counter(all_words)
top_20 = word_counts.most_common(20)

words, counts = zip(*top_20)

plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Word')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Split data
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

# Train Logistic Regression
lr_nlp = LogisticRegression()
lr_nlp.fit(X_tfidf_train, y_text_train)

# Evaluate
y_pred = lr_nlp.predict(X_tfidf_test)
print(classification_report(y_text_test, y_pred))

from wordcloud import WordCloud

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (label, name) in enumerate([(0, 'NORMAL'), (1, 'PNEUMONIA')]):
    text = ' '.join(df[df['label'] == label]['cleaned'])
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)
    axes[idx].imshow(wordcloud)
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.show()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# Tokenize
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(X_text_train)

X_lstm_train = pad_sequences(tokenizer.texts_to_sequences(X_text_train), maxlen=20)
X_lstm_test = pad_sequences(tokenizer.texts_to_sequences(X_text_test), maxlen=20)

# Build LSTM
lstm_model = models.Sequential([
    layers.Embedding(500, 32, input_length=20),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_lstm = lstm_model.fit(X_lstm_train, y_text_train,
                               epochs=10,
                               validation_split=0.2)