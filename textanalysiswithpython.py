import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import spacy
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
nlp = spacy.load('en_core_web_sm')
data = pd.read_csv("C:\\CODING\\PYTHON PROJECTS\\TEXT_ANALYSIS\\Topic Modelling\\articles.csv", encoding='latin-1')
print(data.head())
titles_text = ''.join(data['Title'])
wordcloud = WordCloud(width=800, height=400, background_color= 'white').generate(titles_text)
fig = px.imshow(wordcloud, title='Word Cloud of Titles')
fig.update_layout(showlegend=False)
fig.show()
data['Sentiment'] = data['Article'].apply(lambda x: TextBlob(x).sentiment.polarity)
fig = px.histogram(data, x='Sentiment', title='Sentiment Distribution')
fig.show()