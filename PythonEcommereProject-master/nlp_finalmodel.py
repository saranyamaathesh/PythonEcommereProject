#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:33:02 2020

@author: saranya
"""

# -*- coding: utf-8 -*-

###Loading the required packages
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###reading the csv file
df=pd.read_csv('data_file.csv')
df.head()
df.shape
df.info()


###To how many null values are present
df.isnull().sum().sort_values(ascending=False)

###Deleting the unused columns
data=df.drop(['asin','answerTime','questionType','unixTime','answerType'],axis=1)

###find the unique values
df1= data.drop_duplicates()
df1.shape

###drop the null values
df_new = df1.dropna()
df_new.shape
df_new.describe()
df_new.dtypes
df_new.isnull().sum()


###loading the packages for further analysis
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('stopwords')


###RAKE short for domain independent keyword extraction algorithm
from rake_nltk import Rake
df_new['Key_words'] = ''
r = Rake()
for index, row in df_new.iterrows():
    r.extract_keywords_from_text(row['question'])
    key_words_dict_scores = r.get_word_degrees()
    row['Key_words'] = list(key_words_dict_scores.keys())

df_new.head()

###creating bag_of_words
df_new['Bag_of_words'] = ''
columns = ['Key_words']
for index, row in df_new.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words


###selecting only the rows needed for the model building
df = df_new[['question','answer','Bag_of_words']]

Question_answer = ' '.join(df['Bag_of_words'])



###Word cloud for visualizing the bag of words

wordcloud=WordCloud(
                    background_color='white',
                    width=1800,
                    height=1400
                   ).generate(str(df['Bag_of_words']))
fig = plt.figure(figsize = (10, 15))
plt.axis('off')
plt.imshow(wordcloud)


df.head()


###creating tfidf vector to find the freq words in the document to the freq of docs the word appears in it
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(df['question']))
print(tfidf_matrix.shape)



###defining a function to find the similarity based information retrival
def ask_question(Bag_of_words):
    query_vect = tfidf_vectorizer.transform([Bag_of_words])

    similarity = cosine_similarity(query_vect, tfidf_matrix)
    top_5_simmi = similarity[0].argsort()[-5:][::-1]
    count =1
    for i in top_5_simmi:
        print('ANSWER:- ',count)
        print('answer: ', df.iloc[i]['answer'])
        print('similarity: {:.2%}'.format(similarity[0, i]))
        count+=1


###user entered text
ask_question(input(' Your question is: '))


​
