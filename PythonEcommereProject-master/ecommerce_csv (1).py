import json
import csv
import ast
import os
import pandas as pd
import numpy as np
import sklearn as sk
import scipy as sc
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
import nltk
import re
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import nltk
from nltk.stem import LancasterStemmer
from spacy.lang.en import English
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()
from spacy.lang.en.stop_words import STOP_WORDS
#nltk.download()
#nltk.download('stopwords')
#nltk.download()
#Setting up the workimg directory

os.chdir('D:\\datascience\\data science assignments\\')

# Opening the JSON file and loading the data into variable
# Use English stemmer.
stemmer = SnowballStemmer("english")

data=[]
with open("qa_Electronics.json",encoding='utf-8') as json_file:
   for i in json_file:
       c=ast.literal_eval(i)
       data.append(c)
#type(data)   

#writing the data into csv file
       
data_file=open('data_file.csv','w')

#create the CSV writer Object

csv_writer=csv.writer(data_file)

#Counter variable used for writing variables into csv file

header=['questionType','asin','answerTime','unixTime','question','answerType','answer']
csv_writer.writerow(header)

#looping the data line by line
for line in data:
    default_list=['','','','','','',''] 
    # if column data is unavilable insert NAN
    if len(line)!=0:    #cheking row is empty of not
        for j in line:
            if j== "questionType":
                default_list[0]=line[j]
            if j== "asin":
                default_list[1]=line[j]
            if j== "answerTime":
                default_list[2]=line[j]
            if j== "unixTime":
                default_list[3]=line[j]
            if j== "question":
                default_list[4]=line[j]
            if j== "answerType":
                default_list[5]=line[j]
            if j== "answer":
                default_list[6]=line[j]
            
        csv_writer.writerow(default_list)   
        
df = pd.read_csv('data_file.csv')
df.to_csv('qa_electtrimmed.csv', index=False)
df.isna().sum()

  #to drop if answer in the row has a nan
df_answernan=df.dropna(subset=['answer'])  
df_answernan.isna().sum()
df.isna().sum()

df_answernan = df_answernan.drop(columns="unixTime")
df_answernan = df_answernan.drop(columns="answerTime")
df_answernan.isna().sum()
df.isna().sum()
#df.loc[:,"answer"] = df.answer.apply(lambda x : " ".join(re.findall('[\w]+',x)))


df_answernan.loc[:,"answer"] = df_answernan.answer.apply(lambda x : " ".join(re.findall('[\w]+',x)))
stop = stopwords.words('english')
#make it all lower case
df_answernan['answer'] = df_answernan.answer.apply(lambda x: x.lower())

df_answernan['answerstop'] = df_answernan['answer'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))



  #to drop if question in the row has a nan
df_answernan=df_answernan.dropna(subset=['question'])  
df_answernan.isna().sum()
df.isna().sum()

df_answernan.loc[:,"question"] = df_answernan.question.apply(lambda x : " ".join(re.findall('[\w]+',x)))
#make it all lower case
df_answernan['question'] = df_answernan.question.apply(lambda x: x.lower())

df_answernan['questionstop'] = df_answernan['question'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

df_answernan.to_csv('qa_trimmed.csv', index=False)

#second module
from rake_nltk import Rake
df_answernan['answerstop_keywords'] = ''
r = Rake()
for index, row in df_answernan.iterrows():
    r.extract_keywords_from_text(row['answerstop'])
    key_words_dict_scores = r.get_word_degrees()
    row['answerstop_keywords'] = list(key_words_dict_scores.keys())

#second module
from rake_nltk import Rake
df_answernan['questionstop_keywords'] = ''
r = Rake()
for index, row in df_answernan.iterrows():
    r.extract_keywords_from_text(row['questionstop'])
    key_words_dict_scores = r.get_word_degrees()
    row['questionstop_keywords'] = list(key_words_dict_scores.keys())


##creating unique identity
#df_answernan['answerstop_keywords'] = df_answernan['answerstop_keywords'].map(lambda x: x.split(','))
#df_answernan['questionstop_keywords'] = df_answernan['questionstop_keywords'].map(lambda x: x.split(','))

#for index, row in df_answernan.iterrows():
 #   row['answerstop_keywords'] = [x.lower().replace(' ','') for x in row['answerstop_keywords']]
  #  row['questionstop_keywords'] = [x.lower().replace(' ','') for x in row['questionstop_keywords']]
    
df_answernan['Bag_of_words'] = ''
columns = ['answerstop_keywords', 'questionstop_keywords']
for index, row in df_answernan.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
    

text = df_answernan.answerstop_keywords.values
text1= df_answernan.questionstop_keywords.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text)) 
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


wordcloud1 = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text1))

fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud1, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

df_answernan.to_csv('qa_trimmed.csv', index=False)   
import pandas as pd
data1=pd.read_csv("file:///D://datascience//data science assignments//qa_trimmed.csv")
data1.columns

##########################lDA Model Building for Question and Answer####################################

import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()


###LDA model for answer
words1=pd.Series(data1['Bag_of_words']).apply(lambda x: x.split())##tokenizing
dictionary1=gensim.corpora.Dictionary(words1)##mapping words with their integer id's


doc_term_matrix1=[dictionary1.doc2bow(doc) for doc in words1]##shows no.of times word appears in each document
bow_doc_x = doc_term_matrix1[1]


LDA=gensim.models.ldamodel.LdaModel
lda_model1=LDA(corpus=doc_term_matrix1, id2word=dictionary1, num_topics=5)
lda_topics=lda_model1.print_topics()
lda_topics
vis1=pyLDAvis.gensim.prepare(lda_model1, doc_term_matrix1, dictionary1)
pyLDAvis.show(vis1)


# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model1.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

data_file.close()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer(tokenizer = my_tokenizer)

tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(df_final['question']))

print(tfidf_matrix.shape)


 
from sklearn.metrics.pairwise w cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
count_matrix = count.fit_transform(df_answernan['Bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


indices = pd.Series(df_answernan['question'])
def recommend(question, cosine_sim = cosine_sim):
    recommended_question = []
    idx = indices[indices == question].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:11].index)
    
    for i in top_10_indices:
        recommended_question.append(list(df_answernan['question'])[i])
        
    return recommended_question


bow_corpus = [dictionary1.doc2bow(words1) for words1 in words1]



#from itertools import islice
#def take(n, iterable):
 #   return list(islice(iterable, n))

#items_5 = take(2, dictionary1.iteritems())
#items_5

#print('Number of unique tokens: %d' % len(dictionary1)) # roughly 134301 less than in presentation
#print('Number of articles: %d' % len(bow_corpus)) # roughly 314207 less than in presentation

#print(bow_corpus[0])

#def tokenization_report(bow, keyword_num):
 #   keyword_tokens = bow[keyword_num]
  #  sorted_tokens = sorted(keyword_tokens, key=lambda x: x[1], reverse=True)
   # for i in range(len(bow[keyword_num])):
    #   print("Word {} (\"{}\") appears {} time(s).".format(sorted_tokens[i][0],                                                          dictionary1[sorted_tokens[i][0]],                                                          sorted_tokens[i][1]))
               
#tokenization_report(bow_corpus, 0)



#from fuzzywuzzy import fuzz

#def get_ratio(row):
 #   name = data1['question']
  #  name1 = 'macbook air'
   # return fuzz.token_set_ratio(name, name1)

#data1[data1.apply(get_ratio, axis=1) > 70].head(3)



#stemmer words cleaner

#from nltk.corpus import wordnet
#import nltk
#from nltk.stem import WordNetLemmatizer 
# 1. Init Lemmatizer
#lemmatizer = WordNetLemmatizer()


#def get_frequentWord():
#    top_N = 200
#    word_dist = nltk.FreqDist(df_answernan['answerstop'])
#    print('All frequencies')
#    rslt=pd.DataFrame(word_dist.most_common(top_N),columns=['Word','Frequency'])
#    df_answernan.answerstop.str.split(expand=True).stack().value_counts()

#print(df_answernan['asin'] == '0594033926') 
#print(df.iloc[np.where(df_answernan.asin.values=='0594033926')])

#group = df_answernan.groupby('asin')
#df_answernanm = group.apply(lambda x: x['answerstop'].unique())
#df_answernan['answertrip']= DataFrame(df_answernanm).transpose()

#def remove_stopWords(txt_tokenizer):
#    txt_clean = [word for word in txt_tokenizer if word not in stop]
#    return txt_clean
#df_answernan['answer_removtoken']=df_answernan['answer'].apply(lambda x: remove_stopWords(x))
#df_answernan.loc[:,"answer"] = df_answernan.answer.apply(lambda x: [item for item in x if item not in stop])

#def get_wordnet_pos(word):
#    """Map POS tag to first character lemmatize() accepts"""
#    tag = nltk.pos_tag([word])[0][1][0].upper()
#    tag_g = {"J": wordnet.ADJ,
#                "N": wordnet.NOUN,
#                "V": wordnet.VERB,
#                "R": wordnet.ADV}
#    return tag_g.get(tag, wordnet.NOUN)
# 3. Lemmatize a Sentence with the appropriate POS tag
#sentence = "The striped bats are hanging on their feet for best smaller specified mounted inches already plot plunty shaped thanks charging installed fitting enlarged places definitely measures"
#df_answernan['stemm']=[ for w in nltk.word_tokenize(df_answernan['answerstop'])]
#df_answernan['stemm']= df_answernan['answerstop'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
#df_answernan['answerstem']=df_answernan['answerstop'].apply(lambda x:[lemmatizer.lemmatize(w, get_wordnet_pos(w))                                         for w in nltk.word_tokenize(x)])
# Print most common word
