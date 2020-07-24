# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:29:37 2020

@author: Anirudh Raghavan
"""

import pandas as pd

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

import numpy as np

reviews = pd.read_csv("Hotel_Reviews_extract.csv", index_col = False)

vocab = str()

for i in range(reviews.shape[0]):
    vocab = vocab + reviews.iloc[i,0]
    vocab = vocab + reviews.iloc[i,1]


len(vocab)

def tokenizer(text):
    return text.split()

def tokenzier_porter(text):
    porter=PorterStemmer()
    return [porter.stem(words) for words in text.split()]



nltk.download('punkt')

reviews.iloc[0,1]

tokenizer(reviews.iloc[1,1])


tokenzier_porter(reviews.iloc[0,1])


from nltk.tokenize import word_tokenize 
  
example_sent = reviews.iloc[1,1]
  
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(vocab) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence) 


from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = [vocab]


# create the transform
vectorizer = CountVectorizer(stop_words = 'english')
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)

vectorizer

# encode document
vector = vectorizer.transform([reviews.iloc[1,0]+reviews.iloc[1,1]])
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(type(vector.toarray())

dir(vectorizer)

vectorizer.vocabulary_.keys()

file_name = "vocab.txt"
file = open(file_name,"a")
file.write(vocab)
file.close()    


len(vectorizer.vocabulary_)

vector_reviews = pd.DataFrame(columns = list(vectorizer.vocabulary_.keys()))


vector_reviews = np.array(list(vectorizer.vocabulary_.keys()))

print("Empty array:")
print(arr)
vector_reviews = np.append(vector_reviews, vector, axis=0)


vector_reviews = np.empty((0,len(vectorizer.vocabulary_)))

vector_reviews.shape[0]

for i in range(vector_reviews.shape[0],reviews.shape[0]):
    print(i)
    vector = vectorizer.transform([reviews.iloc[i,0]+reviews.iloc[i,1]])
    vector = vector.toarray()
    vector_reviews = np.append(vector_reviews, vector, axis=0)




vector = vector.toarray()
vector_reviews_1 = np.transpose(vector_reviews)
vector_reviews = np.stack((vector_reviews, vector), axis = 0) 
 
S = np.hstack(np.array([[review] for review in range(reviews.shape[0])))
        

    