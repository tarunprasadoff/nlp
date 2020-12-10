from __future__ import print_function
import gensim.downloader as api # package to download text corpus
import nltk # text processing
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import csv
import os

# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')

# collect all words to be removed
stop = stopwords.words('english') + list(string.punctuation)

actual_words = []
cleaned_words = []
unique_words = set()

# remove stop words
print('removing stop words from text corpus')
for words in data:
    current_nonstop_words = [w for w in words if w not in stop]
    cleaned_words += current_nonstop_words
    actual_words += words

    for ns in current_nonstop_words:
        unique_words.add(ns)

# print statistics
print(len(actual_words), 'words BEFORE cleaning stop words and punctuations')
print(len(cleaned_words), 'words AFTER cleaning stop words and punctuations')
print('vocabulary size: ', len(unique_words))

# 'cleaned_words' and 'unique_words' to create a word2vec model

print("Saving Unique Word Array")

with open('unique_words.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % unique for unique in unique_words)

wordToIndexDictionary = {w: idx for (idx, w) in enumerate(unique_words)}
indexToWordDictionary = {idx: w for (idx, w) in enumerate(unique_words)}

print("Saving Word-Index Dictionaries")

with open("wordToIndexDictionary.txt",'w+') as f:
    f.write(str(wordToIndexDictionary))

with open("indexToWordDictionary.txt",'w+') as f:
    f.write(str(indexToWordDictionary))

print("Preparing SkipGram Dataset")

# Preparing the context centre word pairs
window_size=2

center_context_idx_pairs = pd.DataFrame(columns=["Center","Context"])
center_context_idx_pairs.to_csv("SkipGram.csv",index=False)
indices = [wordToIndexDictionary[word] for word in cleaned_words]
# Identifying the context for each word as the centre word
counter = 0
total_length = len(indices)

f = open('SkipGram.csv', 'a')
with f:
    writer = csv.writer(f)
    for center_word_pos in range(window_size,(total_length-window_size)):
        center_word_idx = indices[center_word_pos]
        # For a window size of 2
        writer.writerows(np.array([(center_word_idx,indices[center_word_pos-2]),(center_word_idx,indices[center_word_pos-1]),(center_word_idx,indices[center_word_pos+1]),(center_word_idx,indices[center_word_pos+2])]))

print("Preparing CBOW Dataset")

# Preparing the context centre word pairs
window_size=2

center_context_idx_pairs = pd.DataFrame(columns=["Context:-2","Context:-1","Context:1","Context:2","Centre"])
center_context_idx_pairs.to_csv("CBOW.csv",index=False)
# Identifying the context for each word as the centre word
counter = 0
total_length = len(indices)

f = open('CBOW.csv', 'a')
with f:
    writer = csv.writer(f)
    for center_word_pos in range(window_size,(total_length-window_size)):
        center_word_idx = indices[center_word_pos]
        # For a window size of 2
        writer.writerow(np.array([indices[center_word_pos-2],indices[center_word_pos-1],indices[center_word_pos+1],indices[center_word_pos+2],center_word_idx]))

print("Preparing Train and Test Sets for SkipGram")

skipGramDf = pd.read_csv("SkipGram.csv")

sgTrainIndices = np.random.rand(len(skipGramDf)) < 0.8

sgTrain = skipGramDf[sgTrainIndices]
sgTrain.to_csv("SkipGram_Train.csv", index=False)

sgTest = skipGramDf[~sgTrainIndices]
sgTest.to_csv("SkipGram_Test.csv", index=False)

os.remove("SkipGram.csv")

print("Preparing Train and Test Sets for CBOW")

cbowDf = pd.read_csv("CBOW.csv")

cbowTrainIndices = np.random.rand(len(cbowDf)) < 0.8

cbowTrain = cbowDf[cbowTrainIndices]
cbowTrain.to_csv("CBOW_Train.csv", index=False)

cbowTest = cbowDf[~cbowTrainIndices]
cbowTest.to_csv("CBOW_Test.csv", index=False)

os.remove("CBOW.csv")