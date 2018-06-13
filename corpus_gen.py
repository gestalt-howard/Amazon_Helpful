# Script for compiling corpus vocabulary
from modules.get_ready import unzip_df
from modules.get_ready import flatten
from modules.get_ready import text_processor

from gensim.models import Word2Vec

import pandas as pd
import pickle
import string
import nltk
import pdb
import os
import shutil
import collections

from collections import Counter
from nltk.corpus import stopwords


# Load training data and test data
parent_path, [train_df, test_df] = unzip_df(False)

# Extract all textual information to assemble training corpus
reviews = list(train_df.reviewText.values)
summaries = list(train_df.summary.values)
corpus_paired = zip(reviews, summaries)
# Join reviews and summaries together
corpus = []
for pair in corpus_paired:
    corpus.append(' '.join(pair))

# Remove punctuation and uncase all text
print('Removing punctuation and uncasing text...')
replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
docu_corpus = [text_processor(doc, replace_punctuation) for doc in corpus]
word_corpus = [w for doc in docu_corpus for w in doc]

# Train and save a Word2Vec embedding model
print('Training Word2Vec model on training set corpus...')
model = Word2Vec(docu_corpus, min_count=1, size=50, window=6)
model_vocab_list = list(model.wv.vocab)  # Get vocab list from Word2Vec trained model

# Create lookup list of categories
train_categs = train_df.categories.values
test_categs = test_df.categories.values
# Flatten and aggregate into comprehensive categories
train_categs = [list(flatten(x)) for x in train_categs]
test_categs = [list(flatten(x)) for x in test_categs]
comp_categs = train_categs + test_categs
# Table for removing punctuations
replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
# Assemble list of comprehensive split categories
comprehensive = [word for group in comp_categs for cat in group for word in cat.translate(replace_punctuation).split()]
comp_dict = collections.Counter(comprehensive)
comp_keys = list(comp_dict.keys())

# Prepare for saving...
corpus_folder = parent_path + 'corpus_data/'
corpus_name = corpus_folder + 'train_corpus_vocab.pickle'
model_name = corpus_folder + 'word_model.bin'
cat_dict_name = corpus_folder + 'categories.pickle'
if os.path.exists(corpus_folder):
    print('Removing old corpus data folder...')
    shutil.rmtree(corpus_folder)
if not os.path.exists(corpus_folder):
    print('Generating new corpus data folder')
    os.mkdir(corpus_folder)

# Word embedding model
print('Saving word embedding model...')
model.save(model_name)
# Corpus vocabulary
print('Saving training corpus vocabulary into pickle file...')
with open(corpus_name, 'wb') as handle:
    pickle.dump(model_vocab_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Categories dictionary
print('Saving categories dictionary...')
with open(cat_dict_name, 'wb') as handle:
    pickle.dump(comp_keys, handle, protocol=pickle.HIGHEST_PROTOCOL)
