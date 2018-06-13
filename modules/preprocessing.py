# Script containing preprocessing class
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale

from modules.get_ready import flatten
from modules.get_ready import text_processor

import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import numpy as np
import pandas as pd
import pdb
import collections
import pickle
import os


class preprocessing(object):
    """
    A dataframe object that has preprocessing capabilities
    """
    def __init__(self, df, train_vocab, categs_dict):
        """Initializes class parameters"""
        super()
        self.df = df
        self.categs_dict = categs_dict
        self.train_vocab = train_vocab
        self.replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        self.tSVD = TruncatedSVD(n_components=100)
        self.pca_model = PCA(n_components=800)


    @staticmethod
    def id_parser(id_list):
        """Function for parsing ID values"""
        num_list = np.asarray([int(elem[1:]) for elem in id_list], dtype=np.float64)
        scaled_list = num_list/np.max(num_list)
        return(scale(scaled_list))


    def convert_to_index(self, sentence):
        """Method for transforming categories by lookup indexes"""
        placeholder = np.zeros(len(self.categs_dict))
        for word in sentence:
            placeholder[self.categs_dict.index(word)] += 1
        return(placeholder)


    def vocab_validate(self, corpus):
        """Function for cross-referencing corpus words with train_vocab words
        Returns: list of documents that have been cross-referenced with training vocabulary"""
        # Remove punctuation and uncase
        processed_corpus = list(map(lambda x: text_processor(x, self.replace_punctuation), corpus))
        processed_corpus = [' '.join(group) for group in processed_corpus]
        # Join separate documents together
        master_string = ' | '.join(processed_corpus)
        # Split master string into individual words
        split_string = np.asarray([w for w in master_string.split() if w!=' ' or w!=None])
        # Assemble counts in split string
        corpus_count = Counter(split_string)
        corpus_idx_loc = {}
        for key in corpus_count.keys():
            corpus_idx_loc[key] = []
        for idx, word in enumerate(split_string):  # Construct index locations of words
            corpus_idx_loc[word].append(idx)
        # Find out which words to remove
        remove_words = [w  for w in list(corpus_idx_loc.keys()) if w not in self.train_vocab]
        # Remove words
        idx_mask = np.ones(len(split_string), dtype=bool)
        remove_words.remove('|')
        for word in remove_words:
            idx_mask[corpus_idx_loc[word]] = False
        split_string = ' '.join(split_string[idx_mask]).split('|')  # Apply boolean mask and split by pipe
        return([doc.strip() for doc in split_string if doc!=' ' or doc!=None])


    def tfidf_fit(self):
        """Fits a model of the tfidf transform"""
        review_and_summary = list(zip(self.df.reviewText.values, self.df.summary.values))
        corpus = [' '.join(x) for x in review_and_summary]
        validated_corpus = self.vocab_validate(corpus)
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True,
                                     use_idf=True, smooth_idf=True)
        vectorizer.fit(validated_corpus)
        return(vectorizer)


    def word_embed(self, model):
        """Creates an embedding of words in the corpus"""
        review_and_summary = list(zip(self.df.reviewText.values, self.df.summary.values))
        corpus = [' '.join(x) for x in review_and_summary]
        validated_corpus = self.vocab_validate(corpus)
        embedded_docs = np.zeros((len(validated_corpus), 50))
        for i, doc in enumerate(validated_corpus):
            trans_doc = np.zeros((len(doc), 50))
            for j, word in enumerate(doc.split()):
                trans_doc[j] = model[word]
            embedded_docs[i] = np.nanmean(trans_doc, axis=0)
        return(embedded_docs)


    def category_id(self):
        """Return the values of **category_id** as a list"""
        return(np.asarray(self.df.categoryID.values))


    def parse_categories(self, categs):
        """Method for parsing given list of categories"""
        flat_categs = [list(flatten(group)) for group in categs]
        categs_list = []
        for group in flat_categs:
            group_list = []
            for sentence in group:
                for word in sentence.translate(self.replace_punctuation).split():
                    group_list.append(word)
            categs_list.append(group_list)
        return(categs_list)


    def categories(self):
        """Returns a binarized representation of the categories column"""
        categs = self.df.categories.values
        categs_list = self.parse_categories(categs)
        master_placeholder = np.zeros((len(categs_list), len(self.categs_dict)))
        for i, sentence in enumerate(categs_list):
            master_placeholder[i, :] = self.convert_to_index(sentence)
        # Perform a PCA dimensionality reduction
        transformed_cats = self.pca_model.fit_transform(master_placeholder)
        return(transformed_cats)


    def item_id(self):
        """Method for extracting item ID from raw dataframe"""
        return(self.id_parser(self.df.itemID.values))


    def reviewer_id(self):
        """Method for extracting reviewer ID from raw dataframe"""
        return(self.id_parser(self.df.reviewerID.values))


    def rating(self):
        """Returns the values of **rating** as a list"""
        return(np.asarray(self.df.rating.values))


    def review_text(self):
        """Transforms the review text into a vector representation using the TFIDF and PCA"""
        vectorizer = self.tfidf_fit()
        review_text = self.vocab_validate(list(self.df.reviewText.values))
        review_text = vectorizer.transform(review_text)
        return(self.tSVD.fit_transform(review_text))


    def review_length(self):
        """Finds the length of a review and returns a list of all lengths"""
        review_text = list(self.df.reviewText.values)
        return(np.asarray([len(x) for x in review_text]))


    def review_time(self):
        """Translates the time string into 3-element array"""
        review_time = list(self.df.reviewTime.values)
        review_time = [text_processor(date, self.replace_punctuation) for date in review_time]
        return(np.asarray(list(map(lambda x: [int(i) for i in x], review_time))))


    def helpful(self):
        """Filters out the **outOf** entry and returns a list of values"""
        helpful = list(self.df.helpful.values)
        out_of = np.asarray([x['outOf'] for x in helpful], dtype=np.int32)
        return(out_of)


    def save_data(self, data, dname, fname):
        """Saves data into desired directory in pickle format"""
        if not os.path.exists(dname):
            os.mkdir(dname)
        fpath = dname + fname
        with open(fpath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_data(self, dname, fname):
        """Loads pickle data from desired directory"""
        fpath = dname + fname
        with open(fpath, 'rb') as handle:
            data = pickle.load(handle)
        return(data)


    def process_data(self, dname, fname, model):
        """Performs all calculations to generate raw data only"""
        data = [self.category_id(), self.rating(), self.review_length(), self.review_time(), self.helpful()]
        # data = [self.category_id(), self.categories(), self.item_id(), self.reviewer_id(), self.rating(),
        #         self.review_text(), self.word_embed(model), self.review_length(), self.review_time(), self.helpful()]
        self.save_data(data, dname, fname)
        print('Raw data has been generated and saved!')


    def row_len_calc(self, data):
        """Method for calculating a length of a row"""
        row_sum = 0
        for group in data:
            if len(group.shape)>1:
                row_sum += group.shape[1]
            else:
                row_sum += 1
        return(row_sum)


    def assemble_dataset(self, dname, f_raw, f_final):
        """Returns the assembled processed dataset"""
        data = self.load_data(dname, f_raw)
        row_sum = self.row_len_calc(data)

        processed_list = np.zeros((self.df.shape[0], row_sum), dtype=np.float16)
        for i in range(self.df.shape[0]):
            col_traverse = 0
            for group in data:
                if type(group[i]) == np.ndarray or type(group[i]) == list:
                    for elem in group[i]:
                        processed_list[i, col_traverse] = np.float16(elem)
                        col_traverse += 1
                else:
                    processed_list[i, col_traverse] = np.float16(group[i])
                    col_traverse += 1
        processed_df = pd.DataFrame(data=processed_list, index=None, columns=[i for i in range(row_sum)])
        processed_df.to_csv(dname + f_final)
        print('Processed data has been assembled and saved!')


    def true_scores(self, dname, score_name):
        """Returns the **nHelpful** metric from the **helpful** entries and saves the list of values"""
        true_scores = self.df.helpful.tolist()
        true_scores = np.asarray([x['nHelpful'] for x in true_scores])
        self.save_data(true_scores, dname, score_name)
        print('True scores have been assembled and saved!')
