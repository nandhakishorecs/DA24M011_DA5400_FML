# ------------------------------------------ Naive Bayes Classifier ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 2
#
#   This file contains code for implementing preprocessing techniques for text data. There are two subclasses
#   in the preprocessing class:  
#               (1) TF IDF Vectoriser 
#               (2) Label Encoder 
#
#   Description of functions in the class 'TFIDFVectoriser':
#
#       - The inputs for the classes are as follows: 
#           *   X: dense matrix with text (np.ndarray)
#           *   When called, the class is initialised with ENGLISH stopwords from sklearn library.
#
#       - fit (X) 
#           *   creates a sparse matrix from a dense numpy array which contains text 
#           *   the conversion is by calculating the Inverse Document Frequency of the tokens in the corpus 
#           *   while conversion, the stop words (common repetitive words), punctuation marks are removed. 
#           *   Once the term frequency (TF) and inverse document frequencies (IDF) are done, the mappings 
#               are stored in a sparse matrix in a compressed row format. 
#
#       - transform(X):
#           *   returns the sparse matrix for a dense text matrix for the given data matrix. It doesn't 
#               changes the data in place. 
#
#       - fit_transform(X):
#           *   returns the sparse matrix for a dense text matrix for the given data matrix. Data is stored
#               as a sparse matrix directly. 
#   
#   Description of functions in the class 'Label Encoder':
#
#       - The inputs for the classes are as follows: 
#           *   y: array with text strings (np.ndarray)
#
#       - fit (X) 
#           *   creates a numpy array with numbers from 0 to n-1 entries 
#           *   For given unique names of classes, the class names are mapped to numbers form 0 to n-1. 
#
#       - transform(X):
#           *   returns the encoded array for a label vector
#
#       - fit_transform(X):
#           *   returns the encoded array for a for the label vector, where the encoding is done inplace. 

# --------------------------------------------------------------------------------------------------------------

# ------------------------------------------- Importing Libraries --------------------------------------------
import numpy as np	                                           # type: ignore
import scipy.sparse	                                           # type: ignore
from scipy.sparse import csr_matrix	                           # type: ignore
from collections import Counter	                               # type: ignore
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS # type: ignore

class Preprocessing: 
    
    class TFIDF_Vectorizer:
        __slots__ = '_max_features', '_min_frequency', '_stop_words', '_vocabulary', '_idf_vector'
        def __init__(self, max_features=None, min_freq=1, stop_words="english"):
            self._max_features = max_features
            self._min_frequency = min_freq
            self._stop_words = ENGLISH_STOP_WORDS if stop_words == "english" else set()
            self._vocabulary = {}
            self._idf_vector = None

        def fit(self, documents: np.ndarray) -> None:
            # Preprocess documents - Remove stop words, convert all characters to lower case and remove punctuation
            preprocessed_docs = [
                [word.lower() for word in doc.split() if word.lower() not in self._stop_words]
                for doc in documents
            ]
            
            # Calculate document frequencies
            doc_freq = Counter()
            for doc in preprocessed_docs:
                unique_terms = set(doc)
                for term in unique_terms:
                    doc_freq[term] += 1

            # Filter terms by min_freq
            vocab = {
                term for term, freq in doc_freq.items() if freq >= self._min_frequency
            }

            # Limit features based on max_features
            if self._max_features:
                vocab = dict(Counter(doc_freq).most_common(self._max_features))

            # Create vocabulary index
            self._vocabulary = {term: i for i, term in enumerate(vocab)}
            
            # Calculate IDF
            num_docs = len(documents)
            self._idf_vector = np.log(num_docs / (1 + np.array([doc_freq[term] for term in self._vocabulary])))
            
            return self

        def transform(self, documents: np.ndarray) -> scipy.sparse.csr_matrix:
            # Initialize data for sparse matrix
            rows, cols, values = [], [], []

            for i, doc in enumerate(documents):
                terms = [word.lower() for word in doc.split() if word.lower() not in self._stop_words]
                term_counts = Counter(terms)

                for term, count in term_counts.items():
                    if term in self._vocabulary:
                        # TF calculation (term frequency)
                        tf = count / len(terms)
                        # TF-IDF calculation (inverse document frequency)
                        tfidf = tf * self._idf_vector[self._vocabulary[term]]
                        rows.append(i)
                        cols.append(self._vocabulary[term])
                        values.append(tfidf)
                        
            # Create sparse matrix
            return csr_matrix((
                values, 
                (rows, cols)), 
                shape = (len(documents), 
                len(self._vocabulary)
            ))

        def fit_transform(self, documents: np.ndarray) -> scipy.sparse.csr_matrix:
            self.fit(documents)
            return self.transform(documents)
    
    class LabelEncoder:
        __slots__ = '_label_to_num', '_num_to_label', '_classes_'
        def __init__(self) -> None:
            self._label_to_num = {}
            self._num_to_label = {}
            self._classes_ = []
            return None

        def fit(self, labels:np.ndarray) -> None:
            unique_labels = set(labels)
            self._classes_ = sorted(unique_labels)
            # Map class names to numbers from 0 - n-1 for 'n' classes
            self._label_to_num = {label: index for index, label in enumerate(self._classes_)}
            self._num_to_label = {index: label for index, label in enumerate(self._classes_)}
            return self

        def transform(self, labels:np.ndarray) -> np.ndarray:
            return np.array([self._label_to_num[label] for label in labels])

        def fit_transform(self, labels:np.ndarray) -> np.ndarray:
            self.fit(labels)
            return self.transform(labels)

        def inverse_transform(self, labels:np.ndarray) -> np.ndarray:
            return np.array([self._num_to_label[idx] for idx in labels])    