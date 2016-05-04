import numpy as np
import pandas as pd
import os

PATH_TO_BBC = "../../data/bbc_data"

class BBCData():
    def __init__(self, path = PATH_TO_BBC):
        self.path = path
        self.freq_mtx  = pd.read_csv(os.path.join(self.path, "bbc.mtx"), sep = " ")
        self.freq_mtx.termid = [int(fl) for fl in self.freq_mtx.termid]
        self.freq_mtx.docid  = [int(fl) for fl in self.freq_mtx.docid]
        self.docids = set()
        self.termids = set()
        self.vocab = [word.strip() for word in open(os.path.join(self.path, "bbc.terms")).readlines()]
        self.get_tfidf_and_sizes()
        self.class_mtx = []
        self.num_classes = 0
        self.initialize_classes()
        self.avg_tfidf_by_class = []
        self.five_best = self.get_tfidf_by_class()
        return

    def get_tfidf_and_sizes(self):
        """
        converts pd dataframe self.tfidf_mtx, which has rows of the form
            termid docid frequency
        into an array with rows
            termid docid tfidf
        """

        #Get basic summary stats
        doc_count_by_term = {} # This measures "common-ness" of each term: +1 for each document containing the term.
        doc_max_freqs = {}
        for i, row in self.freq_mtx.iterrows():
            (termid, docid, frequency) = [self.freq_mtx.iloc[i, j] for j in range(3)]
            #if new termids term, add
            if not doc_count_by_term.has_key(termid):
                self.termids.add(int(termid))
                doc_count_by_term[termid] = 0
            #if new document, add to array
            if not doc_max_freqs.has_key(docid):
                self.docids.add(int(docid))
                doc_max_freqs[docid] = 0
            #Increment counts by one new document with this term or by (possibly
            # several) occurrences of this term in this doc
            doc_count_by_term[termid] += 1
            doc_max_freqs[docid] = max((frequency, doc_max_freqs[docid]))

        self.freq_mtx.tfidf = 0
        for i, row in self.freq_mtx.iterrows():
            (termid, docid, frequency) = [self.freq_mtx.iloc[i, j] for j in range(3)]

            # inverse document frequency: log of (number of documents - number of documents with term)
            idf = np.log(len(self.docids) / float(doc_count_by_term[termid]))

            # term frequency: frequency in the document, normalized so that the most common term has a tf of 1.
            # In other words, divide by the max over words of the word frequencies.
            tf = frequency / float(doc_max_freqs[docid])
            self.freq_mtx.loc[i, "tfidf"] = tf * idf
        return

    def initialize_classes(self):
        self.class_mtx = pd.read_csv(os.path.join(self.path, "bbc.classes"), sep = " ")
        self.num_classes = len(set(self.class_mtx.class_label))
        return

    def get_tfidf_by_class(self):
        """
        averages tfidf over the classes, but using only one pass through the matrix market format data.
        """
        self.avg_tfidf_by_class = pd.DataFrame(data = 0, index = self.termids, columns = range(self.num_classes))
        for i, row in self.freq_mtx.iterrows():
            (termid, docid, freq, tfidf) = [self.freq_mtx.iloc[i, j] for j in range(4)]
            class_label = self.class_mtx.loc[docid - 1, "class_label"]
            assert (docid == self.class_mtx.loc[docid - 1, "docid"])
            self.avg_tfidf_by_class.iloc[int(termid - 1), class_label] += tfidf

        # divide sum by num to get avg
        for c in range(self.num_classes):
            num_docs_in_c = sum(int(george) for george in (self.class_mtx.class_label == c))
            self.avg_tfidf_by_class[[c]] = self.avg_tfidf_by_class[[c]] / num_docs_in_c

        # sort to get top 5 in class
        five_best = [0 for c in range(self.num_classes)]
        for c in range(self.num_classes):
            termids_sorted = self.avg_tfidf_by_class.sort(c, ascending = False).index.copy()
            tfidfs_sorted = self.avg_tfidf_by_class.sort(c, ascending = False)[[c]].copy()
            bestpairs = [(termids_sorted[rank], float(tfidfs_sorted.iloc[rank])) for rank in range(5)]
            five_best[c] = [(self.vocab[termid - 1], tfidf) for (termid, tfidf) in bestpairs]
        return five_best

    def get_dense_data(self):
        """
        :return: pd dataframe where each row corresponds to a document.
         class labels reside in the first column, and tfidf values indexed by term id's thereafter.
        """
        dense_data = pd.DataFrame(0, index = list(self.docids), columns = list(self.termids))
        for i, row in self.freq_mtx.iterrows():
            (termid, docid, freq, tfidf) = [self.freq_mtx.iloc[i, j] for j in range(4)]
            dense_data.loc[docid, termid] = tfidf
        assert dense_data.shape == (len(self.docids), len(self.termids))
        return dense_data