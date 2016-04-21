import random


def random_dataset(n, D):
    """
    Creates a random dataset with points independently uniform on [-0.5, 0.5] in each dimension
    @param n: int - the number of points to create
    @param D: int - the number of dimensions to use
    """
    documents = dict()
    for doc_id in xrange(n):
        document = dict()
        for word_id in xrange(D):
            document[word_id] = random.random() - 0.5
        documents[doc_id + 1] = document
    return documents
