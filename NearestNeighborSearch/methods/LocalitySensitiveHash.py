import math

import methods.Helper as Helper
from methods.NeighborDistance import NeighborDistance
import util.EvalUtil as EvalUtil


class LocalitySensitiveHash(object):
    """
    @ivar documents: dict[int => dict[int => int/float]] list of documents
    @ivar D: int - dimension of documents / vectors (number of unique words)
    @ivar m: int - number of random projections
    @ivar projection_vectors: [[float]] - the projection vectors
    @ivar hashed_documents: dict[int => set(int)] - hash data structure for documents
    """

    def __init__(self, documents, D, m):
        """
        Creates a LocalitySensitiveHash with the specified dimension and number
        of random projections
        @param documents: dict[int => dict[int => int/float]] - the documents
        @param D: int - dimension
        @param m: int - number of projections / hashes
        """
        self.documents = documents
        self.D = D
        self.m = m
        self.projection_vectors = Helper.create_projection_vectors(D, m)
        self.build_hashed_documents()


    def build_hashed_documents(self):
        """
        Builds the hash table of documents.
        """
        self.hashed_documents = dict()
        for doc_id, document in self.documents.iteritems():
            lsh_bin = self.get_bin(document)
            if not self.hashed_documents.has_key(lsh_bin):
                self.hashed_documents[lsh_bin] = set()
            self.hashed_documents[lsh_bin].add(doc_id)


    def nearest_neighbor(self, document, depth):
        """
        Gets the (approximate) nearest neighbor to the given document
        @param document: dict[int => int/float] - a document
        @param depth: int - the maximum number of bits to change concurrently
        """
        hashed_document = self.hash_document(document)
        nearest = self._nearest_neighbor(document, hashed_document, None, depth, 0)
        return nearest


    def _nearest_neighbor(self, document, hashed_document, cur_nearest, depth, next_index):
        """
        Helper function to get the (approximate) nearest neighbor to the given document
        @param document: dict[int => int/float] - a document
        @param hashed_document: [bool] - hashed  document
        @param cur_nearest: NeighborDistance - the currently (approximately) closest neighbor
        @param depth: int - the maximum number of bits to change concurrently
        @param next_index: int - the next bin on which to potentially flip a bit
        """
        if depth < 0:
            return cur_nearest
        if cur_nearest is None:
            cur_nearest = NeighborDistance(0, float("inf"))
        self.check_bin(document, hashed_document, cur_nearest)
        if depth > 0:
            # check the bins one away from the current bin
            # if we still have more depth to go
            for j in xrange(next_index, self.m):
                hashed_document[j] = not hashed_document[j]
                self._nearest_neighbor(document, hashed_document, cur_nearest, depth - 1, j + 1)
                hashed_document[j] = not hashed_document[j]
        return cur_nearest


    def check_bin(self, document, hashed_document, cur_nearest):
        """
        Checks the documents that are hashsed to the given bin and updates with
        nearest neighbor found.
        @param document: dict[int => int/float] - list of documents
        @param hashed_document: [bool] - hashed document
        @param cur_nearest: NeighborDistance - the currently (approximately) nearest neighbor
        """
        # TODO: Fill in code for checking a bin for the nearest neighbor
        #       Code should look through all the documents in a bin and
        #       update cur_nearest with the nearest one found, if closer than cur_nearest already is
        raise Exception("Please implement the LocalitySensitiveHash.check_bin method")


    def get_bin(self, document):
        """
        Gets the bin where a document should be stored.
        @param document: dict[int => int/float] - a document
        """
        return self.convert_boolean_array_to_integer(self.hash_document(document))


    def hash_document(self, document):
        """
        Hashes a document to a boolean array using the set of projection vectors
        @param document: dict[int => int/float] - a document
        """
        hashed_document = [False for _ in xrange(self.m)]
        # TODO: fill in code for creating the hashed document
        raise Exception("Please implement the LocalitySensitiveHash.hash_document method")
        return hashed_document


    def project_document(self, document, vector):
        """
        Projects a document onto a projection vector for a boolean result.
        @param document: dict[int => int/float] - a document
        @param vector: [float] - a projection vector
        """
        dotprod = 0.0
        # TODO: fill in code for projecting the document
        raise Exception("Please implement the LocalitySensitiveHash.project_document method")
        return False


    def convert_boolean_array_to_integer(self, bool_array):
        """
        Converts a boolean array into the corresponding integer value.
        @param bool_array: [bool] - array of boolean values
        """
        value = 0
        for i, val in enumerate(bool_array):
            if val:
                value += math.pow(2, i)
        return value
