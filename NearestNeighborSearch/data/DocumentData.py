def read_in_data(filename, has_header):
    """
    Reads in a data set
    @param filename: string - name of file (with path) read
    @param has_header: boolean - whether the file has a header
    """
    documents = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i < 2 and has_header:
                continue
            parse_in_line(line, documents)
    return documents


def parse_in_line(line, documents):
    """
    Reads in line from .mtx file into documents hash.
    @param line: string - line to parse
    @param documents: dict - hash of documents (doc id -> term id -> count)
    """
    fields = line.split(" ")
    if len(fields) != 3:
        raise Exception("Matrix line has incorrect number of fields: " + line)
    term_id = int(fields[0])
    doc_id = int(fields[1])
    count = int(fields[2])
    if not documents.has_key(doc_id):
        documents[doc_id] = dict()
    documents[doc_id][term_id] = count
