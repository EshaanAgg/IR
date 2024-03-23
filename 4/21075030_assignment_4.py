import sys


# This class is responsible for loading the relevance information from the qrels file
class RelevanceInfo:
    def __init__(self, qrels_file):
        self.qrels_file = qrels_file
        self.relevance_info = self.load_relevance_info()

    # Reads the qrels file and returns a dictionary with the relevance information
    # The key is the query number and the value is another dictionary with the document number as key and the relevance as value (0 or 1)
    def load_relevance_info(self):
        relevance_info = {}
        with open(self.qrels_file, "r") as qrels:
            for line in qrels:
                query_num, _, doc_num, relevance = line.split()
                if query_num not in relevance_info:
                    relevance_info[query_num] = {}
                relevance_info[query_num][doc_num] = int(relevance)
        return relevance_info

    # Returns a list with the relevant documents for a given query number
    def get_relevant_docs(self, query_number):
        if query_number not in self.relevance_info:
            print(f"Query {query_number} not found in relevance info")
            sys.exit(1)

        relevant_docs = []
        for doc, relevance in self.relevance_info[query_number].items():
            if relevance > 0:
                relevant_docs.append(doc)
        return relevant_docs


# This class is responsible for loading the retrieval results from the res file
class RetrievalResults:
    def __init__(self, res_file):
        self.res_file = res_file
        self.retrieved_results = self.load_retrieval_results()

    # Reads the res file and returns a dictionary with the retrieval results
    # The key is the query number and the value is a list with the document numbers
    def load_retrieval_results(self):
        retrieval_results = {}
        with open(self.res_file, "r") as res:
            for line in res:
                query_num, _, doc_num, _, _, _ = line.split()
                if query_num not in retrieval_results:
                    retrieval_results[query_num] = []
                retrieval_results[query_num].append(doc_num)
        return retrieval_results

    # Returns a list with the retrieved documents for a given query number
    def get_retrieved_docs(self, query_number):
        if query_number not in self.retrieved_results:
            print(f"Query {query_number} not found in retrieval results")
            sys.exit(1)

        return self.retrieved_results[query_number]


# This class is responsible for evaluating the retrieval results for a given query
class QueryEvaluator:
    def __init__(self, res_file, qrels_file, query_number):
        self.retrievedResults = RetrievalResults(res_file)
        self.relevanceInfo = RelevanceInfo(qrels_file)
        self.query_number = query_number

    # Computes the average precision for the given query
    def compute_average_precision(self):
        relevant_docs = self.relevanceInfo.get_relevant_docs(self.query_number)
        retrieved_docs = self.retrievedResults.get_retrieved_docs(self.query_number)

        retrieved_relevant_docs = 0
        count_retrieved_docs = 0
        sum_precision = 0

        for doc in retrieved_docs:
            # For every retrieved document, we check increment the count of retrieved documents
            count_retrieved_docs += 1
            if doc in relevant_docs:
                # If the retrieved document is relevant, we increment the count of retrieved relevant documents
                # and add the instantaneous precision to the sum of precisions
                retrieved_relevant_docs += 1
                sum_precision += retrieved_relevant_docs / count_retrieved_docs

        total_relevant_docs = len(relevant_docs)

        # Print some metadata about the query results
        print(f"Total retrieved docs for the query\t\t: {count_retrieved_docs}")
        print(f"Total relevant docs for the query\t\t: {total_relevant_docs}")
        print(f"Retrieved relevant docs for the query\t\t: {retrieved_relevant_docs}")
        print()

        # The average precision is the sum of precisions divided by the total number of relevant documents
        return sum_precision / total_relevant_docs


if __name__ == "__main__":
    # Check if the number of arguments is correct
    if len(sys.argv) != 4:
        print("Usage: python program.py res_file qrels_file query_number")
        sys.exit(1)

    res_file = sys.argv[1]
    qrels_file = sys.argv[2]
    query_number = sys.argv[3]

    # Create a QueryEvaluator object and compute the average precision for the given query
    evaluator = QueryEvaluator(res_file, qrels_file, query_number)
    ap = evaluator.compute_average_precision()
    print(f"Average Precision (AP) for query {query_number}\t\t: {ap}")
