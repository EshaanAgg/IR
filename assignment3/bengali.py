import os
import glob
import json
from math import log10
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from bangla_stemmer.stemmer import stemmer
from bltk.langtools import Tokenizer, remove_stopwords

tokenizer = Tokenizer()
stemmer = stemmer.BanglaStemmer()

bengali_folder_path = os.path.join(os.getcwd(), "data/bengali")
file_paths = [
    f
    for f in glob.glob(os.path.join(bengali_folder_path, "**"), recursive=True)
    if os.path.isfile(f)
][:1000]

bengali_documents = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        bengali_documents.append(file.read())

document_count = len(bengali_documents)


def preprocess_document(document):
    tokens = tokenizer.word_tokenizer(document.lower())
    tokens = remove_stopwords(tokens)
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def create_index(preprocessed_documents):
    # Calculate the term frequency and document frequency for each term
    index = defaultdict(lambda: {"document_frequency": 0, "posting_list": {}})

    for doc_id, doc in enumerate(preprocessed_documents):
        term_frequency = defaultdict(int)
        for term in doc:
            term_frequency[term] += 1

        for term, frequency in term_frequency.items():
            index[term]["document_frequency"] += 1
            index[term]["posting_list"][str(doc_id)] = frequency

    return index


def get_weight_vector_document(index, doc_id):
    weight_vector = {}
    for term in index:
        if str(doc_id) in index[term]["posting_list"]:
            weight_vector[term] = (
                1 + log10(index[term]["posting_list"][str(doc_id)])
            ) * log10(document_count / index[term]["document_frequency"])
        else:
            weight_vector[term] = 0

    return weight_vector


def get_weight_vector_query(index, query_terms):
    weight_vector = {}
    for term in index:
        if term in query_terms:
            weight_vector[term] = (1 + log10(query_terms.count(term))) * log10(
                document_count / index[term]["document_frequency"]
            )
        else:
            weight_vector[term] = 0

    return weight_vector


def cosine_similarity(v1, v2):
    dot_product = sum(v1[term] * v2[term] for term in v1)
    magnitude_v1 = sum(v1[term] ** 2 for term in v1) ** 0.5
    magnitude_v2 = sum(v2[term] ** 2 for term in v2) ** 0.5

    if (magnitude_v1 * magnitude_v2) == 0:
        return 0

    return dot_product / (magnitude_v1 * magnitude_v2)


def vector_space_model_retrieval(query, index):
    query_terms = preprocess_document(query)
    weight_vector_query = get_weight_vector_query(index, query_terms)

    similarity_scores = {}
    for doc_id in range(document_count):
        weight_vector_document = get_weight_vector_document(index, doc_id)
        similarity_scores[doc_id] = cosine_similarity(
            weight_vector_query, weight_vector_document
        )

    return similarity_scores


def write_index(index):
    with open("data/bng_index_vsm", "w") as file:
        file.write(json.dumps(index))


def load_index():
    with open("data/bng_index_vsm", "r") as file:
        return json.loads(file.read())


index = None
# Use the index if it exists, otherwise create the index
if os.path.exists("data/bng_index_vsm"):
    index = load_index()
else:
    preprocessed_documents = [preprocess_document(doc) for doc in bengali_documents]
    index = create_index(preprocessed_documents)
    write_index(index)

print("Vector Space Model Retrieval System")
print("Enter 'exit' to quit the program\n")

while True:
    query = input("Enter your query: ")
    if query == "exit":
        break

    # Print the top 5 documents with the highest similarity scores
    similarity_scores = vector_space_model_retrieval(query, index)
    top_5_docs = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:5]
    for doc_id in top_5_docs:
        print(f"Document {doc_id + 1} - Similarity Score: {similarity_scores[doc_id]}")
