import os
import glob
import json
from bltk.langtools import Tokenizer, remove_stopwords
from bangla_stemmer.stemmer import stemmer
from collections import defaultdict

tokenizer = Tokenizer()
stemmer = stemmer.BanglaStemmer()

bengali_folder_path = os.path.join(os.getcwd(), "data/bengali")
file_paths = [
    f
    for f in glob.glob(os.path.join(bengali_folder_path, "**"), recursive=True)
    if os.path.isfile(f)
]
bengali_documents = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        bengali_documents.append(file.read())
document_count = len(bengali_documents)


def preprocess_document(document):
    tokens = tokenizer.word_tokenizer(document)
    tokens = remove_stopwords(tokens)
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def create_index(documents):
    index = defaultdict(
        lambda: {"term_frequency": 0, "document_frequency": 0, "posting_list": {}}
    )

    for doc_id, document in enumerate(documents):
        unique_terms = set(document)

        for term in unique_terms:
            index[term]["term_frequency"] += document.count(term)
            index[term]["document_frequency"] += 1
            index[term]["posting_list"][doc_id] = document.count(term)

    return index


def boolean_retrieval(query, index):
    query_terms = preprocess_document(query)
    result_docs = set(range(document_count))

    for term in query_terms:
        if term in index:
            result_docs = result_docs.intersection(
                set([int(doc_id) for doc_id in index[term]["posting_list"].keys()])
            )

    return result_docs


def write_index(index):
    with open("data/ben_index", "w") as file:
        file.write(json.dumps(index))


def load_index():
    with open("data/ben_index", "r") as file:
        return json.loads(file.read())


index = None
# Use the index if it exists, otherwise create the index
if os.path.exists("data/ben_index"):
    index = load_index()
else:
    preprocessed_documents = [preprocess_document(doc) for doc in bengali_documents]
    index = create_index(preprocessed_documents)
    write_index(index)

print("Boolean Retrieval System")
print("Enter 'exit' to quit the program\n")

while True:
    query = input("Enter your query: ")
    if query == "exit":
        break

    for term in query.split():
        if term in index:
            print(f"Term: {term}")
            print(f"Term Frequency: {index[term]['term_frequency']}")
            print(f"Document Frequency: {index[term]['document_frequency']}")
            print()
        else:
            print(f"Term: {term} not found in any document")
            print()

    result_documents = list(boolean_retrieval(query, index))
    print(f"Number of documents retrieved: {len(result_documents)}")

    # Display the middle content of atmax 5 documents
    for doc_id in result_documents[:5]:
        print(f"Document {doc_id + 1}: {bengali_documents[doc_id][100:200]}...")
        print()
