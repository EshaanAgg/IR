import os
import glob
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# Download the NLTK data for tokenization and stopwords
import nltk

nltk.download("punkt")
nltk.download("stopwords")

english_folder_path = os.path.join(os.getcwd(), "data/english")
file_paths = [
    f
    for f in glob.glob(os.path.join(english_folder_path, "**"), recursive=True)
    if os.path.isfile(f)
]
english_documents = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        english_documents.append(file.read())
document_count = len(english_documents)


def preprocess_document(document):
    tokens = word_tokenize(document.lower())

    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    porter_stemmer = PorterStemmer()
    tokens = [porter_stemmer.stem(token) for token in tokens]

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
    with open("data/eng_index", "w") as file:
        file.write(json.dumps(index))


def load_index():
    with open("data/eng_index", "r") as file:
        return json.loads(file.read())


index = None
# Use the index if it exists, otherwise create the index
if os.path.exists("data/eng_index"):
    index = load_index()
else:
    preprocessed_documents = [preprocess_document(doc) for doc in english_documents]
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

    # Display the matched content of atmax 5 documents
    for doc_id in result_documents[:5]:
        lines = english_documents[doc_id].split("\n")
        for line in lines:
            words = line.split()
            if len(set(words).intersection(set(query.split()))) != 0:
                print("Document ID:", doc_id)
                print(f"Content: ...{line}...\n")
                break
