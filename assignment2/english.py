import os
import glob
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


def boolean_retrieval(query, index, documents):
    query_terms = preprocess_document(query)
    result_docs = set(range(len(documents)))

    for term in query_terms:
        if term in index:
            result_docs = result_docs.intersection(
                set(index[term]["posting_list"].keys())
            )

    return result_docs


preprocessed_documents = [preprocess_document(doc) for doc in english_documents]
index = create_index(preprocessed_documents)

print(
    f"Boolean Retrieval System | Index created successfully for {len(preprocessed_documents)} documents"
)
print("Enter 'exit' to quit the program\n")

while True:
    query = input("Enter your query: ")
    if query == "exit":
        break

    result_documents = boolean_retrieval(query, index, preprocessed_documents)
    for doc_id in result_documents:
        print(
            f"Document {doc_id + 1}: {english_documents[doc_id][:50]}..."
        )  # Displaying the first 50 characters of the document

    for term in query.split():
        if term in index:
            print(f"Term: {term}")
            print(f"Term Frequency: {index[term]['term_frequency']}")
            print(f"Document Frequency: {index[term]['document_frequency']}")
            print(f"Posting List: {index[term]['posting_list']}")
            print("=" * 30)
