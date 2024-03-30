import os
import sys
import zipfile
import numpy as np
from collections import defaultdict

# Delimeters to skip while reading the file
delimeters_to_skip = [
    "Newsgroups:",
    "Path:",
    "From:",
    "Subject:",
    "Message-ID:",
    "Sender:",
    "Nntp-Posting-Host:",
    "Organization:",
    "References:",
    "Distribution:",
    "Date:",
    "Lines:",
]


class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    # Reads the file, skip the lines starting with delimeters_to_skip, and return the token frequency
    def get_token_frequency(self):
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        token_frequency = defaultdict(int)
        for line in lines:
            if any([line.startswith(delim) for delim in delimeters_to_skip]):
                continue

            tokens = line.lower().split()
            for token in tokens:
                token_frequency[token] += 1

        return token_frequency


class NaiveBayesClass:
    def __init__(self, class_name, class_path):
        self.class_name = class_name
        self.class_files = self.get_class_files(class_path)
        self.doc_count = len(self.class_files)

    def get_class_files(self, class_path):
        class_files = []
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            class_files.append(FileReader(file_path))
        return class_files

    def get_vocabulary(self):
        vocabulary = set()
        for file in self.class_files:
            vocabulary.update(file.get_token_frequency().keys())
        return vocabulary

    def train(self, vocabulary, total_doc_count):
        self.prior = self.doc_count / total_doc_count
        vocab_size = len(vocabulary)
        self.conditional_prob = defaultdict(lambda: 1 / vocab_size)

        file_token_frequencies = [
            file.get_token_frequency() for file in self.class_files
        ]

        total_token_count = {}
        total_count = vocab_size
        for token in vocabulary:
            total_token_count[token] = sum(
                [tf[token] for tf in file_token_frequencies if token in tf]
            )
            total_count += total_token_count[token]

        for token in vocabulary:
            self.conditional_prob[token] = (total_token_count[token] + 1) / (
                total_count
            )

    def predict(self, token_frequency):
        prob = np.log(self.prior)
        for token in token_frequency:
            prob += np.log(self.conditional_prob[token]) * token_frequency[token]
        return prob


class NaiveBayesClassifier:
    def __init__(self, root_dir, eval_dir):
        self.root_dir = root_dir
        self.eval_dir = eval_dir

        self.classes = self.get_classes()
        self.class_count = len(self.classes)
        self.total_doc_count = sum([cls.doc_count for cls in self.classes])
        self.vocabulary = self.get_vocabulary()

        print(f"Total documents: {self.total_doc_count}")
        print(f"Vocabulary size: {len(self.vocabulary)}")

        for index, cls in enumerate(self.classes):
            print(f"[{index+1}/{self.class_count}] Training class {cls.class_name}")
            cls.train(self.vocabulary, self.total_doc_count)

    def get_classes(self):
        classes = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            classes.append(NaiveBayesClass(class_name, class_path))
        return classes

    def get_vocabulary(self):
        vocabulary = set()
        for cls in self.classes:
            vocabulary.update(cls.get_vocabulary())
        return vocabulary

    def predict_file(self, file_path):
        file = FileReader(file_path)
        max_prob = float("-inf")
        max_class = None

        for cls in self.classes:
            prob = cls.predict(file.get_token_frequency())
            if prob > max_prob:
                max_prob = prob
                max_class = cls.class_name

        return max_class

    def predict(self):
        predictions = []
        for file_name in os.listdir(self.eval_dir):
            file_path = os.path.join(self.eval_dir, file_name)
            predictions.append((self.predict_file(file_path), file_name))
        return predictions


# Utitlity functions


# Read the files from a folder, batch them into groups of 40, and write the nth batch to a zip file
def batch_files_and_zip(root_dir, zip_file_path, n):
    files = os.listdir(root_dir)
    files.sort(key=lambda x: int(x.split("-")[0]))

    batch_size = 40
    num_batches = len(files) // batch_size

    # Write the 23rd batch to a zip file
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        for i in range(num_batches):
            if i == n - 1:
                for j in range(batch_size):
                    file_path = os.path.join(root_dir, files[i * batch_size + j])
                    zipf.write(file_path, os.path.basename(file_path))


# batch_files_and_zip("data/test", "data/21075030_test_file.zip", 23)

if __name__ == "__main__":
    # Get the test zip file path and output file path from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python main.py <test_zip_file_path> <output_file_path>")
        sys.exit()

    test_zip_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Unzip the test file
    with zipfile.ZipFile(test_zip_file_path, "r") as zipf:
        zipf.extractall("test")

    # NOTE: We assume that the training data is present in the "data/newsgroups" folder
    classifier = NaiveBayesClassifier("data/newsgroups", "test")

    # Predict the probabilities and the class for each file, and write the output to a file
    predictions = classifier.predict()
    with open(output_file_path, "w") as f:
        for prediction, file_name in predictions:
            f.write(f"{file_name}, {prediction}\n")
