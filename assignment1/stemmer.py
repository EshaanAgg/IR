from bangla_stemmer.stemmer import stemmer
import os
import glob


def get_stemmed_count(folder_path):
    # Recursively search for all files in the given folder
    file_paths = [
        f
        for f in glob.glob(os.path.join(folder_path, "**"), recursive=True)
        if os.path.isfile(f)
    ]

    print(f"{len(file_paths)} files found.")

    word_list = set()
    stemmed_word_list = set()
    word_count = 0

    stmr = stemmer.BanglaStemmer()

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read().lower().split()
                word_list.update(content)
                word_count += len(content)
                for word in content:
                    stemmed_word_list.add(stmr.stem(word))
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")

    print(f"Total words: {word_count}")
    print(f"Unique words: {len(word_list)}")
    print(f"Unique stemmed words: {len(stemmed_word_list)}")


folder_path = os.path.join(os.getcwd(), "data/bengaliNews")
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    get_stemmed_count(folder_path)
else:
    print("Invalid folder path.")
