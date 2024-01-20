import os
import glob
from collections import Counter
import matplotlib.pyplot as plt


def calculate_zipf_law(folder_path):
    word_frequencies = Counter()

    # Recursively search for all files in the given folder
    file_paths = [
        f
        for f in glob.glob(os.path.join(folder_path, "**"), recursive=True)
        if os.path.isfile(f)
    ]

    print(f"{len(file_paths)} files found.")

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read().lower().split()
                word_frequencies.update(content)
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")

    # Sort word frequencies in descending order
    sorted_frequencies = sorted(word_frequencies.values(), reverse=True)

    # Plot Zipf's law
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(sorted_frequencies) + 1),
        sorted_frequencies,
        marker="o",
        linestyle="-",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()


folder_path = os.path.join(os.getcwd(), "data/english")
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    calculate_zipf_law(folder_path)
else:
    print("Invalid folder path.")
