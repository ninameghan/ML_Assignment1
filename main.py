import numpy as np
import pandas as pd
from sklearn import svm


def main():
    training_r, training_s, test_r, test_s = split_data()


# TASK 1
def split_data():
    data = pd.read_excel("movie_reviews.xlsx")

    # Training data
    training_data = data[data["Split"] == "train"]
    training_reviews = training_data["Review"]
    training_sentiments = training_data["Sentiment"]

    # Test data
    test_data = data[data["Split"] == "test"]
    test_reviews = test_data["Review"]
    test_sentiments = test_data["Sentiment"]

    print("Positive reviews in training set: ", training_sentiments.value_counts()["positive"])
    print("Negative reviews in training set: ", training_sentiments.value_counts()["negative"])
    print("Positive reviews in test set: ", test_sentiments.value_counts()["positive"])
    print("Negative reviews in test set: ", test_sentiments.value_counts()["negative"])

    return training_reviews, training_sentiments, test_reviews, test_sentiments


main()
