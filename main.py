import math

import numpy as np
import pandas as pd
from sklearn import svm
import re


def main():
    training_r, training_s, test_r, test_s = split_data()
    training_words = extract_features(training_r, 4, 100)
    word_occurences_pos, word_occurences_neg = feature_frequency(training_r, training_s, training_words)
    word_likelihoods, prior_pos, prior_neg = calculate_feature_likelihood(word_occurences_pos, word_occurences_neg, training_words, training_s)
    test_review = "This movie is horrible, I hated it !!!"
    test = classification(test_review, prior_pos, prior_neg, word_likelihoods)
    print(test)


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


# TASK 2
def extract_features(training_r, min_word_length, min_word_occurrence):
    word_occurrences = {}
    filtered_words = []

    # Remove non-alphanumeric values from reviews
    reviews = training_r.str.replace('[^a-zA-Z0-9\s]', '', regex=True)

    # Convert reviews to lowercase
    reviews = reviews.str.lower()

    # Split reviews into individual words
    words = reviews.to_string(index=False).split()

    # Extract words that meet requirements
    for word in words:
        if len(word) >= min_word_length:
            if word in word_occurrences:
                word_occurrences[word] = word_occurrences[word] + 1
            else:
                word_occurrences[word] = 1

    for word in word_occurrences:
        if word_occurrences[word] >= min_word_occurrence:
            filtered_words.append(word)

    return filtered_words


# TASK 3
def feature_frequency(training_r, training_s, training_words):
    word_occurences_pos = dict.fromkeys(training_words, 0)
    word_occurences_neg = dict.fromkeys(training_words, 0)
    positive_reviews = training_r[training_s == "positive"].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
    positive_reviews = positive_reviews.str.lower()
    negative_reviews = training_r[training_s == "negative"].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
    negative_reviews = negative_reviews.str.lower()

    is_in_review = False

    for review in positive_reviews:
        is_first_occurence = True
        review_words = review.split()
        for word in review_words:
            if word in training_words and is_first_occurence:
                is_in_review = True
                is_first_occurence = False
            if is_in_review:
                word_occurences_pos[word] += 1
                is_in_review = False

    for review in negative_reviews:
        is_first_occurence = True
        review_words = review.split()
        for word in review_words:
            if word in training_words and is_first_occurence:
                is_in_review = True
                is_first_occurence = False
            if is_in_review:
                word_occurences_neg[word] += 1
                is_in_review = False

    return word_occurences_pos, word_occurences_neg


# TASK 4
def calculate_feature_likelihood(word_occurences_pos, word_occurences_neg, unique_words, sentiments):
    word_likelihood = {}
    alpha = 1

    total_reviews = len(sentiments)
    total_pos_reviews = sentiments.value_counts()["positive"]
    total_neg_reviews = sentiments.value_counts()["negative"]

    for word in unique_words:
        word_count_pos = word_occurences_pos.get(word, 0)
        word_count_neg = word_occurences_neg.get(word, 0)

        # P[word is present in review[review is positive]]
        likelihood_pos = (word_count_pos + alpha) / (total_pos_reviews + 2 * alpha)
        # P[word is present in review[review is negative]]
        likelihood_neg = (word_count_neg + alpha) / (total_neg_reviews + 2 * alpha)

        word_likelihood[word] = {
            'positive': likelihood_pos,
            'negative': likelihood_neg
        }

    # Prior P[review is positive]
    prior_positive = total_pos_reviews / total_reviews

    # Prior P[review is negative]
    prior_negative = total_neg_reviews / total_reviews

    print("P[review is positive]: ", prior_positive)
    print("P[review is negative]: ", prior_negative)
    return word_likelihood, prior_negative, prior_negative


# TASK 5
def classification(review, prior_pos, prior_neg, likelihoods):
    likelihood_pos = 0.0
    likelihood_neg = 0.0

    # Split review into words + preprocessing
    processed_review = re.sub(r'[^a-zA-Z0-9\s]', '', review)
    words = processed_review.split()

    for word in words:
        if word in likelihoods:
            # Use logarithms for numerical stability
            likelihood_pos += math.log(likelihoods[word]['positive'])
            likelihood_neg += math.log(likelihoods[word]['negative'])

    # Apply Naive Bayesian Classification
    log_pos = math.log(prior_pos) + likelihood_pos
    log_neg = math.log(prior_neg) + likelihood_neg

    if log_pos > log_neg:
        return "positive"
    else:
        return "negative"


main()
