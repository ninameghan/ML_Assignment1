import math

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import re


def main():
    # TASK 1
    training_r, training_s, test_r, test_s = split_data()

    # # TASK 2
    # training_words = extract_features(training_r, 4, 100)

    # # TASK 3
    # word_occurences_pos, word_occurences_neg = feature_frequency(training_r, training_s, training_words)

    # # TASK 4
    # word_likelihoods, prior_pos, prior_neg = calculate_feature_likelihood(word_occurences_pos, word_occurences_neg, training_words, training_s)

    # # TASK 5
    # test_review = "This movie is great, I loved it !!!"
    # test = classification(test_review, prior_pos, prior_neg, word_likelihoods)
    # print(test)

    # TASK 6
    # Find optimal word length for evaluation
    print("***** FIND OPTIMAL WORD LENGTH *****")
    # optimal_word_length = find_optimal_word_length(training_r, training_s)

    # Perform evaluation on test dataset with optimal word length
    print("***** TEST EVALUATION *****")
    final_evaluation(test_r, test_s, 4)


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

    print("Positive reviews in training set:", training_sentiments.value_counts()["positive"])
    print("Negative reviews in training set:", training_sentiments.value_counts()["negative"])
    print("Positive reviews in test set:", test_sentiments.value_counts()["positive"])
    print("Negative reviews in test set:", test_sentiments.value_counts()["negative"])

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
    positive_reviews = training_r[training_s == "positive"].str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()
    negative_reviews = training_r[training_s == "negative"].str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()

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

    return word_likelihood, prior_positive, prior_negative


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

    log_pos = math.log(prior_pos) + likelihood_pos
    log_neg = math.log(prior_neg) + likelihood_neg

    if log_pos > log_neg:
        return "positive"
    else:
        return "negative"


# TASK 6
def evaluation(training_r, training_s, min_word_length):
    # Create a K-Fold object to perfrom 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True)

    # List to store accuracy scores of each fold
    accuracy_scores = []
    true_labels = []
    predicted_labels = []

    # Cross validation loop splitting training & test data
    for train_index, test_index in kf.split(training_r):
        train_reviews = training_r.iloc[train_index]
        test_reviews = training_r.iloc[test_index]
        train_sentiments = training_s.iloc[train_index]
        test_sentiments = training_s.iloc[test_index]

        # Extract features, calculate word occurrences, likelihoods, and priors
        training_words = extract_features(train_reviews, min_word_length, 100)
        word_occurences_pos, word_occurences_neg = feature_frequency(train_reviews, train_sentiments, training_words)
        word_likelihoods, prior_pos, prior_neg = calculate_feature_likelihood(word_occurences_pos, word_occurences_neg,
                                                                              training_words, train_sentiments)
        # Counters for correct + incorrect predictions
        correct = 0
        incorrect = 0

        # Classify each review in the test dataset
        for i, review in enumerate(test_reviews):
            predicted_s = classification(review, prior_pos, prior_neg, word_likelihoods)
            true_labels.append(test_sentiments.iloc[i])
            predicted_labels.append(predicted_s)
            # Compare predicted sentiment with the actual sentiment
            if predicted_s == test_sentiments.iloc[i]:
                correct += 1
            else:
                incorrect += 1

        # Calculate accuracy for current fold
        accuracy = correct / (correct + incorrect)
        accuracy_scores.append(accuracy)

    # Calculate average accuracy
    average_accuracy = np.mean(accuracy_scores)
    print("Classification with min word length:", min_word_length)
    print("Accuracy:", average_accuracy)

    return average_accuracy, true_labels, predicted_labels


def find_optimal_word_length(training_r, training_s):
    # Create list of evaluations with different min word lengths
    evaluations = []
    for i in range(10):
        average_accuracy, true_labels, predicted_labels = evaluation(training_r, training_s, i + 1)
        evaluations.append(average_accuracy)

    # Find the optimal word length from the above list
    optimal_evaluation_accuracy = max(evaluations)
    optimal_word_length = evaluations.index(optimal_evaluation_accuracy) + 1

    print("Optimal evaluation accuracy:", optimal_evaluation_accuracy)
    print("Optimal word length:", optimal_word_length)

    return optimal_word_length


def final_evaluation(test_r, test_s, optimal_word_length):
    average_accuracy, true_labels, predicted_labels = evaluation(test_r, test_s, optimal_word_length)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    total_correct_pos = cm[1][1]
    total_correct_neg = cm[0][0]
    total_incorrect_pos = cm[0][1]
    total_incorrect_neg = cm[1][0]

    print("Confusion Matrix:\n", cm)

    total_pos = total_correct_pos + total_incorrect_neg
    total_neg = total_correct_neg + total_incorrect_pos

    # True + false percentages
    true_pos_per = (total_correct_pos / total_pos) * 100
    false_pos_per = (total_incorrect_pos / total_pos) * 100
    true_neg_per = (total_correct_neg / total_neg) * 100
    false_neg_per = (total_incorrect_neg / total_neg) * 100

    print("Percentage of true positive:", true_pos_per)
    print("Percentage of true negative:", true_neg_per)
    print("Percentage of false positive:", false_pos_per)
    print("Percentage of false negative:", false_neg_per)

    # Accuracy score
    print("Accuracy score:", average_accuracy)


main()
