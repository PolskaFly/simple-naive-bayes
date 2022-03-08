#########################
#    Robert  Fabbro     #
#      1001724536       #
#########################

from numpy import float64
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from csv import reader
from collections import Counter
from numpy import mean
from numpy import std
from scipy.stats import norm
import numpy

# load the data
def load_csv(filename):
    dataset_x = list()
    dataset_y = list()
    with open(filename, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            dataset_x.append([row[0], row[1], row[2], row[3]])
            dataset_y.append(row[4])
    
    return dataset_x, dataset_y

# converts a column to float64
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float64(row[column].strip())

# measure the accuracy of the output predictions
def accuracy_metric(actual, predicted):
    correct = 0
    for i, val in enumerate(actual):
        if val == predicted[i]:
            correct += 1
    return (correct/float(len(actual))) * 100

# split dataset based on some percentage
def split_dataset(X, y, split_percent):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_percent, random_state=42)

    return X_train, X_test, y_train, y_test

# conpute the probability of each class P(C)
def probability_class(y):
    length = len(y)
    count_dict = dict(Counter(y))
    
    for key in count_dict:
        count_dict[key] = (count_dict[key]/length)

    return count_dict

# genereate a distribution for the given column
def generate_distribution(x):
    mu = mean(x)
    sigma = std(x)
    dist = norm(mu, sigma)

    return dist

# generate all pdfs required for computating P(X|C)
def generate_pdfs(x, y, classes):
    pdf_dict = dict()
    x = numpy.array(x)
    y = numpy.array(y)
    for item in classes:
        x_class = x[y==item]
        x0 = generate_distribution(x_class[:,0])
        x1 = generate_distribution(x_class[:,1])
        x2 = generate_distribution(x_class[:,2])
        x3 = generate_distribution(x_class[:,3])
        pdf_dict[item] = [x0, x1, x2, x3]

    return pdf_dict

# calculation P(C|X)
def calculate_probability(X, class_prob, pdf_dict):
    return class_prob * pdf_dict[0].pdf(X[0]) * pdf_dict[1].pdf(X[1]) * pdf_dict[2].pdf(X[2]) * pdf_dict[3].pdf(X[3])

def naive_bayes(file):
    # load data
    data_x, data_y = load_csv(file)

    for i in range(4):
        str_column_to_float(data_x, i)

    # split data into training/validation set
    X_train, X_test, y_train, y_test = split_dataset(data_x, data_y, .33)
    classes = set(y_train)
    class_map = list()
    for item in classes:
        class_map.append(item)
    
    # find class count probability
    class_probabilities = probability_class(y_train)

    # find probability of data | C
    pdfs = generate_pdfs(X_train, y_train, class_map)

    # find probability of C | data
    prediction = list()
    prediction2 = list()
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    for X in X_test:
        compute_prediction = []
        for item in class_map:
            compute_prediction.append(calculate_probability(X, class_probabilities[item], pdfs[item]))
        prediction.append(class_map[compute_prediction.index(max(compute_prediction))])
        prediction2.append(clf.predict([X]))

    print("Implemented Naive Bayes: ", accuracy_metric(y_test, prediction))
    print("Scikit-learn Naive Bayes: ", accuracy_metric(y_test, prediction2))

            

naive_bayes("iris.csv")