import math
import numpy
import pandas as pd
from sklearn.metrics import confusion_matrix


#Calculate gaussian naive bayes
def gaussian_naive_bayes(x, mean, stdev):
    exp = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exp


def main():
    #read csv file and shuffle data to get spam and not spam data
    data = pd.read_csv('spambase.data', sep=',')
    data_set = data.values
    numpy.random.shuffle(data_set)

    #divide the data set and create a test and train set of 2300
    train_set = data_set[0:int(data_set.shape[0] / 2)]
    test_set = data_set[int(data_set.shape[0] / 2):data_set.shape[0]]

    #Calculate the probability of spam and not spam data
    train_set_num_rows, train_set_num_cols = train_set.shape
    train_set_labels = train_set[:, train_set_num_cols - 1]
    train_set_prior_prob_spam = numpy.count_nonzero(train_set_labels == 1) / train_set_num_rows
    train_set_prior_prob_not_spam = 1 - train_set_prior_prob_spam

    #Separated spam and not spam
    train_set_spam = []
    train_set_not_spam = []
    for i in range(train_set_num_rows):
        if train_set[i, train_set_num_cols - 1] == 1:
            train_set_spam.append(train_set[i])
        else:
            train_set_not_spam.append(train_set[i])

    #Calculate the mean of the spam and not spam data
    train_set_mean_spam = numpy.mean(train_set_spam, axis=0)
    train_set_mean_not_spam = numpy.mean(train_set_not_spam, axis=0)

    #Calculate the standard deviation of spam and not spam
    train_set_sd_spam = numpy.std(train_set_spam, axis=0)
    train_set_sd_not_spam = numpy.std(train_set_not_spam, axis=0)

    #Update the standard deviation to 0.0001 for 0 standard deviation values
    train_set_sd_spam = numpy.where(train_set_sd_spam == 0.0, 0.0001, train_set_sd_spam)
    train_set_sd_not_spam = numpy.where(train_set_sd_not_spam == 0.0, 0.0001, train_set_sd_not_spam)

    #Separate the test data into spam and not spam
    test_set_num_rows, test_set_num_cols = test_set.shape
    test_set_labels = test_set[:, test_set_num_cols - 1]
    test_set_prior_prob_spam = numpy.count_nonzero(test_set_labels == 1) / test_set_num_rows
    test_set_prior_prob_not_spam = 1 - test_set_prior_prob_spam

    #Run the baive bayes on test data
    prediction = []
    for i in range(test_set_num_rows):
        sum_1 = math.log(test_set_prior_prob_spam)
        sum_0 = math.log(test_set_prior_prob_not_spam)
        for j in range(test_set_num_cols):
            probability_0 = gaussian_naive_bayes(test_set[i, j], train_set_mean_not_spam[j], train_set_sd_not_spam[j])
            probability_1 = gaussian_naive_bayes(test_set[i, j], train_set_mean_spam[j], train_set_sd_spam[j])
            if probability_0 != 0.0:
                sum_0 = sum_0 + math.log(probability_0)
            if probability_1 != 0.0:
                sum_1 = sum_1 + math.log(probability_1)
        if sum_1 > sum_0:
            prediction.append(1)
        else:
            prediction.append(0)

    #print(prediction)
    predictions = numpy.array(prediction)

    #Get the total count of test data
    n = test_set_labels.shape[0]

    #Calculate accuracy, recall, precision and confusion matrix
    accuracy = (test_set_labels == predictions).sum() / n * 100
    TP = ((predictions == 1) & (test_set_labels == 1)).sum()
    FP = ((predictions == 1) & (test_set_labels == 0)).sum()
    #TN = ((predictions == 0) & (test_set_labels == 0)).sum()
    FN = ((predictions == 0) & (test_set_labels == 1)).sum()
    precision = TP / (TP + FP) * 100
    recall = TP / (TP + FN) * 100

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Confusion matrix:\n', confusion_matrix(test_set_labels, predictions))


if __name__ == "__main__":
    main()