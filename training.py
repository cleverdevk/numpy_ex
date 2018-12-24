import numpy as np
import pandas as pd


def read_data():
    print("----------START READ DATA----------")
    training_data = pd.read_csv('./data/training.csv', sep=',')
    np_training_data = np.array(training_data)

    label_data = pd.read_csv('traning_set_label.csv', sep=',')
    np_label_data = np.array(label_data)

    print("[NP_TRAINING_DATA] Shape")
    print(np_training_data.shape)
    print("[NP_LABEL_DATA] Shape")
    print(np_label_data.shape)
    print("----------COMPLETE READ DATA----------")

    return np_training_data, np_label_data

def read_data_test():
    print("----------START READ TEST DATA----------")
    testing_data = pd.read_csv('./data/test.csv', sep=',')
    np_testing_data = np.array(testing_data)

    print("[NP_TESTING_DATA] Shape ")
    print(np_testing_data.shape)
    print("----------COMPLETE READ TEST DATA----------")

    return np_testing_data
