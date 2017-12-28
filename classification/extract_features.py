from sklearn.preprocessing import Normalizer, StandardScaler

import csv
import random


def readData():
    X = []
    Y = []
    with open('data/train.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            print x


if __name__ == '__main__':
    readData()