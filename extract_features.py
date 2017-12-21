import csv
import random
import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import Normalizer, StandardScaler

def readHouseData():

    categorical_features =  {"MSZoning":["A", "C", "FV", "I", "RH", "RL", "RP", "RM"],
                             "Neighborhood": ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
                                              "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "Names", "NoRidge",
                                              "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW",
                                              "Somerst", "StoneBr", "Timber", "Veenker"],
                             "BldgType": ["1Fam", "2FmCon", "Duplx", "TwnhsE", "TwnhsI"],
                             "HouseStyle": ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"],
                             "Foundation": ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"],
                             "SaleType": ["WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"],
                             "SaleCondition": ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"]
                             }
    numerical_feautures = set(["LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "TotalBsmtSF",
                               "Bedroom", "Kitchen", "TotRmsAbvGrd", "Fireplaces", "GarageArea", "GarageCars",
                               "PoolArea", "MiscVal"])
    binary_features = {"Street": "Grvl", "CentralAir": "Y"}
    rank_features = {"Utilities": {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1, "ELO": 0},
                     "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                     "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                     "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
                     "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
                     "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
                     "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                     "PoolQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "NA": 0},
                     }

    feature_domain = []
    with open('data/train.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        f = 0
        for row in spamreader:
            if f == 0:
                f = 1
                names = row
            else:
                featur_v = row
                for i in xrange(len(featur_v)):
                    if len(feature_domain) <= i:
                        feature_domain.append({})
                    if featur_v[i] not in feature_domain[i].keys():
                        feature_domain[i][featur_v[i]] = len(feature_domain[i])

    # for i in xrange(len(names)):
    #     print names[i], ':', feature_domain[i]

    with open('data/train.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        flag = 0

        X = []
        Y = []

        for row in spamreader:
            if flag == 0:
                flag = 1
            else:
                featur_v = row

                x = []
                total_area = 0.0
                bathroom = 0.0
                sold_time = 0.0

                categorical = []
                numerical = []
                rank = []
                binary = []

                for i in xrange(1, len(featur_v)):
                    if names[i] in ["1stFlrSF", "2ndFlrSF"]:
                        x.append(float(featur_v[i]))
                        total_area += float(featur_v[i])
                    elif names[i] == "GrLivArea":
                        x.append(float(featur_v[i]) - total_area) # the area above 2nd floor

                    if names[i] in ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]:
                        bathroom += float(featur_v[i])

                    if names[i] == "YrSold":
                        sold_time += float(featur_v[i])
                    elif names[i] == "MoSold":
                        sold_time += (float(featur_v[i]) - 1) / 12

                    if names[i] in numerical_feautures:
                        numerical.append(float(featur_v[i]))
                    elif names[i] in binary_features.keys():
                        binary.append(float(featur_v[i] == binary_features[names[i]]))
                    elif names[i] in rank_features.keys():
                        rank.append(1.0*rank_features[names[i]][featur_v[i]])
                    # remove all categorical features
                    # elif names[i] in categorical_features:
                    #     for c_value in categorical_features[names[i]]:
                    #         categorical.append(float(featur_v[i] == c_value))

                    if names[i] == "SalePrice":
                        Y.append(float(featur_v[i]))

                x.append(bathroom)
                x.append(sold_time)
                x = x + numerical + rank + binary + categorical

                # all type is float
                for v in x:
                    assert isinstance(v, float)

                X.append(x)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # x_scaler = Normalizer()
    # y_scaler = Normalizer()

    tmp_Y = [[y] for y in Y]

    scaled_X = x_scaler.fit_transform(X)
    scaled_Y = []
    for row in y_scaler.fit_transform(tmp_Y):
        scaled_Y.append(row[0])

    # print scaled_Y, scaled_X

    return scaled_X, scaled_Y
    # return X, Y


def generateData(X, Y, toShuffle=True):
    # print len(X), len(Y)
    pairs = []
    for i in xrange(len(X)):
        pairs.append([X[i], Y[i]])

    if toShuffle:
        random.shuffle(pairs)
    training_X = []
    training_Y = []
    test_X = []
    test_Y = []
    for i in xrange(len(pairs)):
        if i < 1200:
            training_X.append(pairs[i][0])
            training_Y.append(pairs[i][1])
        else:
            test_X.append(pairs[i][0])
            test_Y.append(pairs[i][1])

    return training_X, training_Y, test_X, test_Y


if __name__ == '__main__':
    X, Y = readHouseData()
    training_X, training_Y, test_X, test_Y = generateData(X, Y)
