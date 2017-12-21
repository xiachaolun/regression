
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def generateData(N=1000):
    w0 = 0.5
    w1 = 1
    Y = []
    X = []
    for i in xrange(0, N):
        x = random.uniform(0, 1)
        X.append(x)
        y = x * w1 + w0
        Y.append(y + random.uniform(-1, 1))
    return X, Y

def loss(X, Y, W):
    N = len(X)
    est_Y = []
    for i in xrange(N):
        est_Y.append(X[i] * W[1] + W[0])
    return mean_squared_error(Y, est_Y)

def fit(X, Y):
    w1 = np.cov(X, Y)[0][1] / np.var(X)
    w0 = np.mean(Y) - w1 * np.mean(X)
    return w0, w1

def fitBySKLearn(X, Y):
    new_X = []
    for x in X:
        new_X.append([x])
    lr = linear_model.LinearRegression()
    lr.fit(new_X, Y)
    predict_Y = lr.predict(new_X)
    print "error:", mean_squared_error(Y, predict_Y)

def fit_Constant(X, Y):
    alpha = 1.0
    w0 = 0.5
    w1 = 1

    N = len(X)

    iteration = 0
    pre_loss = float('inf')
    while iteration < 100000:
        cur_loss = loss(X, Y, [w0, w1])
        # print iteration, "loss:", cur_loss, w0, w1
        if (pre_loss + 1e-10 < cur_loss):
            print 'DIVERGE!'
            break
        if (pre_loss - 1e-10 < cur_loss):
            print 'Converge!', iteration
            break
        pre_loss = cur_loss
        iteration += 1
        g_w0 = 0
        g_w1 = 0
        for i in xrange(N):
            g_w0 += w1*X[i] + w0 - Y[i]
            g_w1 += w1*X[i]*X[i] + w0*X[i] - X[i]*Y[i]
        w0 -= alpha * g_w0 / N
        w1 -= alpha * g_w1 / N

    return w0, w1

def fit_Diminising(X, Y):
    alpha = 1.0
    w0 = 0.5
    w1 = 1

    N = len(X)

    iteration = 0
    pre_loss = float('inf')
    while iteration < 100000:
        cur_loss = loss(X, Y, [w0, w1])
        # print iteration, "loss:", cur_loss, w0, w1
        if (pre_loss + 1e-10 < cur_loss):
            print 'DIVERGE!'
            break
        if (pre_loss - 1e-10 < cur_loss):
            print 'Converge!', iteration
            break
        pre_loss = cur_loss
        iteration += 1
        g_w0 = 0
        g_w1 = 0
        for i in xrange(N):
            g_w0 += w1*X[i] + w0 - Y[i]
            g_w1 += w1*X[i]*X[i] + w0*X[i] - X[i]*Y[i]
        w0 -= alpha / math.sqrt(iteration) * g_w0 / N
        w1 -= alpha / math.sqrt(iteration) * g_w1 / N

    return w0, w1

def fit_Armijo(X, Y):
    w0 = 0.1
    w1 = 2
    epsilon = 0.00001

    N = len(X)

    iteration = 0
    pre_loss = float('inf')
    while iteration < 100000:
        cur_loss = loss(X, Y, [w0, w1])
        # print iteration, "loss:", cur_loss, w0, w1
        if (pre_loss + 1e-10 < cur_loss):
            print 'DIVERGE!'
            break
        if (pre_loss - 1e-10 < cur_loss):
            print 'Converge!', iteration
            break
        pre_loss = cur_loss
        iteration += 1

        g_w0 = 0
        g_w1 = 0
        for i in xrange(N):
            g_w0 += w1 * X[i] + w0 - Y[i]
            g_w1 += w1 * X[i] * X[i] + w0 * X[i] - X[i] * Y[i]

        alpha = 0.5
        while loss(X, Y, [w0-alpha*g_w0, w1-alpha*g_w1]) > loss(X, Y, [w0, w1]) - alpha * epsilon * (g_w0*g_w0+g_w1*g_w1):
            alpha *= 0.5

        w0 -= alpha*g_w0
        w1 -= alpha*g_w1

    return w0, w1

def make_plot(X, Y, W):
    plt.plot(X, Y, 'ro')
    X_test = np.linspace(0, 1, num=10)
    Y_test = []
    for x in X_test:
        Y_test.append(W[0] + W[1]*x)
    plt.plot(X_test, Y_test)
    plt.show()



if __name__ == "__main__":
    X, Y = generateData()

    fitBySKLearn(X, Y)

    w0, w1 = fit_Constant(X, Y)
    print w0, w1, 'error:', loss(X, Y, [w0, w1])

    w0, w1 = fit_Diminising(X, Y)
    print w0, w1, 'error:', loss(X, Y, [w0, w1])

    w0, w1 = fit_Diminising(X, Y)
    print w0, w1, 'error:', loss(X, Y, [w0, w1])

    print 'Armijo Rule'
    w0, w1 = fit_Armijo(X, Y)
    print w0, w1, 'error:', loss(X, Y, [w0, w1])

    w0, w1 = fit(X, Y)
    print w0, w1, 'error:', loss(X, Y, [w0, w1])
    # make_plot(X, Y, [w0, w1])

