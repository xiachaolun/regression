import math
import numpy as np
import random
import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from extract_features import readHouseData, generateData

from sklearn.preprocessing import StandardScaler

class LinearRegression(object):

    # loss function:
    # 1/(2N) * sum_i (Y[i] - Y_est[i])^2 + lambda/2 * sum_j W[j]^2

    def __init__(self):
        X, Y = readHouseData()
        # self.X is training
        self.X, self.Y, self.test_X, self.test_Y = generateData(X, Y)
        self.N = len(self.X)
        self.M = len(self.X[0])
        self.reg_lambda = 1
        self.max_iteration = 2000
        # only for Ridge provided by Python http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_coeffs.html
        self.alpha = self.N # the larger, the smaller are the weights

    def closedFormSolution(self):
        # lr = linear_model.LinearRegression()
        lr = linear_model.Ridge(alpha=self.alpha)
        lr.fit(self.X, self.Y)
        return lr.intercept_, lr.coef_

    def getEstimateY(self, x, w0, W):
        y = w0
        for j in xrange(len(W)):
            y += x[j] * W[j]
        return y

    def getDataError(self, w0, W, test=1):
        if test:
            X = self.test_X
            Y = self.test_Y
        else:
            X = self.X
            Y = self.Y

        y_est = []
        for x in X:
            y_est.append(self.getEstimateY(x, w0, W))

        return round(math.sqrt(mean_squared_error(y_est, Y)), 4)

    def loss(self, w0, W, reg_lambda):
        l = 0
        for i in xrange(self.N):
            l += (self.Y[i] - self.getEstimateY(self.X[i], w0, W)) ** 2

        l /= self.N

        l += np.linalg.norm(W)**2 * reg_lambda
        # for j in xrange(self.M):
            # l += reg_lambda * W[j]**2
        return l/2

    def batchGradientDescend(self):
        learning_rate = 0.01 * 200 / self.N # 0.01 is good for N = 200

        best_w0, best_W = self.closedFormSolution()
        best_loss = self.loss(best_w0, best_W, self.reg_lambda)

        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, self.M)

        iteration = 0
        pre_loss = float('inf')

        while iteration < self.max_iteration:
            cur_loss = self.loss(w0, W, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss < cur_loss):
                print 'Gradient DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                print 'Gradient CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1
            gradient_w0 = 0
            gradient_W = np.zeros(self.M)
            Y_est = []
            for i in xrange(self.N):
                Y_est.append(self.getEstimateY(self.X[i], w0, W))

            for i in xrange(self.N):
                gradient_w0 += (Y_est[i] - self.Y[i]) / self.N

            for j in xrange(self.M):
                gradient_W[j] = W[j] * self.reg_lambda
                for i in xrange(self.N):
                    gradient_W[j] += (self.X[i][j] * (Y_est[i] - self.Y[i])) / self.N

            w0 -= learning_rate * gradient_w0
            W -= learning_rate * gradient_W

        return w0, W

    def batchGradientDescendWithArmijoRule(self):
        # learning_rate = 0.01 * 200 / self.N # 0.01 is good for N = 200

        eta = 0.001
        best_w0, best_W = self.closedFormSolution()
        best_loss = self.loss(best_w0, best_W, self.reg_lambda)

        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, self.M)

        iteration = 0
        pre_loss = float('inf')

        while iteration < self.max_iteration:
            cur_loss = self.loss(w0, W, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss < cur_loss):
                print 'Gradient with Armijo Rule DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                print 'Gradient with Armijo Rule CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1
            gradient_w0 = 0
            gradient_W = np.zeros(self.M)
            Y_est = []
            for i in xrange(self.N):
                Y_est.append(self.getEstimateY(self.X[i], w0, W))

            for i in xrange(self.N):
                gradient_w0 += (Y_est[i] - self.Y[i]) / self.N

            for j in xrange(self.M):
                gradient_W[j] = W[j] * self.reg_lambda
                for i in xrange(self.N):
                    gradient_W[j] += (self.X[i][j] * (Y_est[i] - self.Y[i])) / self.N

            learning_rate = 1.0

            weights = w0**2 + np.linalg.norm(W)**2

            while self.loss(w0-learning_rate * gradient_w0, W-learning_rate * gradient_W, self.reg_lambda) > cur_loss - eta * learning_rate * weights:
                learning_rate *= 0.5

            print "learning rate:", learning_rate

            # print learning_rate

            w0 -= learning_rate * gradient_w0
            W -= learning_rate * gradient_W

        return w0, W


    def SGD(self):
        learning_rate = 0.1

        best_w0, best_W = self.closedFormSolution()
        best_loss = self.loss(best_w0, best_W, self.reg_lambda)

        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, self.M)

        iteration = 0
        pre_loss = float('inf')

        while iteration < self.max_iteration:
            cur_loss = self.loss(w0, W, self.reg_lambda)
            # print best_loss, pre_loss, cur_loss
            if (pre_loss < cur_loss):
                print 'SGD DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                print 'SGD CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1

            idx = range(self.N)
            # random.shuffle(idx)

            for i in idx:
                gradient_w0 = self.getEstimateY(self.X[i], w0, W) - self.Y[i]
                gradient_W = np.zeros(self.M)
                for j in xrange(self.M):
                    gradient_W[j] = self.X[i][j] * (self.getEstimateY(self.X[i], w0, W) - self.Y[i]) + self.reg_lambda * W[j]

                w0 -= learning_rate * gradient_w0 / self.N
                for j in xrange(self.M):
                    W[j] -= learning_rate * gradient_W[j] / self.N

        return w0, W

    def SGDWithBatch(self, batch_size=10):
        learning_rate = 0.02

        best_w0, best_W = self.closedFormSolution()
        best_loss = self.loss(best_w0, best_W, self.reg_lambda)

        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, self.M)

        iteration = 0
        pre_loss = float('inf')

        current_count = 0

        gradient_w0 = 0
        gradient_W = np.zeros(self.M)

        while iteration < self.max_iteration:
            cur_loss = self.loss(w0, W, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss + 1e-5 < cur_loss):
                print 'SGDB DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                print 'SGDB CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1

            idx = range(self.N)
            random.shuffle(idx)

            for i in idx:
                gradient_w0 += (self.getEstimateY(self.X[i], w0, W) - self.Y[i])

                for j in xrange(self.M):
                    gradient_W[j] += self.X[i][j] * (self.getEstimateY(self.X[i], w0, W) - self.Y[i]) + self.reg_lambda * W[j]

                current_count = (current_count + 1) % batch_size
                if current_count == 0:
                    w0 -= learning_rate * gradient_w0 / self.N
                    gradient_w0 = 0
                    for j in xrange(self.M):
                        W[j] -= learning_rate * gradient_W[j] / self.N
                        gradient_W[j] = 0

        return w0, W

    def SGDWithBatchWithArmijoRule(self, batch_size=10):
        # learning_rate = 0.01

        eta = 0.0001

        best_w0, best_W = self.closedFormSolution()
        best_loss = self.loss(best_w0, best_W, self.reg_lambda)

        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, self.M)

        iteration = 0
        pre_loss = float('inf')

        current_count = 0

        gradient_w0 = 0
        gradient_W = np.zeros(self.M)

        learning_rates = []

        while iteration < self.max_iteration:
            cur_loss = self.loss(w0, W, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss + 1e-5 < cur_loss):
                print 'SGDB with Arjimo Rule DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                print 'SGDB with Arjimo Rule CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1

            idx = range(self.N)
            random.shuffle(idx)

            for i in idx:
                gradient_w0 += (self.getEstimateY(self.X[i], w0, W) - self.Y[i])

                for j in xrange(self.M):
                    gradient_W[j] += self.X[i][j] * (self.getEstimateY(self.X[i], w0, W) - self.Y[i]) + self.reg_lambda * W[j]

                current_count = (current_count + 1) % batch_size
                if current_count == 0:
                    learning_rate = 1.0
                    while learning_rate > 1e-16 and self.loss(w0-learning_rate*gradient_w0, W-learning_rate*gradient_W, self.reg_lambda) > cur_loss - eta * learning_rate * (w0**2 + np.linalg.norm(W)**2):
                        learning_rate *= 0.5

                    learning_rates.append(learning_rate)

                    w0 -= learning_rate * gradient_w0 / self.N
                    gradient_w0 = 0
                    for j in xrange(self.M):
                        W[j] -= learning_rate * gradient_W[j] / self.N
                        gradient_W[j] = 0

        print "average learning rate:", np.mean(learning_rates)

        return w0, W

if __name__ == '__main__':
    for i in xrange(100):
        lr = LinearRegression()

        closed_w0, closed_W = lr.closedFormSolution()

        # gradient_time = time.time()
        SGDB_w0, SGDB_W = lr.SGDWithBatch()
        # gradient_time = time.time() - gradient_time

        SGDB_Armijo_w0, SGDB_Armijo_W = lr.SGDWithBatchWithArmijoRule()

        # SGD_time = time.time()
        # SGD_w0, SGD_W = lr.SGD()
        # SGD_time = time.time() - SGD_time

        # print "time:", "gradient:", gradient_time, "SGD:", SGD_time

        # SGDB_w0, SGDB_W = lr.SGDWithBatch()
        print "training error:", lr.getDataError(closed_w0, closed_W, 0), lr.getDataError(SGDB_w0, SGDB_W, 0), lr.getDataError(SGDB_Armijo_w0, SGDB_Armijo_W, 0)
        print "test error:", lr.getDataError(closed_w0, closed_W), lr.getDataError(SGDB_w0, SGDB_W), lr.getDataError(SGDB_Armijo_w0, SGDB_Armijo_W)
