import math
import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from extract_features import readHouseData, generateData

from sklearn.preprocessing import StandardScaler

class LinearRegression(object):
    reg_lambda = 1.0
    max_iteration = 5000

    def fitByDefaultLR(self, training_X, training_Y):
        # lr = linear_model.LinearRegression()
        lr = linear_model.Ridge(alpha=1/self.reg_lambda)
        lr.fit(training_X, training_Y)
        predicted_Y = lr.predict(training_X)
        # print "LR closed form solution error:", mean_squared_error(predicted_Y, training_Y)
        # print lr.intercept_, lr.coef_
        return lr.intercept_, lr.coef_

    def getEstimateY(self, x, W, w0):
        y = w0
        for j in xrange(len(W)):
            y += x[j] * W[j]
        return y

    def loss(self, X, Y, W, w0, reg_lambda):
        N = len(X)
        M = len(X[0])
        l = 0
        for i in xrange(N):
            l += (Y[i] - self.getEstimateY(X[i], W, w0))**2

        for j in xrange(M):
            l += reg_lambda * W[j]**2
        # print "loss function:", l/2, " training MSE:", mean_squared_error(Y, Y_est)
        return l/2 / N

    def batchGradientDescend(self, X, Y):
        learning_rate = 0.00005

        best_w0, best_W = self.fitByDefaultLR(X, Y)
        best_loss = self.loss(X, Y, best_W, best_w0, 1.0)

        # print best_w0, best_W

        N = len(X)
        M = len(X[0])
        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, M)

        iteration = 0
        pre_loss = float('inf')

        while iteration < self.max_iteration:
            cur_loss = self.loss(X, Y, W, w0, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss + 1e-5 < cur_loss):
                # print 'Gradient DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                # print 'Gradient CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1
            gradient_w0 = 0
            gradient_W = np.zeros(M)
            Y_est = []
            for i in xrange(N):
                Y_est.append(self.getEstimateY(X[i], W, w0))

            for i in xrange(N):
                gradient_w0 += Y_est[i] - Y[i]

            for j in xrange(M):
                gradient_W[j] = W[j] * self.reg_lambda
                for i in xrange(N):
                    gradient_W[j] += X[i][j] * (Y_est[i] - Y[i])

            w0 -= learning_rate * gradient_w0
            for j in xrange(M):
                W[j] -= learning_rate * gradient_W[j]

        # print w0, W
        # print loss(X, Y, W, w0, 0) / loss(X, Y, best_W, best_w0, 0)

        return w0, W

    def SGD(self, X, Y):
        learning_rate = 0.00001

        best_w0, best_W = self.fitByDefaultLR(X, Y)
        best_loss = self.loss(X, Y, best_W, best_w0, 1.0)

        N = len(X)
        M = len(X[0])
        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, M)


        iteration = 0
        pre_loss = float('inf')

        while iteration < self.max_iteration:
            cur_loss = self.loss(X, Y, W, w0, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss + 1e-5 < cur_loss):
                # print 'SGD DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                # print 'SGD CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1

            idx = range(N)
            random.shuffle(idx)

            for i in idx:
                w0 -= learning_rate * (self.getEstimateY(X[i], W, w0) - Y[i])
                for j in xrange(M):
                    W[j] -= learning_rate * (X[i][j] * (self.getEstimateY(X[i], W, w0) - Y[i]) + self.reg_lambda * W[j])

        return w0, W

    def SGDWithBatch(self, X, Y):
        learning_rate = 0.00001

        best_w0, best_W = self.fitByDefaultLR(X, Y)
        best_loss = self.loss(X, Y, best_W, best_w0, self.reg_lambda)

        # print best_w0, best_W

        N = len(X)
        M = len(X[0])
        w0 = random.uniform(-1, 1)
        W = np.random.uniform(-1, 1, M)

        iteration = 0
        pre_loss = float('inf')

        batch_size = 10
        current_count = 0

        gradient_w0 = 0
        gradient_W = np.zeros(M)

        while iteration < self.max_iteration:
            cur_loss = self.loss(X, Y, W, w0, self.reg_lambda)
            # print best_loss, cur_loss, cur_loss / best_loss
            if (pre_loss + 1e-5 < cur_loss):
                # print 'SGD with batch DIVERGES!'
                break
            if (pre_loss - 1e-6 < cur_loss):
                # print 'SGD with batch CONVERGES!', iteration
                break
            pre_loss = cur_loss
            iteration += 1

            idx = range(N)
            random.shuffle(idx)

            for i in idx:
                current_count = (current_count + 1) % batch_size
                gradient_w0 += self.getEstimateY(X[i], W, w0) - Y[i]
                if current_count == 0:
                    w0 -= learning_rate * gradient_w0
                    gradient_w0 = 0

                for j in xrange(M):
                    gradient_W[j] += X[i][j] * (self.getEstimateY(X[i], W, w0) - Y[i]) + self.reg_lambda * W[j]
                    if current_count == 0:
                        W[j] -= learning_rate * gradient_W[j]
                        gradient_W[j] = 0

        return w0, W

if __name__ == '__main__':
    for i in xrange(1000):
        X, Y = readHouseData()
        training_X, training_Y, test_X, test_Y = generateData(X, Y)
        lr = LinearRegression()
        closed_w0, closed_W = lr.fitByDefaultLR(training_X, training_Y)
        gradient_w0, gradient_W = lr.batchGradientDescend(training_X, training_Y)
        SGD_w0, SGD_W = lr.SGD(training_X, training_Y)
        SGDBatch_w0, SGDBatch_W = lr.SGDWithBatch(training_X, training_Y)
        # print "loss", lr.loss(training_X, training_Y, closed_W, closed_w0, 1.0), \
        #     lr.loss(training_X, training_Y, gradient_W, gradient_w0, 1.0), \
        #     lr.loss(training_X, training_Y, SGD_W, SGD_w0, 1.0), \
        #     lr.loss(training_X, training_Y, SGDBatch_W, SGDBatch_w0, 1.0)
        Y_closed = []
        Y_gradient = []
        Y_SGD = []
        Y_SGD_batch = []
        for x in training_X:
            Y_closed.append(lr.getEstimateY(x, closed_W, closed_w0))
            Y_gradient.append(lr.getEstimateY(x, gradient_W, gradient_w0))
            Y_SGD.append(lr.getEstimateY(x, SGD_W, SGD_w0))
            Y_SGD_batch.append(lr.getEstimateY(x, SGDBatch_W, SGDBatch_w0))

        error = mean_squared_error(training_Y, Y_closed)
        print "training error:", mean_squared_error(training_Y, Y_gradient) / error, \
            mean_squared_error(training_Y, Y_SGD) / error, \
            mean_squared_error(training_Y, Y_SGD_batch) / error
