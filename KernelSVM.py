import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import math

'''
This is a class which implements kernel SVM with SGD as an optimizer
'''


class SVM(object):
    def __init__(self, kernelfunc, lam):
        '''
        This is the construction function of SVM.
        :param kernelfunc: the kernel function you want to use, For example, you can use RBF or Linear
        :param lam: This is the superparameter which controls the weight of regularization term. The larger lam
                is, the less training error comes.
        '''
        self._ker = kernelfunc
        self._lam = lam

    def _normalize(self, data):
        '''
        This is the function used to normalize data, especially when you use Linear kernel. Because Linear
        kernel's inner product can be very large, and thus the gradient below may overflow.
        :param data: the data you want to normalize
        :return: normalized data
        '''
        return (data - np.mean(data)) / (data.max() - data.min())

    def train(self, x_train, y_train):
        '''
        This is the training function, which uses SGD with momentum to optimize
        :param x_train: training data
        :param y_train: training data's label
        :return:
        '''
        row, col = x_train.shape

        # add bias term
        append = np.ones((row, 1))
        x = np.hstack((x_train, append))
        y = y_train

        # alpha is parameters according to representer theorem
        self._alpha = np.ones(row)
        self._kernelMat = np.zeros((row, row))

        # compute kernel matrix
        for i in range(row):
            for j in range(row):
                self._kernelMat[i, j] = self._ker(x[i], x[j])

        # use hinge loss as loss function
        def hinge_loss(arg):
            return max(0, 1 - arg[0] * arg[1])

        # compute loss
        def loss():
            l = 0.0
            t = np.dot(self._kernelMat, self._alpha)
            all = map(hinge_loss, zip(t, y))
            l += reduce(lambda x1, x2: x1 + x2, all, 0.0) / row
            l += self._lam / 2 * np.dot(np.dot(self._alpha, self._kernelMat), self._alpha)
            return l

        # params for SGD with momentum
        v = np.zeros(row)
        eta = 0.9
        epochs = 1000
        llist = []
        lr = 0.001
        rounds = 1
        gd = np.zeros(row)

        # perform SGD
        for epoch in range(epochs):
            for i, d in enumerate(x):
                rounds += 1

                # compute gradient on sample x(i)
                if 1 - y[i] * np.dot(self._kernelMat[i], self._alpha) >= 0:
                    gd = -y[i] * self._kernelMat[i] + \
                         row * self._lam * self._kernelMat[i] * self._alpha[i]
                else:
                    gd = row * self._lam * self._kernelMat[i] * self._alpha[i]

                # you may need to scale lr with time
                # if rounds % 1000 == 0:
                # lr = lr / 10
                if rounds % 50 == 0:
                    llist.append(loss())

                # SGD with momentum
                v = eta * v + gd * lr
                self._alpha -= v

        # remove useless data for the model
        temp = np.fabs(self._alpha) > 1e-6
        self._supalpha = self._alpha[temp]
        self._supvec = x[temp]

        plt.plot([i for i in range(len(llist))], llist)
        plt.show()

    def predict(self, x_test):
        '''
        This is the function to test data
        :param x_test: data need to test
        :return: label for test data
        '''
        row, col = x_test.shape
        # add bias term
        append = np.ones((row, 1))
        x_new = np.hstack((x_test, append))
        result = []
        for i in x_new:
            t = 0
            len_data = len(self._supvec)
            for j in range(len_data):
                t += self._ker(i, self._supvec[j]) * self._supalpha[j]
            if t > 0:
                result.append(1)
            else:
                result.append(-1)
        return np.array(result)


# RBF kernel
sigma = 1.0
RBF = lambda x, y: math.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))

# Linear kernel
Linear = lambda x, y: np.dot(x, y)


def test():
    svm = SVM(RBF, 0.01)
    N = 30
    t1 = np.random.randn(N, 2) + np.array([4, 4])
    t2 = np.random.randn(N, 2) * 10 + np.array([4, 4])
    x_train = np.vstack((t1, t2))

    z1 = np.ones(N, dtype=int)
    z2 = np.ones(N, dtype=int)
    y_train = np.hstack((-z1, z2))

    plt.plot(x_train[:30, 0], x_train[:30, 1], 'go')
    plt.plot(x_train[30:, 0], x_train[30:, 1], 'bo')
    plt.show()
    svm.train(x_train, y_train)
    tag = svm.predict(x_train)
    temp = tag - y_train
    print("error rate is: ", len(temp[temp != 0]) / len(temp))


test()
