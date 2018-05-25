import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

'''
This is a class which implements kernel SVM with gradient descent
'''

class SVM(object):
    def __init__(self, kernelfunc, lam):
        self._ker = kernelfunc
        self._lam = lam

    def train(self, x_train, y_train, discount=0.75):
        row, col = x_train.shape
        append = np.ones((row, 1))
        x = np.hstack((x_train, append))
        self._xtrain = x
        y = y_train
        gradz = np.zeros(row)

        self._alpha = np.zeros(row)
        self._kernelMat = np.zeros((row, row))

        for i in range(row):
            for j in range(row):
                self._kernelMat[i, j] = self._ker(x[i], x[j])

        def hinge_loss(arg):
            t = 1 - arg[0] * arg[1]
            if t > 0:
                gradz[arg[2]] = -arg[1]
                return t
            else:
                return 0.0

        def loss():
            l = 0.0
            gradz[:] = 0.0
            t = np.dot(self._kernelMat, self._alpha)
            all = map(hinge_loss, zip(t, y, range(row)))
            l += reduce(lambda x1, x2: x1 + x2, all, 0.0) / row
            l += self._lam / 2 * np.dot(np.dot(self._alpha, self._kernelMat), self._alpha)
            #print(self._kernelMat * gradz.reshape((row, 1)))
            grad_alpha = np.sum(self._kernelMat * gradz.reshape((row, 1)), axis=0)/row  \
                + self._lam * np.sum(self._kernelMat * self._alpha.reshape((row, 1)), axis=0)
            return l, grad_alpha

        # params for SGD with momentum
        v = np.zeros(row)
        rounds = 1000
        eta = 0.9
        llist = []
        lr = 0.01
        gd = np.zeros(row)
        for i in range(rounds):
            if (i+1)%100 == 0:
                lr = lr/10
            l, gd = loss()
            llist.append(l)
            v = eta * v + lr * gd
            self._alpha -= v

        temp = np.fabs(self._alpha) > 1e-6
        self._supalpha = self._alpha[temp]
        self._supvec = x[temp]

        plt.plot([i for i in range(len(llist))], llist)
        print(np.sum(x * self._alpha.reshape((row, 1)), axis=0))
        plt.show()

    def predict(self, x_test):
        row, col = x_test.shape
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


svm = SVM(lambda x, y: np.dot(x, y), 0.001)


N = 30
t1 = np.random.randn(N, 2) + np.array([1, 0])
t2 = np.random.randn(N, 2) + np.array([30, 40])
x_train = np.vstack((t1, t2))

z1 = np.ones(N, dtype=int)
z2 = np.ones(N, dtype=int)
y_train = np.hstack((-z1, z2))
'''
x_train = np.array([
    [-2, 4],
    [4, 1],
    [1, 6],
    [2, 4],
    [6, 2]
])

y_train = np.array([-1,-1,1,1,1])
'''

svm.train(x_train, y_train)
tag = svm.predict(x_train)
temp = tag - y_train
print("error rate is: ",len(temp[temp!=0])/len(temp))
