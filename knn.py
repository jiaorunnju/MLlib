import numpy as np
import heapq
from collections import Counter
import matplotlib.pyplot as plt
'''
This is a file implements K-nearest neighbour classifier
'''

class KNN(object):
    def __init__(self, k, metric="L2"):
        self._k = k
        self._traindata = None
        self._metrictable = {
            'L1': lambda x, y: np.sum(np.abs(x-y)),
            'L2': lambda x, y: np.sum(np.square(x-y))
        }
        self._metric = self._metrictable[metric]

    def train(self, data, label):
        self._traindata = data
        self._label = label

    def predict(self, data):
        result = []
        if self._traindata is None:
            raise RuntimeError("You should train first")
        for i in data:
            t = [(self._metric(i, self._traindata[j]), self._label[j]) for j in range(len(self._traindata))]
            heapq.heapify(t)
            first_k = [heapq.heappop(t) for i in range(self._k)]
            tag_counter = Counter([i[1] for i in first_k])
            #print(tag_counter.most_common(1))
            result.append(tag_counter.most_common(1)[0][0])
        return np.array(result)

knn = KNN(2)
N = 30
t1 = np.random.randn(N, 2) + np.array([1, 0])
t2 = np.random.randn(N, 2) + np.array([3, 4])
x_train = np.vstack((t1, t2))

z1 = np.zeros(N, dtype=int)
z2 = np.ones(N, dtype=int)
y_train = np.hstack((z1, z2))

x_test = np.random.randn(5,2)+np.array([2, 2])
knn.train(x_train, y_train)
for i in zip(x_test, knn.predict(x_test)):
    print(i[0],i[1])

plt.plot(t1[:,0], t1[:,1], 'ro')
plt.plot(t2[:,0], t2[:,1], 'b*')
plt.plot(x_test[:,0],x_test[:,1],'go')
plt.show()