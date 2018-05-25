import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd
from numpy import dot

def PCA(data, dim):
    """
    use svd to decomposite rather than eig
    @para data : row vectors of data
    @para dim  : number of dimentions to save
    @ret  y    : centerized and projected data
    @ret  accum: how much variance remained
    @ret  V    : eigenvectors of X.T*X
    @ret  X_new: data after PCA
    """
    _,A,V = svd(data)
    #print(np.allclose(V,V1.T))
    mean = np.mean(data,axis=0)
    lamda = A*A
    if dim <= 0 or dim > len(A):
        dim = len(A)
    V[dim:,:]=0
    accum = np.sum(lamda[:dim])/np.sum(lamda)
    y = np.dot(V,(data-mean).T).T
    X_new = np.dot(y,V)+mean
    return y,accum,V,X_new

if __name__ == '__main__':
    data = np.random.randn(100,2)
    data = np.dot(data,np.array([[2,1],[1,2]]))+4
    y,accum,V,X = PCA(data,1)
    plt.scatter(data[:,0],data[:,1],alpha=0.5,marker='o')
    plt.scatter(X[:,0],X[:,1],alpha=0.5,marker='o')
    a = data.copy()
    a[:,1] = 1
    b = data[:,1]
    c = dot(inv(dot(a.T,a)),dot(a.T,b))
    d = dot(a,c)
    plt.scatter(a[:,0],d)
    plt.show()