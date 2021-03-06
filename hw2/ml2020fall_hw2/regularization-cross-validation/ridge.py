import numpy as np
import scipy
from scipy import linalg

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    x1=np.vstack((np.ones((1, X.shape[1])), X))
    # YOUR CODE HERE
    # begin answer
    w=scipy.linalg.pinv(np.matmul(x1,x1.T)+np.identity(P+1)*lmbda)
    w=np.matmul(w,x1)
    w=np.matmul(w,y.T)
    # end answer
    return w
