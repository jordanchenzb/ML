import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    N_test=x.shape[0]
    N=x_train.shape[0]
    M2xy=np.matmul(x,x_train.T)
    Mx2=np.square(x).sum(axis=1)
    My2=np.square(x_train).sum(axis=1)
    D=(-2*M2xy+np.matrix(Mx2).T+My2)
    
    #print(D.shape,N,N_test)
    
    y=np.zeros(N_test) 
    index=np.asarray(np.argsort(D))
    #print(np.asarray(index[0]).reshape(-1))
    
    
    #print(index[0,:].shape)
    for i in range(N_test):
        #print('##',D[i,:].shape)
        #t=np.argsort(D[i,:])
        
        #print(t.reshape(-1).shape)
        y[i]=scipy.stats.mode(y_train[index[i][:k]])[0][0]
    # end answer

    return y
