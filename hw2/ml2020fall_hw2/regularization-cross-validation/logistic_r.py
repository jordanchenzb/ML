import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    #print(P)
    x1=np.vstack((np.ones((1, X.shape[1])), X))
    y=(y+1)/2
    # YOUR CODE HERE
    k=0   
    ita=0.0001
    theta=0.00001
    
    e=1/(1+np.exp(-np.matmul(w.T,x1)))
    gradient=np.matmul(x1,(e-y).T)+lmbda*w
    fg=np.sum(np.matmul(gradient.T,gradient))
    while(fg*ita > theta and k<=10000):
        w=w-ita*gradient
        e=1/(1+np.exp(-np.matmul(w.T,x1)))
        gradient=np.matmul(x1,(e-y).T)+lmbda*w
    
        #l=np.matmul(y,np.log(e).T)+np.matmul(1-y,np.log(1-e).T)+lmbda*np.matmul(w.T,w)
        #print(l)
        fg=np.sum(np.matmul(gradient.T,gradient))
        k=k+1
    # begin answer
    # end answer
    return w
