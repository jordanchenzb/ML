import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    #print(P,N)
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    x1=np.vstack((np.ones((1, X.shape[1])), X))
    y=(y+1)/2
    #print(y)
    #print(x1)
     # begin answer
    k=0   
    ita=0.01
    theta=0.001

    e=1/(1+np.exp(-np.matmul(w.T,x1)))
    gradient=-np.matmul(x1,(y-e).T)
    
    #l=np.matmul(y,np.log(e).T)+np.matmul(1-y,np.log(1-e).T)
    
    
    #print(gradient)
    fg=np.sum(np.matmul(gradient.T,gradient))
    while(fg*ita > theta and k<=10000):
        w=w-ita*gradient
        e=1/(1+np.exp(-np.matmul(w.T,x1)))
        gradient=-np.matmul(x1,(y-e).T)
    
        #l=np.matmul(y,np.log(e).T)+np.matmul(1-y,np.log(1-e).T)
        #print(l)
        fg=np.sum(np.matmul(gradient.T,gradient))
        k=k+1
    # end answer
    
    return w
