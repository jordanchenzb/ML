import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    
    iters = 0
    # YOUR CODE HERE
    f1=np.zeros((1,N))
    flag=np.ones((1,N))
    maxloop=1000
    s=1
    # begin answer
    
    X1=np.vstack((np.ones((1, X.shape[1])), X))
    #y = np.sign(np.matmul(w.T, np.vstack((np.ones((1, X.shape[1])), X))))
    while iters<maxloop:
        s=1
        for i in range(N):
            #if flag[0,i]==1:
                f1[0,i]=np.sign(np.matmul(w.T,X1[:,i]))*y[0,i]
                if f1[0,i]<=0:
                    #print(f1[0,i],end='')
                    w=(w+X1[:,i].reshape(P+1,1)*y[0,i]).reshape((P+1),1)
                    s=0
                else:
                    flag[0,i]=0
        iters=iters+1
        #print(s)
        if s:
            break        
    
    # end answer
    #print(iters)
    
    return w, iters