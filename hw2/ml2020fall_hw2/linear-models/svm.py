import numpy as np
from scipy import optimize
def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    #print(P)
    w = np.zeros((P + 1, 1))
    num = 0
    x1=np.vstack((np.ones((1, X.shape[1])), X))
    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    # end answer
    def func(w,x,y):
        return 0.5*np.matmul(w.T,w)
    def constraint(w,x,y):
        return (y*(np.matmul(w.T,x))-1).reshape(N)
    cons={'type':'ineq','fun':constraint,'args':(x1,y)}
    w1=optimize.minimize(func,w,args=(x1,y),constraints=cons)
    #print(w1)
    w=w1['x']
    
    a=y*(np.matmul(w.T,x1))-1
    #print(np.sum(a<0.00001))
    num=np.sum(a<0.00001)    
    
    return w, num

