import numpy as np
import math

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    for i in range(K):
        x=(X.T-Mu[:,i].T).T
        si=Sigma[:,:,i]
        s=np.linalg.inv(si)
        d=np.linalg.det(si)
        x1=np.dot(s,x)
        x2=np.sum(x*x1,axis=0)
        l=1/(2*math.pi*d**0.5)*np.exp(-0.5*x2)
        l1=(l*Phi[i]).reshape(N)  
        #print(l1.shape)
        p[:,i]=l1
    px=np.sum(p,axis=1)
    p=(p.T/px.T).T
    
    # end answer
    
    return p
    