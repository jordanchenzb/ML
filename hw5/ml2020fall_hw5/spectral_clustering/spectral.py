import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    n=W.shape[0]
    D=np.zeros((n,n))
    d=np.zeros((n,n))
    for i in range(n):
        D[i,i]=np.sum(W[i])
        d[i,i]=1/np.sqrt(D[i,i])
        if d[i,i]==0:
            print('f')
    L=D-W
    #print(d)
    nL=np.matmul(d,L)
    nL=np.matmul(nL,d)
    
    '''
    for i in range(n):
        for j in range(i+1,n):
            if nL[i,j]!=nL[j,i]:
                print('(',L[i,j],',',L[j,i],')',end='')
    '''
    #print(np.nanmax(nL),np.nanmax(d))
    evalue,evector=np.linalg.eig(nL)
    
    index=np.argsort(evalue.real)
    #print(evalue[index])
    #print(evector[index[:k],:])
    #print(evector[:,index[:10]].shape)
    
    idx=kmeans(evector[:,index[1:2]].real,k)
    return idx
    # end answer
