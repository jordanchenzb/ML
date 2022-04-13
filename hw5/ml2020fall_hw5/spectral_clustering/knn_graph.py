import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    n,p=X.shape
    G=np.matmul(X,X.T)
    H=np.tile(np.diag(G),(n,1))
    #print(H.shape,G.shape)
    D=np.sqrt(H.T+H-2*G)
    #D[np.where(D==0)]=10000
    
    index=np.argsort(D,axis=1)
    #print(np.nanmax(D))
    #print(np.nanmin(D[np.where(D!=0)]))
    #print(np.mean(D))
    #print(np.mean(D,axis=1))
    D=np.exp(-D/(np.var(X)))
    #print(np.nanmin(D[np.where(D!=0)]))
    #D[np.where(D<threshold)]=0
    W=np.zeros((n,n))
    for i in range(n):
        
        W[i][index[i][1:k+1]]=D[i][index[i][1:k+1]]
    W[np.where(W<threshold)]=0
    
    '''
    for i in range(n):
        for j in range(i+1,n):
            if W[i,j]==0 and W[j,i]!=0:
                W[i,j]=W[j,i]
            elif W[j,i]==0 and W[i,j]!=0:
                W[j,i]=W[i,j]
    '''
    
    return W
    
    # end answer
