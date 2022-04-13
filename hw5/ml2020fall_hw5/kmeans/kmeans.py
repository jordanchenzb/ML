import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE
    
    # begin answer
    
    n,p=x.shape
    votes=np.zeros((n,1000))
    kcentral=x[np.random.choice(n,k),:]
    iter_ctrs=[kcentral.tolist()]
    iter=0
    kkcent=np.ones((k,p))
    t=100
    #cl=np.zeros(n)
    while t>0 :
        
        iter+=1
        global cl
        cl=np.zeros(n)
        for i in range(n):
            c=np.sum((np.square(kcentral-x[i])),axis=1)
            
            cl[i]=np.argmin(c)
        for j in range(k):
            
            if np.sum(cl==j)!=0:
                kkcent[j]=np.sum(x[np.where(cl==j)],axis=0)/np.sum(cl==j)
            else:
                kkcent[j]=kcentral[j]
            #print(iter_ctrs.shape,kkcent.shape)
        t=np.sum(np.square(kcentral-kkcent))
        #print(kcentral,kkcent)
        kcentral=np.copy(kkcent)
        iter_ctrs.append(kkcent.tolist())
        #print(t)
    #print(iter_ctrs)
    idx=cl.astype(int).tolist()
    
    iter_ctrs=np.array(iter_ctrs)
    ctrs=iter_ctrs[iter]
    # end answer

    return idx, ctrs, iter_ctrs
