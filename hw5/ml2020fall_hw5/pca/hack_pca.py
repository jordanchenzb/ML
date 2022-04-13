import numpy as np
import matplotlib.pyplot as plt

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)
    H, W, _ = np.shape(img_r)
    k=50
    # YOUR CODE HERE
    img_gray=0.299*img_r[:, :, 0] + 0.587*img_r[:, :, 1]+0.114*img_r[:, :, 2]
    data_points=[]
    threshold=50
    for i in range(H):
        for j in range(W):
            if img_gray[i, j] > threshold:
                data_points.append([i, j])
    data_points = np.asarray(data_points)            
    avg=np.mean(data_points,axis=0)
    norm=data_points-avg
    cov=np.matmul(norm.T,norm)
    eigval,eigvector=np.linalg.eig(cov)
    rot_mat = np.array([[eigvector[0,0], eigvector[0,1]],
                        [eigvector[1,0], eigvector[1,1]]]).T
    place_mat = np.zeros([H, W, 2],dtype=int)
    for i in range(H):
        for j in range(W):
            place_mat[i, j] =np.matmul(rot_mat,np.array([i,j]))
    img=np.zeros([H,W], dtype=float)
    mean_x=np.mean(place_mat[:,:,0])
    mean_y =np.mean(place_mat[:,:,1])
    for i in range(H):
        for j in range(W):
            place_mat[i,j] = place_mat[i,j] - np.array([mean_x,mean_y])+np.array([H/2, W/2])
    for i in range(H):
        for j in range(W):
            if 0 <= place_mat[i,j,0] < H and 0<= place_mat[i,j,1] < W:
                img[place_mat[i,j,0],place_mat[i,j,1]]=img_gray[i,j]
    
    
    #print(img_gray.shape)
    # begin answer\'''
    '''
    norm_img=img_gray-np.mean(img_gray)
    cov_img=np.matmul(norm_img,norm_img.T)
    eigval,eigvector=np.linalg.eig(cov_img)
    index=np.argsort(-eigval)
    a=eigvector[:,index[0]]
    '''
    
    return img
    # end answer