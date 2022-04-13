import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Args:
      x: (np.array) containing input data, of shape (N, d_1, ..., d_k)
      w: (np.array) weights, of shape (D, M)
      b: (np.array) biases, of shape (M,)

    Returns:
      out: output, of shape (N, M)
      cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # begin answer
    out=np.matmul(x.reshape(x.shape[0],-1),w)+b
    # end answer
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine layer.

    Args:
      dout: Upstream derivative, of shape (N, M)
      cache: Tuple of:
        x: Input data, of shape (N, d_1, ... d_k)
        w: Weights, of shape (D, M)

    Returns:
      dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      dw: Gradient with respect to w, of shape (D, M)
      db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # begin answer
    dx=np.matmul(dout,w.T).reshape(x.shape)
    dw=np.matmul(x.reshape(x.shape[0],-1).T,dout)
    db=np.sum(dout,axis=0)
    # end answer
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Args:
      x: Inputs, of any shape

    Returns:
      out: Output, of the same shape as x
      cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    # begin answer
    out=x*(x>0)
    # end answer
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Args:
      dout: Upstream derivatives, of any shape
      cache: Input x, of same shape as dout

    Returns:
      dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    # begin answer
    dx=dout*(x>0)
    # end answer
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Args:
      x: Input data of shape (N, C, H, W)
      w: Filter weights of shape (F, C, HH, WW)
      b: Biases, of shape (F,)
      conv_param: A dictionary with the following keys:
        'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        'pad': The number of pixels that will be used to zero-pad the input.

    Returns:
      out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
      cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    # begin answer
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    s = stride
    out = np.zeros((N, F, H_new, W_new))

    for i in range(N):       # ith image    
        for f in range(F):   # fth filter        
            for j in range(H_new):            
                for k in range(W_new):                
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]
    # end answer
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Args:
      dout: Upstream derivatives.
      cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns:
      dx: Gradient with respect to x
      dw: Gradient with respect to w
      db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # begin answer
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in range(N):       # ith image    
        for f in range(F):   # fth filter        
            for j in range(H_new):            
                for k in range(W_new):                
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]                
                    dw[f] += window * dout[i, f, j, k]                
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
    # end answer
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max pooling layer.

    Args:
      x: Input data, of shape (N, C, H, W)
      pool_param: dictionary with the following keys:
        'pool_height': The height of each pooling region
        'pool_width': The width of each pooling region
        'stride': The distance between adjacent pooling regions

    Returns:
      out: Output data
      cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    # begin answer
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = int(1 + (H - HH) / s)
    W_new = int(1 + (W - WW) / s)
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):    
        for j in range(C):        
            for k in range(H_new):            
                for l in range(W_new):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s] 
                    out[i, j, k, l] = np.max(window)

    
    # end answer
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max pooling layer.

    Args:
      dout: Upstream derivatives
      cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
      dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    # begin answer
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = int(1 + (H - HH) / s)
    W_new = int(1 + (W - WW) / s)
    dx = np.zeros_like(x)
    for i in range(N):    
        for j in range(C):        
            for k in range(H_new):            
                for l in range(W_new):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]                
                    m = np.max(window)               
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]

    # end answer
    return dx


def svm_loss(x, y):
    """Computes the loss and gradient using for multiclass SVM classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
