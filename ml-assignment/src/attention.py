import numpy as np

def softmax(x, axis=-1):
    """
    Compute softmax values for each set of scores in x.
    Subtract max for numerical stability.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes the Scaled Dot-Product Attention.
    
    Args:
        Q (numpy.ndarray): Queries matrix of shape (..., seq_len_q, d_k)
        K (numpy.ndarray): Keys matrix of shape (..., seq_len_k, d_k)
        V (numpy.ndarray): Values matrix of shape (..., seq_len_v, d_v)
        mask (numpy.ndarray, optional): Mask tensor to mask out values. 
                                        Should be broadcastable to (..., seq_len_q, seq_len_k).
                                        Elements with 1 are kept, 0 are masked (set to -inf).
                                        Alternatively, if boolean, False is masked.
                                        Commonly, 0 or -inf is used directly in other frameworks, 
                                        but here we'll assume 1 for keep, 0 for mask.
    
    Returns:
        output (numpy.ndarray): The weighted sum of values. Shape (..., seq_len_q, d_v)
        attention_weights (numpy.ndarray): The attention weights. Shape (..., seq_len_q, seq_len_k)
    """
    
    # 1. Matmul Q and K^T
    # We need to transpose the last two dimensions of K for the dot product
    # K is (..., seq_len_k, d_k), so K^T (conceptually) is (..., d_k, seq_len_k)
    # np.matmul handles the broadcasting for the batch dimensions automatically
    # and performs matrix multiplication on the last two dimensions.
    # So we explicitly transpose the last two dims of K.
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1))
    
    # 2. Scale
    scores = scores / np.sqrt(d_k)
    
    # 3. Mask (optional)
    if mask is not None:
        # We assume mask has 0 for positions to mask out.
        # We set them to a very large negative number so softmax makes them 0.
        # If mask is boolean, we treat False as mask out.
        if mask.dtype == bool:
             scores = np.where(mask, scores, -1e9)
        else:
             # Assuming 1 is keep, 0 is mask
             scores = np.where(mask == 1, scores, -1e9)
    
    # 4. Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # 5. Matmul with V
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
