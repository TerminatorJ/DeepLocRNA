# %%
import numpy as np
import tensorflow as tf

from metamotif.metrics import jsd1m

# %%
@tf.function()
def motif_point_similarity(pwm_1, pwm_2, boolean_mask=None, weights=None, sim_fn=jsd1m):
    """Computes the point-similarity between two position-weight-matrices.

    Args:
        pwm_1 (tf.Tensor): PWM_1
        pwm_2 (tf.Tensor): PWM_2
        boolean_mask (list, optional): 1D binary mask, indicating positions to include (1) and ignore (0). Defaults to None.
        weights (list, optional): 1D position-wise weights. Defaults to None.
        sim_fn (function, optional): Similarity function. Defaults to jsd1m.

    Returns:
        tf.Tensor: Scalar of motif similarity. In range [0, 1] in case of jsd1m. 
    """
    
    pwm_1, pwm_2 = tf.cast(pwm_1, tf.float64), tf.cast(pwm_2, tf.float64)
    
    tf.debugging.assert_equal(pwm_1.shape, pwm_1.shape)
    
    sim = sim_fn(pwm_1, pwm_2)
    
    if weights is not None:
        sim = tf.multiply(sim, weights)
    if boolean_mask is not None:
        sim = tf.boolean_mask(sim, tf.cast(boolean_mask, tf.bool))
        
    return tf.reduce_mean(sim)

# %%
@tf.function()
def tile_pwm(pwm, n):
    """Replicate/tile the given position-weight-matric n-times along axis 0. 

    Args:
        pwm (tf.Tensor): PWM.
        n (int): Number of times to tile the PWM.

    Returns:
        tf.Tensor: Tiles of shape (n, *pwm.shape)
    """
    
    tf.debugging.assert_rank(pwm, 2)
    return tf.reshape(tf.tile(pwm, tf.constant([n, 1], dtype=tf.int64)), (n, pwm.shape[0], pwm.shape[1]))

# %%
def sliding_window_view(x, window_size):
    """Generates (sliding) windows along the first axis of the input. Note: this is not a tf.function and may slow down performance. 

    Args:
        x (tf.Tensor): Input tensor. 
        window_size (int): Sliding window size

    Returns:
        tf.Tensor: Sliding windows of shape (x.shape[0] - window_size + 1, window_size, *x.shape[1:]) 
    """
    
    windows = np.zeros(shape = (x.shape[0] - window_size + 1, window_size, *x.shape[1:]))
    for i in range(windows.shape[0]):
        windows[i, ] = x[i:(i+window_size), ]
    return windows

# %%
@tf.function()
def tf_sliding_window_view(x, window_size):
    """TF version of sliding_window_view."""
    
    return tf.py_function(func=sliding_window_view, inp=[x, window_size], Tout=tf.float64)


# %%
@tf.function()
def pad_pwm(pwm, padding=0):
    """Pads the input position-weight matrix. 

    Args:
        pwm (tf.Tensor): Input PWM.
        padding (int, optional): Padding size. Defaults to 0.

    Returns:
        tf.Tensor: Padded PWM. 
    """
    return tf.pad(pwm, [[padding, padding,], [0, 0]], 'CONSTANT')

# %%
@tf.function()
def make_missing_mask(pwm_1, pwm_2):
    """Given two PWMs (same shape) with missing (row-sum = 0) positions, creates an indicator-mask of positions present in both PWMs. 

    Args:
        pwm_1 (tf.Tensor): PWM_1.
        pwm_2 (tf.Tensor): PWM_2.

    Returns:
        tf.Tensor: 1D binary tensor, indicating which positions are occupied by both PWMs. 
    """
    
    return tf.cast(tf.logical_and(tf.cast(tf.reduce_sum(pwm_1, axis=1), tf.bool), tf.cast(tf.reduce_sum(pwm_2, axis=1), tf.bool)), tf.float64)

# %%
@tf.function()
def _map_masked_point_similarity(pwm_a_b):
    """Helper function to broadcast similarity computation over two lists of PWMs with same shape.

    Args:
        pwm_a_b (tf.Tensor): Paired list of PWMs of shape (N, 2, length, depth). 

    Returns:
        tf.Tensor: 1D tensor of pairwise similarities. 
    """
    
    pwm_a, pwm_b = pwm_a_b[0], pwm_a_b[1]
    mask = make_missing_mask(pwm_a, pwm_b)
    return motif_point_similarity(pwm_a, pwm_b, boolean_mask=mask)

#@tf.function()
def motif_similarity(pwm_1, pwm_2, min_size=3, reduce=tf.reduce_max):
    """Computes the similarity between two position-weight matrices (PWMs). 
    
    Given a minimum overlap size of min_size, the shorter PWM is shifted over the longer PWM,
    with similarity being computed at each step. The thus obtained vector of similarities is then 
    reduced (default: max) to obtain the final scalar similarity value between the two PWMs. 

    Args:
        pwm_1 (tf.Tensor): PWM_1.
        pwm_2 (tf.Tensor): PWM_2.
        min_size (int, optional): Minimum overlap size. Defaults to 3.
        reduce (function, optional): Similarity reduce function. Defaults to tf.reduce_max.

    Returns:
        tf.Tensor: Scalar similarity value between the two PWMs. 
    """
    
    pwm_1, pwm_2 = tf.cast(pwm_1, tf.float64), tf.cast(pwm_2, tf.float64)
    
    # assign larger PWM to pwm_1
    if pwm_1.shape[0] < pwm_2.shape[0]:
        pwm_1, pwm_2 = pwm_2, pwm_1
    
    # pad the longer PWM and create sliding windows over it
    pwm_1_padded = pad_pwm(pwm_1, padding=(pwm_2.shape[0] - min_size))
    pwm_1_padded_windows = tf_sliding_window_view(pwm_1_padded, window_size=pwm_2.shape[0])
    #print(pwm_1_padded_windows[0])
    
    # tile the shorted PWM to match the number of sliding windows
    pwm_2_tiled = tile_pwm(pwm_2, pwm_1_padded_windows.shape[0])
    #print(pwm_2_tiled[0])

    # compute the dinstance between pwm_2 and all windows of pwm_1
    window_sims = tf.map_fn(_map_masked_point_similarity, tf.stack([pwm_1_padded_windows, pwm_2_tiled], axis=1))

    return reduce(window_sims)

# %%
def motif_similarity_to_reference(pwm, reference, reduce=tf.reduce_max, **kwargs):
    """Compute the similarity between a PWM and a (list) of reference PWMs. 
    
    Similarity between the PWM and a single reference is computed via motif_similarity, 
    using provided kwargs. Similarity values between the PWM and all references are reduce via a 
    reduction function (default: max). 

    Args:
        pwm (tf.Tensor): PWM. 
        reference (tf.Tensor or list): (List of) reference PWMs.
        reduce (function, optional): Reduce function. Defaults to tf.reduce_max.

    Returns:
        tf.Tensor: Scalar of similarity to references. 
    """
    
    if isinstance(reference, list):
        pass
    else:
        reference = [reference]
    return reduce([motif_similarity(pwm, pwm_r, **kwargs) for pwm_r in reference])