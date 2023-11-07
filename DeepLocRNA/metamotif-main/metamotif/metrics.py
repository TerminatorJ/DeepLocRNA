# %%
import tensorflow as tf

# %%
@tf.function
def log(x, basis=None):
    """Compute the logarithm to a given basis. 

    Args:
        x (tf.Tensor): Tensor. 
        basis (int, optional): Basis. Defaults to None (2).

    Returns:
        tf.Tensor: Logarithm of elements in tensor to given basis. 
    """
    
    if basis is None:
        return tf.math.log(x)
    else:
        return tf.math.log(x) / tf.math.log(tf.cast(basis, x.dtype))

# %%
@tf.function
def kld(q, p, basis=None):
    """Computes the KL divergence with a given log-basis."""
    
    p = tf.convert_to_tensor(p)
    q = tf.cast(q, p.dtype)
    q = tf.keras.backend.clip(q, tf.keras.backend.epsilon(), 1)
    p = tf.keras.backend.clip(p, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(q * log(q / p, basis=basis), axis=-1)

# %%
@tf.function
def jsd(p, q, basis=2):
    """Computes the JS divergence with a given log-basis."""
    
    m = (p + q) / 2
    return kld(p, m, basis=basis)/2 + kld(q, m, basis=basis)/2

# %%
def jsd1m(p, q):
    """Computes 1 - JSD to log-basis 2."""
    
    return 1 - jsd(p, q, basis=2)