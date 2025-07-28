import numpy as np

def swor_gumbel_uniform(n, k):
    """
    Uniformly select k elements out of n objects without replacement.
    Takes linear time, uses the Gumbel max trick and introselect.
    Note: the order in which elements appear is not specified but not random

    see: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
    and https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    """
    assert k <= n
    if k == 0:
        return []
    G = np.random.gumbel(0, 1, size=(n))   # Gumbel noise

    return np.argpartition(G, -k)[-k:]   # select k largest indices in Gumbel noise


