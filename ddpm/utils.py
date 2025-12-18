def extract(a, t, x_shape):
    """

    :param a: 各种量
    :param t: 时间 (batchsize,)
    :param x_shape: x的形状
    :return: 将物理量 拿出来 扩展到各个维度上面
    """
    bs, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(bs, *((1, ) * (len(x_shape) - 1)))