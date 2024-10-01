import numpy as np

# rescaling function, recsales the variables so the integral is done between [-1,1] instead of range [a,b]
def func_rescale(xp=None, range=None, f=None):
    weight = (range[1] - range[0]) * 0.5
    x = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return(weight * f(x=x))


def func_gaussian_intg(f, a, b, n):
    # Get nodes and weights for n-point Gaussian quadrature
    nodes, weights = np.polynomial.legendre.leggauss(n)
    
    # Compute the integral using the rescaled nodes and weights
    integral = sum(func_rescale(xp=node, range=[a, b], f=f) * weight for node, weight in zip(nodes, weights))

    return integral