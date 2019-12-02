from __future__ import absolute_import

import numpy as np
from cdg.geometry import SphericalManifold, HyperbolicManifold
from keras import backend as K
from keras.layers import Layer, Average, Concatenate


# Uniform ######################################################################
def spherical_uniform(size, dim=3, r=1.):
    """
    Samples points from a uniform distribution on a spherical manifold.
    Uniform sampling on the sphere can be achieved by sampling from a Gaussian
    in the ambient space of the CCM, and then projecting the samples onto the
    sphere.
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: positive float, the radius of the CCM;
    :return: np.array of shape (size, dim).
    """
    samples = np.random.normal(0, 1, (size, dim))
    samples = spherical_clip(samples, r=r)
    return samples


def hyperbolic_uniform(size, dim=3, r=-1., low=-1., high=1., projection='upper'):
    """
    Samples points from a uniform distribution on a hyperbolic manifold. Uniform
    sampling on a hyperbolic CCM can be achieved by sampling from a uniform
    distribution in the ambient space of the CCM, and then projecting the
    samples onto the CCM.
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: negative float, the radius of the CCM;
    :param low: lower bound of the uniform distribution from which to sample;
    :param high: upper bound of the uniform distribution from which to sample;
    :param projection: 'upper', 'lower', or 'both'. Whether to project points
    always on the upper or lower branch of the hyperboloid, or on both based
    on the sign of the last coordinate.
    :return: np.array of shape (size, dim).
    """
    samples = np.random.uniform(low, high, (size, dim))
    if projection == 'both':
        sign = np.sign(samples[..., -1:])
    elif projection == 'upper':
        sign = 1
    elif projection == 'lower':
        sign = -1
    else:
        raise NotImplementedError('Possible projection modes: \'both\', '
                                  '\'upper\', \'lower\'.')
    samples[..., -1:] = sign * np.sqrt((samples[..., :-1] ** 2).sum(-1, keepdims=True) + r ** 2)

    return samples


def _ccm_uniform(size, dim=3, r=0., low=-1., high=1., projection='upper'):
    """
    Samples points from a uniform distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a uniform distribution in the ambient
    space.
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: float, the radius of the CCM;
    :param low: lower bound of the uniform distribution from which to sample;
    :param high: upper bound of the uniform distribution from which to sample;
    :param projection: 'upper', 'lower', or 'both'. Whether to project points
    always on the upper or lower branch of the hyperboloid, or on both based
    on the sign of the last coordinate.
    :return: np.array of shape (size, dim).
    """
    if r < 0.:
        return hyperbolic_uniform(size, dim=dim, r=r, low=low, high=high,
                                  projection=projection)
    elif r > 0.:
        return spherical_uniform(size, dim=dim, r=r)
    else:
        return np.random.uniform(low, high, (size, dim))


def ccm_uniform(size, dim=3, r=0., low=-1., high=1., projection='upper'):
    """
    Samples points from a uniform distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a uniform distribution in the ambient
    space.
    If a list of radii is passed instead of a single scalar, then the sampling
    is repeated for each value in the list and the results are concatenated
    along the last axis (e.g., see [Grattarola et al. (2018)](https://arxiv.org/abs/1805.06299)).
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: floats or list of floats, radii of the CCMs;
    :param low: lower bound of the uniform distribution from which to sample;
    :param high: upper bound of the uniform distribution from which to sample;
    :param projection: 'upper', 'lower', or 'both'. Whether to project points
    always on the upper or lower branch of the hyperboloid, or on both based
    on the sign of the last coordinate.
    :return: if `r` is a scalar, np.array of shape (size, dim). If `r` is a
    list, np.array of shape (size, len(r) * dim).
    """
    if isinstance(r, int) or isinstance(r, float):
        r = [r]
    elif isinstance(r, list) or isinstance(r, tuple):
        r = r
    else:
        raise TypeError('Radius must be either a single value, a list'
                        'of values (or a tuple).')
    to_ret = []
    for r_ in r:
        to_ret.append(_ccm_uniform(size, dim=dim, r=r_, low=low, high=high,
                                   projection=projection))
    return np.concatenate(to_ret, -1)


# Normal #######################################################################
def spherical_normal(size, tangent_point, r, dim=3, loc=0., scale=1.):
    """
    Samples points from a normal distribution on a spherical manifold.
    Normal sampling on the sphere works by sampling from a Gaussian on the
    tangent plane, and then projecting the sampled points onto the sphere using
    the Riemannian exponential map.
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates);
    :param dim: dimension of the ambient space;
    :param r: positive float, the radius of the CCM;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: np.array of shape (size, dim).
    """
    samples = np.random.normal(loc=loc, scale=scale, size=(size, dim - 1))
    samples = exp_map(samples, r, tangent_point)
    return samples


def hyperbolic_normal(size, tangent_point, r, dim=3, loc=0., scale=1.):
    """
    Samples points from a normal distribution on a hyperbolic manifold.
    Normal sampling on a hyperbolic CCM works by sampling from a Gaussian on the
    tangent plane, and then projecting the sampled points onto the CCM using
    the Riemannian exponential map.
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates);
    :param r: positive float, the radius of the CCM;
    :param dim: dimension of the ambient space;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: np.array of shape (size, dim).
    """
    samples = np.random.normal(loc=loc, scale=scale, size=(size, dim - 1))
    return exp_map(samples, r, tangent_point)


def _ccm_normal(size, dim=3, r=0., tangent_point=None, loc=0., scale=1.):
    """
    Samples points from a Gaussian distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a Gaussian distribution in the
    ambient space.
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.
    :param r: float, the radius of the CCM;
    :param dim: dimension of the ambient space;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: np.array of shape (size, dim).
    """
    if tangent_point is None:
        tangent_point = np.zeros((dim, ))
        tangent_point[-1] = np.abs(r)
    if r < 0.:
        return hyperbolic_normal(size, tangent_point, r, dim=dim, loc=loc, scale=scale)
    elif r > 0.:
        return spherical_normal(size, tangent_point, r, dim=dim, loc=loc, scale=scale)
    else:
        return np.random.normal(loc, scale, (size, dim))


def ccm_normal(size, dim=3, r=0., tangent_point=None, loc=0., scale=1.):
    """
    Samples points from a Gaussian distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a Gaussian distribution in the
    ambient space.
    If a list of radii is passed instead of a single scalar, then the sampling
    is repeated for each value in the list and the results are concatenated
    along the last axis (e.g., see [Grattarola et al. (2018)](https://arxiv.org/abs/1805.06299)).
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.
    :param r: floats or list of floats, radii of the CCMs;
    :param dim: dimension of the ambient space;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: if `r` is a scalar, np.array of shape (size, dim). If `r` is a
    list, np.array of shape (size, len(r) * dim).
    """
    if isinstance(r, int) or isinstance(r, float):
        r = [r]
    elif isinstance(r, list) or isinstance(r, tuple):
        r = r
    else:
        raise TypeError('Radius must be either a single value, a list'
                        'of values (or a tuple).')

    if tangent_point is None:
        tangent_point = [None] * len(r)
    elif isinstance(tangent_point, np.ndarray):
        tangent_point = [tangent_point]
    elif isinstance(tangent_point, list) or isinstance(tangent_point, tuple):
        pass
    else:
        raise TypeError('tangent_point must be either a single point or a'
                        'list of points.')

    if len(r) != len(tangent_point):
        raise ValueError('r and tangent_point must have the same length')

    to_ret = []
    for r_, tp_ in zip(r, tangent_point):
        to_ret.append(_ccm_normal(size, dim=dim, r=r_, tangent_point=tp_,
                                  loc=loc, scale=scale))
    return np.concatenate(to_ret, -1)


# Generic ######################################################################
def get_ccm_distribution(name):
    """
    :param name: 'uniform' or 'normal', name of the distribution.
    :return: the callable function for sampling on a generic CCM;
    """
    if name == 'uniform':
        return ccm_uniform
    elif name == 'normal':
        return ccm_normal
    else:
        raise ValueError('Possible distributions: \'uniform\', \'normal\'')


# Euclidean manifold ###########################################################
def euclidean_distance(x, y):
    """
    Euclidean distance between points. Can be used as user-defined metric for
    sklearn.neighbors.DistanceMetric.
    :param x: one-dimensional np.array;
    :param y: one-dimensional np.array;
    :return: distance between the given points.
    """
    return np.linalg.norm(x - y, axis=-1, keepdims=True)


# Spherical manifold ###########################################################
def is_spherical(x, r=1.):
    """
    Boolean membership to spherical manifold.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: positive float, the radius of the CCM;
    :return: boolean np.array, True if the points are on the CCM.
    """
    return (x ** 2).sum(-1).astype(np.float32) == r ** 2


def spherical_clip(x, r=1.):
    """
    Clips points in the ambient space to a spherical CCM of radius `r`.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: positive float, the radius of the CCM;
    :return: np.array of same shape as x.
    """
    x = x.copy()
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return r * (x / norm)


def spherical_distance(x, y):
    """
    Geodesic distance between points on a spherical CCM. Can be used as
    user-defined metric for sklearn.neighbors.DistanceMetric.
    :param x: one-dimensional np.array;
    :param y: one-dimensional np.array;
    :return: distance between the given points.
    """
    return np.arccos(np.clip(np.dot(x, y.T), -1, 1))


# Hyperbolic manifold ##########################################################
def is_hyperbolic(x, r=-1.):
    """
    Boolean membership to hyperbolic manifold.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: negative float, the radius of the CCM;
    :return: boolean np.array, True if the points are on the CCM.
    """
    return ((x[..., :-1] ** 2).sum(-1) - x[..., -1] ** 2).astype(np.float32) == - r ** 2


def hyperbolic_clip(x, r=-1., axis=-1):
    """
    Clips points in the ambient space to a hyperbolic CCM of radius `r`, by f
    orcing the `axis` coordinate of the points to be
    \(X_{axis} = \sqrt{\sum\limits_{i \neq {axis}} X_{i}^{2} + r^{2}}\).
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: negative float, the radius of the CCM;
    :param axis: int, the axis along which to clip;
    :return: np.array of same shape as x.
    """
    x = x.copy()
    free_components_idxs = np.delete(np.arange(x.shape[-1]), axis)
    x[..., axis] = np.sqrt(np.sum(x[..., free_components_idxs] ** 2, -1) + (r ** 2))
    return x


def hyperbolic_inner(x, y):
    """
    Computes the inner product between points in the pseudo-euclidean
    ambient space of a hyperbolic manifold.
    Works also for 2D arrays of points.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param y: np.array, coordinates are assumed to be in the last axis;
    :return: the inner product matrix.
    """
    minkowski_ipm = np.eye(x.shape[-1])
    minkowski_ipm[-1, -1] = -1
    inner = x.dot(minkowski_ipm).dot(y.T)
    return np.clip(inner, -np.inf, -1)


def hyperbolic_distance(x, y):
    """
    Geodesic distance between points on a hyperbolic CCM. Can be used as
    user-defined metric for sklearn.neighbors.DistanceMetric.
    :param x: one-dimensional np.array;
    :param y: one-dimensional np.array;
    :return: the computed distance.
    """
    inner = hyperbolic_inner(x, y)
    return np.arccosh(-inner)


# Generic CCM ##################################################################
def exp_map(x, r, tangent_point=None):
    """
    Let \(\mathcal{M}\) be a CCM of radius `r`, and \(T_{p}\mathcal{M}\) the
    tangent plane of the CCM at point \(p\) (`tangent_point`).
    This function maps a point `x` on the tangent plane to the CCM, using the
    Riemannian exponential map.
    :param x: np.array, point on the tangent plane (intrinsic coordinates);
    :param r: float, radius of the CCM;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if `None`, defaults to `[0., ..., 0., r]`.
    :return: the exp-map of x to the CCM (extrinsic coordinates).
    """
    extrinsic_dim = x.shape[-1] + 1
    if tangent_point is None:
        tangent_point = np.zeros((extrinsic_dim,))
        tangent_point[-1] = np.abs(r)
    if isinstance(tangent_point, np.ndarray):
        if tangent_point.shape != (extrinsic_dim,) and tangent_point.shape != (1, extrinsic_dim):
            raise ValueError('Expected tangent_point of shape ({0},) or (1, {0}), got {1}'.format(extrinsic_dim, tangent_point.shape))
        if tangent_point.ndim == 1:
            tangent_point = tangent_point[np.newaxis, ...]
        if not belongs(tangent_point, r)[0]:
            raise ValueError('Tangent point must belong to manifold {}'.format(tangent_point))
    else:
        raise TypeError('tangent_point must be np.array or None')

    if r > 0.:
        return SphericalManifold.exp_map(tangent_point, x)
    elif r < 0.:
        return HyperbolicManifold.exp_map(tangent_point, x)
    else:
        return x


def log_map(x, r, tangent_point=None):
    """
    Let \(\mathcal{M}\) be a CCM of radius `r` and \(T_{p}\mathcal{M}\) the
    tangent plane of the CCM at point \(p\) (`tangent_point`).
    This function maps a point `x` on the CCM to the tangent plane, using the
    Riemannian logarithmic map.
    :param x: np.array, point on the CCM (extrinsic coordinates);
    :param r: float, radius of the CCM;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.
    :return: the log-map of x to the tangent plane (intrinsic coordinates).
    """
    extrinsic_dim = x.shape[-1]
    if tangent_point is None:
        tangent_point = np.zeros((extrinsic_dim,))
        tangent_point[-1] = np.abs(r)
    if isinstance(tangent_point, np.ndarray):
        if tangent_point.shape != (extrinsic_dim,) and tangent_point.shape != (1, extrinsic_dim):
            raise ValueError('Expected tangent_point of shape ({0},) or (1, {0}), got {1}'.format(extrinsic_dim, tangent_point.shape))
        if tangent_point.ndim == 1:
            tangent_point = tangent_point[np.newaxis, ...]
        if not belongs(tangent_point, r)[0]:
            raise ValueError('Tangent point must belong to manifold {}'.format(tangent_point))
    else:
        raise TypeError('tangent_point must be np.ndarray or None')

    if r > 0.:
        return SphericalManifold.log_map(tangent_point, x)
    elif r < 0.:
        return HyperbolicManifold.log_map(tangent_point, x)
    else:
        return x


def belongs(x, r):
    """
    Boolean membership to CCM of radius `r`.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: float, the radius of the CCM;
    :return: boolean np.array, True if the points are on the CCM.
    """
    if r > 0.:
        return is_spherical(x, r)
    elif r < 0.:
        return is_hyperbolic(x, r)
    else:
        return np.ones(x.shape[:-1]).astype(np.float32)


def clip(x, r, axis=-1):
    """
    Clips points in the ambient space to a CCM of radius `r`.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: float, the radius of the CCM;
    :param axis: axis along which to clip points in the hyperbolic case (`r < 0`);
    :return: np.array of same shape as x.
    """
    if r > 0.:
        return spherical_clip(x, r)
    elif r < 0.:
        return hyperbolic_clip(x, r, axis=axis)
    else:
        return x


def get_distance(r):
    """
    :param r: float, the radius of the CCM;
    :return: the callable distance function for the CCM of radius `r`.
    """
    if r > 0.:
        return spherical_distance
    elif r < 0.:
        return hyperbolic_distance
    else:
        return euclidean_distance


# Layers #######################################################################
class CCMMembership(Layer):
    """
    Computes the membership of the given points to a constant-curvature
    manifold of radius `r`, as:
    $$
        \\mu(x) = \\mathrm{exp}\\left(\\cfrac{-\\big( \\langle \\vec x, \\vec x \\rangle - r^2 \\big)^2}{2\\sigma^2}\\right).
    $$

    If `r=0`, then \(\\mu(x) = 1\).
    If more than one radius is given, inputs are evenly split across the
    last dimension and membership is computed for each radius-slice pair.
    The output membership is returned according to the `mode` option.

    **Input**

    - tensor of shape `(batch_size, input_dim)`;

    **Output**

    - tensor of shape `(batch_size, output_size)`, where `output_size` is
    computed according to the `mode` option;.

    :param r: int ot list, radia of the CCMs.
    :param mode: 'average' to return the average membership across CCMs, or
    'concat' to return the membership for each CCM concatenated;
    :param sigma: spread of the membership curve;
    """
    def __init__(self, r=1., mode='average', sigma=1., **kwargs):
        super(CCMMembership, self).__init__(**kwargs)
        if isinstance(r, int) or isinstance(r, float):
            self.r = [r]
        elif isinstance(r, list) or isinstance(r, tuple):
            self.r = r
        else:
            raise TypeError('r must be either a single value, or a list/tuple '
                            'of values.')
        possible_modes = {'average', 'concat'}
        if mode not in possible_modes:
            raise ValueError('Possible modes: {}'.format(possible_modes))
        self.mode = mode
        self.sigma = sigma

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        output_part = []
        manifold_size = K.int_shape(inputs)[-1] // len(self.r)

        for idx, r_ in enumerate(self.r):
            start = idx * manifold_size
            stop = start + manifold_size
            part = inputs[..., start:stop]
            sign = np.sign(r_)
            if sign == 0.:
                # This is weird but necessary to make the layer differentiable
                output_pre = K.sum(inputs, -1, keepdims=True) * 0. + 1.
            else:
                free_components = part[..., :-1] ** 2
                bound_component = sign * part[..., -1:] ** 2
                all_components = K.concatenate((free_components, bound_component), -1)
                ext_product = K.sum(all_components, -1, keepdims=True)
                output_pre = K.exp(-(ext_product - sign * r_ ** 2) ** 2 / (2 * self.sigma ** 2))

            output_part.append(output_pre)

        if len(output_part) >= 2:
            if self.mode == 'average':
                output = Average()(output_part)
            elif self.mode == 'concat':
                output = Concatenate()(output_part)
            else:
                raise ValueError()  # Never gets here
        else:
            output = output_part[0]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (1, )
        return output_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(CCMMembership, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
