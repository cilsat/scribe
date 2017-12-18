import numpy as np
from scipy.linalg import inv, det


def glr(x, y, theta=1.82):
    xm = multivariate_normal.logpdf(
        x, np.mean(x, axis=0), np.cov(x, rowvar=False))
    ym = multivariate_normal.logpdf(
        y, np.mean(y, axis=0), np.cov(y, rowvar=False))
    z = np.vstack((x, y))
    zm = multivariate_normal.logpdf(
        z, np.mean(z, axis=0), np.cov(z, rowvar=False))
    return (np.sum(zm) - np.sum(np.hstack((xm, ym)))) / len(z)**theta


def glr2(x, y, theta=1.0):
    cx = np.cov(x, rowvar=0)
    cy = np.cov(y, rowvar=0)
    nx = x.shape[0]
    ny = y.shape[0]
    n = nx + ny
    d = -0.5 * (nx * np.log(det(cx)) + ny * np.log(det(cy)) -
                n * np.log(det((nx / n) * cx + (ny / n) * cy)))
    return d


def bic(x, y, theta=1.0, params={}):
    px = np.log(det(np.cov(x, rowvar=0)))
    py = np.log(det(np.cov(x, rowvar=0)))
    z = np.vstack((x, y))
    pz = np.log(det(np.cov(z, rowvar=0)))
    d = 0.5 * (z.shape[0] * pz - x.shape[0] * px - y.shape[0] * py)
    p = z.shape[1]
    corr = theta * 0.25 * p * (p + 3) * np.log(z.shape[0])
    return d - corr


def kl2(x, y):
    cx = np.cov(x, rowvar=0)
    cy = np.cov(y, rowvar=0)
    cix = inv(cx)
    ciy = inv(cy)
    dxy = np.mean(x, axis=0) - np.mean(y, axis=0)
    d = 0.5 * (np.trace((cx - cy) * (ciy - cix)) +
               np.trace((ciy + cix) * np.outer(dxy, dxy)))
    return d
