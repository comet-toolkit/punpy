"""Use Monte Carlo to propagate uncertainties"""

import numpy as np
import numdifftools as nd
import warnings

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def calculate_Jacobian(fun, x, Jx_diag=False):
    """
    Calculate the local Jacobian of function y=f(x) for a given value of x

    :param fun: flattened measurement function
    :type fun: function
    :param x: flattened local values of input quantities
    :type x: array
    :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
    :rtype Jx_diag: bool, optional
    :return: Jacobian
    :rtype: array
    """
    Jfun = nd.Jacobian(fun)

    if Jx_diag:
        y = fun(x)
        Jfun = nd.Jacobian(fun)
        Jx = np.zeros((len(x), len(y)))
        for j in range(len(y)):
            xj = np.zeros(int(len(x) / len(y)))
            for i in range(len(xj)):
                xj[i] = x[i * len(y) + j]
            Jxj = Jfun(xj)
            for i in range(len(xj)):
                Jx[i * len(y) + j, j] = Jxj[0][i]
    else:
        Jx = Jfun(x)

    if len(Jx) != len(fun(x).flatten()):
        warnings.warn(
            "Dimensions of the Jacobian were flipped because its shape "
            "didn't match the shape of the output of the function "
            "(probably because there was only 1 input qty)."
        )
        Jx = Jx.T

    return Jx


def calculate_flattened_corr(corrs, corr_between):
    """
    Combine correlation matrices for different input quantities, with a correlation
    matrix that gives the correlation between the input quantities into a full
    (flattened) correlation matrix combining the two.

    :param corrs: list of correlation matrices for each input quantity
    :type corrs: list[array]
    :param corr_between: correlation matrix between the input quantities
    :type corr_between: array
    :return: full correlation matrix combining the correlation matrices
    :rtype: array
    """
    totcorrlen = 0
    for i in range(len(corrs)):
        totcorrlen += len(corrs[i])
    totcorr = np.eye(totcorrlen)
    for i in range(len(corrs)):
        for j in range(len(corrs)):
            ist = i * len(corrs[i])
            iend = (i + 1) * len(corrs[i])
            jst = j * len(corrs[j])
            jend = (j + 1) * len(corrs[j])
            totcorr[ist:iend, jst:jend] = (
                corr_between[i, j] * corrs[i] ** 0.5 * corrs[j] ** 0.5
            )
    return totcorr


def separate_flattened_corr(corr, ndim):
    """
    Separate a full (flattened) correlation matrix into a list of correlation matrices
    for each output variable and a correlation matrix between the output variables.

    :param corr: full correlation matrix
    :type corr: array
    :param ndim: number of output variables
    :type ndim: int
    :return: list of correlation matrices for each output variable, correlation matrix between the output variables
    :type corrs: list[array]
    :rtype: list[array], array
    """

    corrs = np.empty(ndim, dtype=object)
    for i in range(ndim):
        corrs[i] = correlation_from_covariance(
            corr[
                int(i * len(corr) / ndim) : int((i + 1) * len(corr) / ndim),
                int(i * len(corr) / ndim) : int((i + 1) * len(corr) / ndim),
            ]
        )

    corrs_between = np.empty((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            corrs_between[i, j] = np.nanmean(
                corr[
                    int(i * len(corr) / ndim) : int((i + 1) * len(corr) / ndim),
                    int(j * len(corr) / ndim) : int((j + 1) * len(corr) / ndim),
                ]
                / corrs[i] ** 0.5
                / corrs[j] ** 0.5
            )

    return corrs, corrs_between


def select_repeated_x(x, u_x, param_fixed, i, repeat_dims, repeat_shape):
    """
    Select one (index i) of multiple repeated entries and return the input quantities and uncertainties for that entry.

    :param x: list of input quantities (usually numpy arrays)
    :type x: list[array]
    :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
    :type u_x: list[array]
    :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
    :type param_fixed: list of bools, optional
    :param i: index of the repeated measurement
    :type i: int
    :param repeat_dims: dimension along which the measurements are repeated
    :type repeat_dims: int
    :param repeat_shape: shape of measurements along which to select repeats
    :type repeat_shape: tuple
    :return: list of input quantities, list of uncertainties for single measurement
    :rtype: list[array]. list[array]
    """
    xb = np.zeros(len(x), dtype=object)
    u_xb = np.zeros(len(u_x), dtype=object)
    for j in range(len(x)):
        selected = False
        if param_fixed is not None:
            if param_fixed[j] == True:
                xb[j] = x[j]
                u_xb[j] = u_x[j]
                selected = True
        if not selected:
            index = list(np.ndindex(repeat_shape))[i]
            xb[j] = x[j]
            u_xb[j] = u_x[j]
            for idim in range(len(repeat_dims)):
                repeat_axis = repeat_dims[idim]
                ii = index[idim]
                if len(xb[j].shape) > repeat_axis:
                    if repeat_axis == 0:
                        xb[j] = xb[j][ii]
                        u_xb[j] = u_xb[j][ii]
                    elif repeat_axis == 1:
                        xb[j] = xb[j][:, ii]
                        u_xb[j] = u_xb[j][:, ii]
                    elif repeat_axis == 2:
                        xb[j] = xb[j][:, :, ii]
                        u_xb[j] = u_xb[j][:, :, ii]
                    else:
                        warnings.warn(
                            "The repeat axis is too large to be dealt with by the"
                            "current version of punpy."
                        )
                else:
                    xb[j] = xb[j][ii]
                    u_xb[j] = u_xb[j][ii]

    return xb, u_xb


def nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=True):
    """
    Find the nearest positive-definite matrix

    :param A: correlation matrix or covariance matrix
    :type A: array
    :return: nearest positive-definite matrix
    :rtype: array

    Copied and adapted from [1] under BSD license.
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [2], which
    credits [3].
    [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    try:
        return np.linalg.cholesky(A3)
    except:

        spacing = np.spacing(np.linalg.norm(A))

        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        if corr == True:
            A3 = correlation_from_covariance(A3)
            maxdiff = np.max(np.abs(A - A3))
            if maxdiff > diff:
                raise ValueError(
                    "One of the correlation matrices is not postive definite. "
                    "Correlation matrices need to be at least positive "
                    "semi-definite."
                )
            else:
                warnings.warn(
                    "One of the correlation matrices is not positive "
                    "definite. It has been slightly changed (maximum difference "
                    "of %s) to accomodate our method." % (maxdiff)
                )
                if return_cholesky:
                    return np.linalg.cholesky(A3)
                else:
                    return A3
        else:
            maxdiff = np.max(np.abs(A - A3) / (A3 + diff))
            if maxdiff > diff:
                raise ValueError(
                    "One of the provided covariance matrices is not postive "
                    "definite. Covariance matrices need to be at least positive "
                    "semi-definite. Please check your covariance matrix."
                )
            else:
                warnings.warn(
                    "One of the provided covariance matrix is not positive"
                    "definite. It has been slightly changed (maximum difference of "
                    "%s percent) to accomodate our method." % (maxdiff * 100)
                )
                if return_cholesky:
                    return np.linalg.cholesky(A3)
                else:
                    return A3


def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky

    :param B: matrix
    :type B: array
    :return: true when input is positive-definite
    :rtype: bool
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def correlation_from_covariance(covariance):
    """
    Convert covariance matrix to correlation matrix

    :param covariance: Covariance matrix
    :type covariance: array
    :return: Correlation matrix
    :rtype: array
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def uncertainty_from_covariance(covariance):
    """
    Convert covariance matrix to uncertainty

    :param covariance: Covariance matrix
    :type covariance: array
    :return: uncertainties
    :rtype: array
    """
    return np.sqrt(np.diag(covariance))


def convert_corr_to_cov(corr, u):
    """
    Convert correlation matrix to covariance matrix

    :param corr: correlation matrix
    :type corr: array
    :param u: uncertainties
    :type u: array
    :return: covariance matrix
    :rtype: array
    """
    return u.reshape((-1, 1)) * corr * (u.reshape((1, -1)))


def convert_cov_to_corr(cov, u):
    """
    Convert covariance matrix to correlation matrix

    :param corr: covariance matrix
    :type corr: array
    :param u: uncertainties
    :type u: array
    :return: correlation matrix
    :rtype: array
    """
    return 1 / u.reshape((-1, 1)) * cov / (u.reshape((1, -1)))
