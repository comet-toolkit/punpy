"""Use Monte Carlo to propagate uncertainties"""

import numpy as np
import numdifftools as nd

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def calculate_Jacobian(fun,x):
    """
    Calculate the local Jacobian of function y=f(x) for a given value of x

    :param fun: measurement function
    :type fun: function
    :param x: local values of input quantities
    :type x: array
    :return: Jacobian
    :rtype: array
    """
    Jfun = nd.Jacobian(fun)
    Jx = Jfun(x.flatten())
    if len(Jx)!=len(fun(x).flatten()):
        print("Dimensions of the Jacobian were flipped because its shape didn't match "
              "the shape of the output of the function. (probably because there was "
              "only 1 input qty)")
        Jx=Jx.T

    #print("Jacobian calculation took:",time.time()-timeb,timeb-timea)
    return Jx

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
