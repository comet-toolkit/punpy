

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

def bell_relative(len,n,sigma=None):
    if sigma is None:
        sigma=(n/2-1)/3**0.5
    corr=np.eye(len)
    for i in range(n):
        idx_row = np.arange(i,len)
        idx_col = np.arange(len-i)
        corr[idx_row,idx_col]=np.exp(-0.5*(i/sigma)**2)
        corr[idx_col,idx_row]=np.exp(-0.5*(i/sigma)**2)
    return corr

def triangular_relative(len,n):
    corr=np.eye(len)
    for i in range(n):
        idx_row = np.arange(i,len)
        idx_col = np.arange(len-i)
        corr[idx_row,idx_col]=(n-i)/n
        corr[idx_col,idx_row]=(n-i)/n
    return corr


