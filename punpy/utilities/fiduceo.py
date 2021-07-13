"""Use Monte Carlo to propagate uncertainties"""

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def return_covariance(data_arr):
    data_shape=data_arr.values.shape
    n_meas=len(data_arr.values.flatten())
    corr_matrix=np.zeros((n_meas,n_meas))
    if data_arr.attrs["pixel_correlation_form"] == "rectangle_absolute":
        pix_corr_matrix = np.zeros((n_meas,n_meas))
        scales=data_arr.attrs["pixel_correlation_scales"]
        starts=data_arr.attrs["pixel_correlation_starts"]
        for start in starts:
            if "inf" in scales[0]:
                scale_pix_st=0
            else:
                scale_pix_st=start-scales[0]
            if "inf" in scales[1]:
                scale_pix_en = n_meas
            else:
                scale_pix_en = start+scales[1]

        pix_corr_matrix[scale_pix_st:scale_pix_en,:]=1
    else:
        print("pixel_correlation_form not recognised.")