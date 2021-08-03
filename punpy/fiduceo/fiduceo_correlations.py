"""Use Monte Carlo to propagate uncertainties"""

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"



def calculate_corr(var,form,dataset,repeat_dims):
    data = dataset[var].values
    if len(data.shape)>1:
        if repeat_dims is None:
            data=dataset[var].values
        elif repeat_dims == 0:
            data=dataset[var].values[0,:]
        elif repeat_dims == 1:
            data=dataset[var].values[:,0]
        else:
            print("this repeat_dims option ins not yet implemented.")
    return np.eye(len(data.flatten()))

#
# def return_covariance(data_arr):
#     data_shape = data_arr.values.shape
#     n_meas = len(data_arr.values.flatten())
#     corr_matrix = np.zeros((n_meas,n_meas))
#     if data_arr.attrs["pixel_correlation_form"] == "rectangle_absolute":
#         pix_corr_matrix = np.zeros((n_meas,n_meas))
#         scales = data_arr.attrs["pixel_correlation_scales"]
#         starts = data_arr.attrs["pixel_correlation_starts"]
#         for start in starts:
#             if "inf" in scales[0]:
#                 scale_pix_st = 0
#             else:
#                 scale_pix_st = start-scales[0]
#             if "inf" in scales[1]:
#                 scale_pix_en = n_meas
#             else:
#                 scale_pix_en = start+scales[1]
#
#             pix_corr_matrix[scale_pix_st:scale_pix_en,:] = 1
#     else:
#         print("pixel_correlation_form not recognised.")
#
#
# def return_correlation(self):
#
#     corr_x = []
#     for ivar,var in enumerate(vars):
#         dims = xx
#         propdims = []
#         propform = []
#         propparams = []
#         i_repeat_dims = []
#         for idim,dim in enumerate(dims):
#             if dim not in repeat_dims:
#                 propdims.append(idim)
#                 propform.append(form[dim])
#                 propparams.append(propparams[dim])
#             else:
#                 i_repeat_dims.append(idim)
#
#         if all(propform == "random"):
#             corr_x.append("rand")
#
#         elif all(propform == "rectangular_absolute"):
#             corr_x.append("syst")
#
#         else:
#             corr_x = self.calculate_corr_from_form(propdims,propform,propparams)
#
#         return corr_x