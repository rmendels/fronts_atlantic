from Canny2 import *
from canny_lib import *
from cartopy import crs
import cmocean
import cv2
import numpy as np
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from skimage.feature import canny
import xarray as xr

def isleap(year):
    from datetime import date, datetime, timedelta
    try:
        date(year,2,29)
        return True
    except ValueError: return False

mur_file_name = '20231015090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
# extract data from MUR file
sst_mur, lon_mur, lat_mur = extract_mur(mur_file_name)
# create front netcdf file to be written to
#front_file = create_canny_nc(file_year, file_month, file_day)
#need to do two calls because of change
edges, x_gradient, y_gradient, magnitude = myCanny(sst_mur, ~sst_mur.mask)
contours = my_contours(edges)
contour_edges, contour_lens = contours_to_edges(contours, edges.shape)
contour_edges = ma.array(contour_edges, mask = sst_mur.mask)

my_title = 'October 15 2023 MUR SST x_gradient, Sigma = 10., Threshold = (.8, .9)'
plot_canny_gradient(x_gradient, contour_edges, lat_mur, lon_mur, title = my_title, fig_size = ([12, 6]) )

my_title = 'October 15 2023 MUR SST y_gradient, Sigma = 10., Threshold = (.8, .9)'
plot_canny_gradient(y_gradient, contour_edges, lat_mur, lon_mur, title = my_title, fig_size = ([12, 6]) )

my_title = 'October 15 2023 MUR SST magnitude, Sigma = 10., Threshold = (.8, .9)'
plot_canny_gradient(magnitude, contour_edges, lat_mur, lon_mur, title = my_title, fig_size = ([12, 6]) )

my_title = 'October 15 2023 MUR SST Contours, Sigma = 10., Threshold = (.8, .9), Min_Len = 20'
plot_canny_contours3(sst_mur, contour_edges, contour_lens, lat_mur, lon_mur, title = my_title, fig_size = ([12, 8]))




root = Dataset(front_file, 'a')
file_time = root.variables['time']
file_time[0] = file_date_num
file_edges = root.variables['edges']
file_x_gradientient = root.variables['x_gradientient']
file_y_gradientient = root.variables['y_gradientient']
file_magnitude_gradient = root.variables['magnitude_gradient']
file_edges[0, 0, :, :] = contour_edges[:, :]
file_x_gradientient[0, 0, :, :] = x_gradientient[:, :]
file_y_gradientient[0, 0, :, :] = y_gradientient[:, :]
file_magnitude_gradient[0, 0, :, :] = magnitude[:, :]
root.close()


