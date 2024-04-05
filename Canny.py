# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

im = np.zeros((128, 128))
im[32:-32, 32:-32] = 1
im = ndimage.rotate(im, 15, mode='constant')
im = ndimage.gaussian_filter(im, 4)
im += 0.2 * np.random.random(im.shape)
edges1 = feature.canny(im)


import numpy as np
from skimage import feature
from scipy import ndimage
from scipy import signal
import xarray as xr
# from PIL import Image
from netCDF4 import Dataset
import matplotlib.pyplot as plt
root = Dataset('/Users/rmendels/WorkFiles/fronts/20171213090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
lat = root.variables['lat'][:]
lon = root.variables['lon'][:]
lat_min = 22.
lat_max = 51.
lon_min = -135.
lon_max = -105.
lat_min_index = np.argwhere(lat == lat_min)
lat_min_index = lat_min_index[0, 0]
lat_max_index = np.argwhere(lat == lat_max)
lat_max_index = lat_max_index[0, 0]
lon_min_index = np.argwhere(lon == lon_min)
lon_min_index = lon_min_index[0, 0]
lon_max_index = np.argwhere(lon == lon_max)
lon_max_index = lon_max_index[0, 0]
lon_1km = lon[lon_min_index:lon_max_index + 1]
lat_1km = lat[lat_min_index:lat_max_index + 1]
sst = root.variables['analysed_sst'][0, lat_min_index:lat_max_index + 1, lon_min_index:lon_max_index + 1 ]
sst = np.squeeze(sst)
sst = sst - 273.15
root.close()
sst_min = np.nanmin(sst)
sst_max = np.nanmax(sst)
sst_median = np.nanmedian(sst)
tol = .2
lower = max(sst_min, (1.0 - tol) * sst_median)
upper = min(sst_max, (1.0 + tol) * sst_median)
signal.medfilt2d(input, kernel_size=3)[source]
edges = feature.canny(sst)
plt.imshow(edges, cmap=plt.cm.gray)
#im = Image.fromarray(sst)
#edges = feature.canny(im)
#plt.imshow(edges, cmap=plt.cm.gray)

import numpy as np
import numpy.ma as ma
from skimage import feature
from scipy import ndimage
from scipy import signal
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs
import cmocean
hv.extension('matplotlib', 'bokeh')

root = Dataset('/Users/rmendels/WorkFiles/fronts/erdGAssta1day_7d9c_7be5_8ad0.nc')
sst_all = root.variables['sst'][:, :, :, :]
lats = root.variables['latitude'][:]
lons = root.variables['longitude'][:]
root.close()
sst_all = np.squeeze(sst_all)
sst_array = xr.open_dataarray('/Users/rmendels/WorkFiles/fronts/erdGAssta1day_7d9c_7be5_8ad0.nc')
#lat = sst_array.latitude.values
#lon = sst_array.longitude.values
#sst_all = sst_array.values
#sst_all = ma.array(sst_all, mask = np.isnan(sst_all), fill_value = np.nan)
#sst_all = np.squeeze(sst_all)
sst = sst_all[0, :, :]
sst_min = np.nanmin(sst)
sst_max = np.nanmax(sst)
sst_median = np.nanmedian(sst.filled(np.nan))
tol = .10
lower = max(sst_min, (1.0 - tol) * sst_median)
upper = min(sst_max, (1.0 + tol) * sst_median)
sst_med = signal.medfilt2d(sst.filled(np.nan), kernel_size=3)
sst_med = ma.array(sst_med, mask = np.isnan(sst_med), fill_value = np.nan)
sst_min1 = np.nanmin(sst_med)
sst_max1 = np.nanmax(sst_med)
sst_median1 = np.nanmedian(sst_med.filled(np.nan))
lower1 = max(sst_min1, (1.0 - tol) * sst_median1)
upper1 = min(sst_max1, (1.0 + tol) * sst_median1)
plt.imshow(edges, cmap=plt.cm.gray)
sigma = 7.
edges = feature.canny(sst, sigma = sigma, mask = ~sst.mask)
plt.imshow(edges, cmap=plt.cm.gray)
edges = feature.canny(sst, mask = ~sst.mask, low_threshold = lower, high_threshold = upper)
plt.imshow(edges, cmap=plt.cm.gray)
edges1 = edges.astype(int)
edges1 = ma.array(edges1, mask = (edges1 == 0))
edges1_xr = xr.DataArray(edges1, coords=[lats, lons], dims=['latitude', 'longitude'], name = 'edge')
sst_xr = xr.DataArray(sst, coords=[lats, lons], dims=['latitude', 'longitude'], name = 'sst')
kdims = ['latitude', 'longitude']
vdims = ['sst']
xr_dataset = gv.Dataset(sst_xr, kdims=kdims, vdims=vdims)
from skimage.util import dtype_limits
low_threshold = 0.1 * dtype_limits(sst, clip_negative=False)[1]
high_threshold = 0.2 * dtype_limits(sst, clip_negative=False)[1]

import numpy as np
import numpy.ma as ma
from skimage import feature
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cmocean
root = Dataset('/Users/rmendels/WorkFiles/fronts/20171213090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
lats = root.variables['lat'][:]
lons = root.variables['lon'][:]
lat_min = np.where(lats == 21)
lat_min = int(lat_min[0])
lat_max = np.where(lats == 55)
lat_max = int(lat_max[0])
lon_min = np.where(lons == -135.)
lon_min = int(lon_min[0])
lon_max = np.where(lons == -105)
lon_max = int(lon_max[0])
sst = root.variables['analysed_sst'][0, (lat_min -1):lat_max, (lon_min - 1):lon_max]
root.close()
lat_grid = lats[ (lat_min -1):lat_max]
lon_grid = lons[ (lon_min - 1):lon_max]
sst_xr = xr.DataArray(sst, coords=[lat_grid, lon_grid], dims=['latitude', 'longitude'], name = 'sst')
sigma = 15.
edges = feature.canny(sst, sigma = sigma, mask = ~sst.mask)
plt.imshow(edges, cmap=plt.cm.gray)
edges1 = edges.astype(int)
edges1 = ma.array(edges1, mask = (edges1 == 0))
edges1_xr = xr.DataArray(edges1, coords=[lat_grid, lon_grid], dims=['latitude', 'longitude'], name = 'edge')

plt.figure
sst_xr.plot(cmap = cmocean.cm.thermal)
edges1_xr.plot(cmap=plt.cm.gray)
plt.show()

