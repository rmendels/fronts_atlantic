"""Generates seafloor gradient NetCDF files from ETOPO15 data using Canny edge detection.

This script processes ETOPO15 bathymetric data to create seafloor gradient information by employing
Canny edge detection techniques. The primary objective is to analyze and visualize seafloor topography
through gradient and edge detection analysis, highlighting significant features such as underwater
mountains, trenches, and ridges.

The script performs the following operations:
1. Extracts depth, longitude, and latitude data from the 'etopo_22.nc' NetCDF file using a custom 
   library function `extract_etopo15`.
2. Utilizes the extracted data to calculate depth gradients, employing another custom function 
   `create_depth_gradient_nc` for initial processing.
3. Applies Canny edge detection on the depth data to calculate x and y gradients and the overall 
   magnitude of the seafloor's gradient, using a custom function `myCanny` from the `Canny2` module.
4. Writes the results of the Canny edge detection (x gradient, y gradient, and magnitude) to a new 
   NetCDF file using the `write_depth_gradient_nc` function.

Dependencies:
    - canny_lib: Custom library assumed to contain the `extract_etopo15` and `create_depth_gradient_nc` functions.
    - Canny2: Custom module assumed to contain the `myCanny` function for Canny edge detection.
    - netCDF4: For reading NetCDF files.
    - numpy: For numerical operations, especially array manipulations.
    - numpy.ma: For handling masked arrays, particularly useful in masking operations on depth data.
    - xarray: Although not explicitly used in the provided snippet, it is listed as an import, indicating its potential use in extended functionalities or data handling.

Example:
    To use this script, ensure that the 'etopo_22.nc' NetCDF file is located in the specified directory and 
    that all custom libraries (`canny_lib` and `Canny2`) are accessible in the Python environment. The script
    can be executed directly to generate output NetCDF files containing seafloor gradient data in the 
    designated output directory.

Attributes:
    f_name (str): The base filename ('etopo_22') for both input and output NetCDF files, used to identify the dataset.
    base_dir (str): The directory path where input data is located and output NetCDF files will be saved.
    depth (MaskedArray): A masked array of depth values extracted from the ETOPO15 dataset.
    lon_etopo (ndarray): An array of longitude values associated with the ETOPO15 data.
    lat_etopo (ndarray): An array of latitude values associated with the ETOPO15 data.
    x_gradient, y_gradient, magnitude (ndarray): Arrays representing the x and y gradients and the overall gradient magnitude of the seafloor, as calculated by Canny edge detection.

Note:
    The script assumes the existence and correct functioning of the custom libraries (`canny_lib` and `Canny2`). It is also tailored for a specific dataset and output requirements. For operational use, especially in different or more dynamic environments, further modifications and error handling mechanisms may be necessary.
"""

from calendar import monthrange
from canny_lib import *
from datetime import date, datetime, timedelta
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import numpy.ma as ma

def isleap(year):
    from datetime import date, datetime, timedelta
    try:
        date(year,2,29)
        return True
    except ValueError: return False

file_date = datetime.now() + timedelta(days = -2)
file_year = file_date.year
file_month = file_date.month
file_day = file_date.day
print('file_year: ' + str(file_year))
print('file_month: ' + str(file_month))
print('file_day: ' + str(file_day))
c_file_year = str(file_year)
c_file_month = str(file_month).rjust(2,'0')
c_file_day = str(file_day).rjust(2,'0')
file_date = datetime(file_year, file_month, file_day, 0)
file_date_num = date2num(file_date, units = 'Hour since 1970-01-01T00:00:00Z')
# create name of MUR file
mur_file_end = '090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
mur_file_name = c_file_year + c_file_month + c_file_day + mur_file_end
sst_mur, lon_mur, lat_mur = extract_mur(mur_file_name)
# create front netcdf file to be written to
front_file = create_canny_nc(file_year, file_month, file_day)
edges, x_gradient, y_gradient, magnitude = myCanny(sst_mur, ~sst_mur.mask)
contours = my_contours(edges)
contour_edges, contour_lens = contours_to_edges(contours, edges.shape)
contour_edges = ma.array(contour_edges, mask = sst_mur.mask)
root = Dataset(front_file, 'a')
file_time = root.variables['time']
file_time[0] = file_date_num
file_edges = root.variables['edges']
file_x_gradient = root.variables['x_gradient']
file_y_gradient = root.variables['y_gradient']
file_magnitude_gradient = root.variables['magnitude_gradient']
file_edges[0, 0, :, :] = contour_edges[:, :]
file_x_gradient[0, 0, :, :] = x_gradient[:, :]
file_y_gradient[0, 0, :, :] = y_gradient[:, :]
file_magnitude_gradient[0, 0, :, :] = magnitude[:, :]
root.close()
my_cmd = 'rsync -avh ' + front_file + ' cwatch@192.168.31.15:' + front_file
os.system(my_cmd)

