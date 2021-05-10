# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# F. Massonnet, 2020
import xarray as xr
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
#for interpolation (you will have to install pyresample first)
import pyresample


# 
filein = "grid_c2Sgpod_z_sn_201204.nc"
varin = "sea_ice_thickness"
varout = "vt_i"  #rfb -- freeboard, at_i -- concentration, vt_i -- thickness

#=============================================

f = Dataset(filein, mode = "r")
field = f.variables[varin][0, :, :]
field_sd = f.variables[varin + "_unc"][0, :, :]
lat = f.variables["lat"][:]
lon = f.variables["lon"][:]
f.close()


# Load lat on of target grid
f = Dataset("mesh_mask_nemo.N3.6_ORCA1L75.nc", mode = "r")
lat_target = f.variables["gphit"][0, :, :]
lon_target = f.variables["glamt"][0, :, :]
f.close()



# Create a pyresample object holding the origin grid:
orig_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)

#Create another pyresample object for the target (curvilinear) grid:
targ_def = pyresample.geometry.SwathDefinition(lons=lon_target, \
                                               lats=lat_target)


out = pyresample.kd_tree.resample_nearest(orig_def, field, \
            targ_def, radius_of_influence = 500000,     fill_value=None)

out_sd = pyresample.kd_tree.resample_nearest(orig_def, field_sd, \
            targ_def, radius_of_influence = 500000,     fill_value=None)
# Write NetCDF
ncfile = Dataset("out.nc", mode = "w", format = 'NETCDF4_CLASSIC')
x_dim = ncfile.createDimension("x", 362)
y_dim = ncfile.createDimension("y", 292)
time_dim = ncfile.createDimension("time", None) # unlimited axis
nav_lat = ncfile.createVariable("nav_lat", np.float32, ("y", "x",))
nav_lon = ncfile.createVariable("nav_lon", np.float32, ("y", "x",))
output =     ncfile.createVariable(varout,   np.float64,("y","x",)) 
   # note: unlimited dimension is leftmost
output_sd =   ncfile.createVariable(varout + "_sd",   np.float64,("y","x",))
nav_lat[:, :] = lat_target
nav_lon[:, :] = lon_target
output[:, :] = out
output_sd[:, :] = out_sd

ncfile.close()
