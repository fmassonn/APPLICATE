# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:48:32 2020

@author: massonnetf
"""

#!/usr/bin/env/python3
#
#
# Author F. Massonnet (June 2020) as part of the APPLICATE project and 
#        participation to Sea Ice Outlook


# Imports of modules
# ------------------

import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '': #or os.name == "posix":
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import font_manager as fm, rcParams
import numpy as np
import matplotlib.pyplot as plt
import wget
import csv
import pandas as pd
import calendar
import datetime
import matplotlib.dates  as mdates
import scipy.stats

from PIL import Image, ImageFont, ImageDraw

from datetime import timedelta
from dateutil.relativedelta import relativedelta

# Close all figures
# -----------------
plt.close("all")    

# Create output folder for figures if does not exist
# --------------------------------------------------
if not os.path.isdir("./figs"):
    os.makedirs("./figs")

# Script parameters
# -----------------
# Mode:
# either "oper" = downloads latest data and produces most up-to-date figure
#        "YYYY" = does some year (but always in operational mode, that is,
#                 not using future data)
#        "econ"= provides economical perspective on the forecast
mode = "oper"
freq = "daily"
# Domain and variable definition
hemi = "north"
diag = "extent"

# Image resolution
dpi = 300

# Estimation of background.
# Order of detrending. 0 = plain climatology, 1 = linear, 2 = quadratic, ...
order = 2

type_trend = ["Climatological", "Linear", "Quadratic"]

# Retrieving the data source file
# -------------------------------

hemi_region = {"south": "Antarctic",
               "north": "Arctic"   ,
              }
rootdir = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/" + hemi + \
           "/daily/data/"
filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"

if mode == "oper":
    if os.path.exists("./data/" + filein):
        os.remove("./data/" + filein)
    wget.download(rootdir + filein, out = "./data/")
    
elif os.path.exists("./data/" + filein):
    print("File already exists, not downloading")
else:
    wget.download(rootdir + filein, out = "./data/")


# Reading the data
# ----------------

# Reading and storing the data. We are going to archive the daily extents
# in a 2-D numpy array with as many rows as there are years, and 365 columns
# (one for each day). The 29th of February of leap years are excluded for 
# ease of analysis
    
    
# Index for looping throught rows in the input file
j = 0

rawdata = list()
with open("./data/" + filein, 'r') as csvfile:
  obj = csv.reader(csvfile, delimiter = ",")
  nd = obj.line_num - 2 # nb data
  for row in obj:
    if j <= 1:
      print("Ignore, header")
    else:
      rawdata.append([datetime.date(int(row[0]), int(row[1]), int(row[2])), float(row[3])])
    j = j + 1


# Detect first and last years of the sample
yearb = 1980 # First full year ; rawdata[0][0].year
yeare = rawdata[-1][0].year - 1 # - 1  to not take ongoing years
nyear = yeare - yearb + 1
nday  = 365

# Create data array
data = np.full((nyear, nday), np.nan)

# Fill it: loop over the raw data and store the extent value
for r in rawdata:
    
    # Ignore if 29th February
    # Day of year
    doy = int(r[0].strftime('%j'))
    if not (calendar.isleap(r[0].year) and doy == 60):
        # Match year
        row = r[0].year - yearb
        # Match day of year. Number of days counted from 1 to 365
        # If leap year and after 29th Feb, subtract 1 to column to erase 29th Feb

        if calendar.isleap(r[0].year) and doy > 60:
            doy -= 1
        # To match Pythonic conventions    
        col = doy - 1
        
        if r[0].year >= yearb and r[0].year <= yeare:
            data[row, col] = r[1]

# Figure

fig, ax = plt.subplots(1, 1, figsize = (4, 3), dpi = dpi)
#fig.set_facecolor("white")
plt.style.use('seaborn-deep')
xmin = np.min(np.nanmin(data[1:-1, :], axis = 1))
xmax = np.max(np.nanmin(data[1:-1, :], axis = 1))
for y in np.arange(yearb, yeare + 1):

    value = np.nanmin(data[y - yearb, :])
    color = plt.cm.RdBu(int((value - xmin) * 255 / (xmax - xmin)))[:3]
    days = np.arange(365)
    #ax.plot3D(days, (yeare - y) * np.ones(len(days)),  \
    #         data[y - yearb, :], color = color, \
    #             alpha = 0.2 + 0.8 * (y - yearb) / (yeare - yearb) )
    ax.plot(days,  \
             data[y - yearb, :], color = color)
    print(color)
    if y == 2012:
        col2012 = color
    if y == 2020:
        col2020 = color
    if y == 1980:
        col1980 = color

# Legends

ax.text(175, 1.4, "2012", color = col2012 )
ax.plot((220, 240), (2.0, 3.3), color = col2012, lw = 1)
ax.text(295, 1.4, "2020", color = col2020 )
ax.plot((290, 265), (2.0, 4.0), color = col2020, lw = 1)
ax.text(225, 10.4, "1980", color = col1980 )
ax.plot((240, 250), (10.0, 7.5), color = col1980, lw = 1)

#ax.set_facecolor("white")

# Ticks months
ndpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
monnam = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ndpmc = np.cumsum(np.array(ndpm))

ax.set_xticks(ndpmc)
ax.set_xticklabels([m[0] + "  " for m in monnam], ha = "right")
ax.set_xlim(1.0, 365.0)
ax.set_ylabel(r"million square km", rotation = 90)
ax.set_yticks([0, 5, 10, 15])
ax.grid()

#ax.yaxis.set_label_coords(-0.2,1.0)
ax.set_ylim(0.0, 18.0)





ax.set_title("Arctic sea ice extent")
#ax.yaxis.label.set_color("white")
#ax.zaxis.label.set_color("white")
#plt.axis("off")

fig.tight_layout()
plt.savefig("./ArcticStripes.png", dpi = dpi)
plt.savefig("./ArcticStripes.pdf")






