# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:48:32 2020

@author: massonnetf
"""

#!/usr/bin/python
#
# Real-time predictions of sea ice extent using damped anomaly persistence
# forecasting. See below for the forecasting scheme.
#
# Author F. Massonnet (June 2020) as part of the APPLICATE project and 
#        participation to Sea Ice Outlook

# Imports of modules

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import matplotlib.pyplot as plt
import wget
import os
import csv
import pandas as pd
import calendar
import datetime
import matplotlib.dates  as mdates
import scipy.stats

# Close all figures
plt.close("all")    

# Domain and variable definition
hemi = "north"
diag = "extent"

# Image resolution
dpi = 300

# Estimation of background.
# Order of detrending. 0 = plain climatology, 1 = linear, 2 = quadratic, ...
order = 2

# -----
hemi_region = {"south": "Antarctic",
               "north": "Arctic"   ,
              }
rootdir = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/" + hemi +"/daily/data/"
filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"

if os.path.exists(filein):
    print("File already exists")
else:
    wget.download(rootdir + filein)


# Reading and storing the data. We are going to archive the daily extents
# in a 2-D numpy array with as many rows as there are years, and 365 columns
# (one for each day). The 29th of February of leap years are excluded for 
# ease of analysis
    
    
# Index for row in the input file
j = 0

rawdata = list()
with open(filein, 'r') as csvfile:
  obj = csv.reader(csvfile, delimiter = ',')
  nd = obj.line_num - 2 # nb data
  for row in obj:
    if j <= 1:
      print("Ignore, header")
    else:
      #extents.append(float(row[3]))
      #mydates.append(datetime.date(int(row[0]), int(row[1]), int(row[2])))
      #mydata.append([datetime.date(int(row[0]), int(row[1]), int(row[2])), \
      #                float(row[3])
      #              ])
      rawdata.append([datetime.date(int(row[0]), int(row[1]), int(row[2])), float(row[3])])
    j = j + 1
     
# Detect first and last years
yearb = rawdata[0][0].year
yeare = rawdata[-1][0].year
nyear = yeare - yearb + 1
nday  = 365
# Create data array
data = np.full((nyear, nday), np.nan)
# Fill it: loop over the raw data and store the extent value
for r in rawdata:
    
    # Ignore if 29th February
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
        
        data[row, col] = r[1]
    

# Damped anomaly persistence forecasts are expressed in the general form as
#
# X_{d+tau} = r * (X_d - X^b_d) + X^b_{d+tau}
#
# with d the initial time, tau the lead time,
# X^b the background estimate (climatology, trend, etc.)
# and r the correlation between X_{d+tau} and X_{d} in the record
#
# Accordingly, the forecast variance is [note that X_d and r are taken as 
# deterministically known]
# r^2 V(X^b_d) + V(X^b{d+tau}) - 2 COV(X^b_d, X^b_{d+tau})
# = (r S(X^b_d) - S(X^b_{d+tau})) ^ 2
# with S the standard deviation
        




# Function to generate forecasts based on input initial time and end time
def damped_anomaly_persistence_forecast(inidate):
    
    # inidate = start date
    # generates forecast until end of the current year
    
    
    
    # If leap: skip
    if inidate.day == 29 and inidate.month == 2:
        inidate += timedelta(days = 1)
        
    print("Forecasting " + str(inidate))
    # Year, month and day of initial condition
    iniyear = inidate.year
    inimon  = inidate.month
    iniday  = inidate.day
    
    # Index of the initial day. To avoid problems with leap years, only the month
    # and day are considered (the year is arbitrarily set to a non-leap one)
    
    inidoy  = (datetime.date(2005, inidate.month, inidate.day) - 
               datetime.date(2005, 1, 1)).days

    # First and last years used to construct the background
    # Choosing yearebk < iniyear allows to put one self in operational context 
    yearbbk = np.max((yearb, iniyear - 50)) #○rawdata[0][0].year
    yearebk = iniyear - 1
    
    if yearebk >= inidate.year:
        print("STOP: Interval for training ends after initial time")
        print("         (Use of future data in retrospective predictions)")
        stop()
    
    # Construction of the background and uncertainty
    background = np.full((nyear, nday), np.nan)
    std_bck        = np.full((nyear, nday), np.nan) 
    
    for d in range(nday):
        series = data[(yearbbk - yearb):(yearebk - yearb  + 1), d]
        years  = np.arange(yearb, yeare + 1)[(yearbbk - yearb):(yearebk- yearb  + 1)]   
        
        # Ignore nans
        years = years[  ~np.isnan(series)]
        series = series[~np.isnan(series)]
        
        # Create background all years including that of initialization
        p, cov = np.polyfit(years, series, order, cov = True)
        background[:, d] = np.polyval(p, np.arange(yearb, yeare + 1))
     
        
        # Uncertainty estimates
        XX = np.matrix([np.arange(yearb, yeare + 1) ** (order - i) for i in range(order + 1)]).transpose()
        covfit = XX * np.matrix(cov) * XX.transpose()
        std_bck[:, d] = np.array([np.sqrt(covfit[i, i]) for i in range(len(XX))])
        
        #plt.plot(np.arange(yearb, yeare + 1), data[:, d])
        #plt.plot(np.arange(yearb, yeare + 1), background[:, d])
        #plt.fill_between(np.arange(yearb, yeare + 1), background[:, d] - onesigma,
        #                                              background[:, d] + onesigma,
        #                                              color = "orange", alpha = 0.5, lw = 0)
        
        del years, series
    
    # Creation of anomalies
    # Anomalies are the raw values minus the background
    data_ano = data - background
    
    # Standard deviation of anomalies
    
    std_ano = np.full(nday, np.nan)
    for d in range(nday):
        series = data_ano[(yearbbk - yearb):(yearebk- yearb  + 1), d]
        std_ano[d] = np.nanstd(series) # np.sqrt(sum(~np.isnan(series)))

    # Computation of standard deviation of anomalies, and 
    # autocorrelation of anomalies at day d with inidoy
    # Also based on selected years
    r = np.full(nday, np.nan)
    for d in range(nday):
        x1tmp = data_ano[(yearbbk - yearb):(yearebk- yearb  + 1), d]
        x2tmp = data_ano[(yearbbk - yearb):(yearebk- yearb  + 1), inidoy]
        
        x1 = x1tmp[~np.isnan(x1tmp * x2tmp)]
        x2 = x2tmp[~np.isnan(x1tmp * x2tmp)]
        
        r[d] = np.corrcoef(x1, x2)[0, 1]
        del x1tmp, x2tmp, x1, x2
    
    # Construction of forecasts
    forecast       = list()
    std_forecast   = list()
    dates_forecast = list()
    verif_data     = list()
    for d in range(inidoy, nday):
        forecast.append(r[d] * data_ano[iniyear - yearb, inidoy] + 
                        background[iniyear - yearb, d])
        std_forecast.append(np.abs(r[d] * std_bck[iniyear - yearb, inidoy] - std_bck[iniyear - yearb, d]))
        
        # Create date time axis for forecast
        tmp = datetime.date(2005, 1, 1) + timedelta(days = d)
        dates_forecast.append(datetime.date(iniyear, tmp.month, tmp.day))
        
        # Output verification data as well
        verif_data.append(data[iniyear - yearb, d])
        
        del tmp
    
    forecast = np.array(forecast)
    std_forecast = np.array(std_forecast)
    
    
    # Return values
    return dates_forecast, forecast, std_forecast, verif_data
    


startdates = [datetime.date(y, 6, 1) for y in np.arange(1991, 2020 + 1)]

#startdates = [datetime.date(2020, 1, 1) + timedelta(days = d) \
#              for d in range((datetime.date(2020, 6, 1) - datetime.date(2020, 1, 1)).days)]




forecast_mean = list()
forecast_year = list()
verif_mean    = list()
# Next, we loop over all initial times and issue a lead-time dependent forecast
for inidate in startdates:
    
    dates_forecast, forecast, std_forecast, verif_data = damped_anomaly_persistence_forecast(inidate)
    
    # Extract september mean
    forecast_mean.append(np.mean([f for d, f in zip(dates_forecast, forecast) if d.month == 9]))
    forecast_year.append(inidate.year)
    # In data as well
    verif_mean.append(np.mean([v for d, v in zip(dates_forecast, verif_data) if d.month == 9]))
    
    # Prepare plots
    # -------------
        
    # Plot current year
    plt.close("all")
    fig, ax= plt.subplots(1, 1, figsize = (6, 6), dpi = dpi)
    
    a = ax
    a.grid()
    a.plot([r[0] for r in rawdata], [r[1] for r in rawdata] , color = [0.0, 0.0, 0.0], label = "Observed")
    a.plot(dates_forecast, forecast, "lightblue", label = "Best forecast estimate")
    
    # Plot uncertainty
    myzs = [50, 95, 99] # quantiles
    zs = [scipy.stats.norm.ppf((100 - (100 - q) / 2 ) / 100) for q in myzs]
    
    for j, z in enumerate(zs):
        a.fill_between(dates_forecast, forecast - z * std_forecast, forecast + z * std_forecast, color = "blue", 
                     alpha = 0.8 / (j + 1), edgecolor = "none", zorder = 0, 
                     label = str(myzs[j]) + "% pred. interval", lw = 0)
    
    
    # Add all-time minimum
    alltimemin = np.nanmin(data)
    a.plot((-1e9, 1e9), (alltimemin, alltimemin), "r--", label = "All-time minimum")
    
    a.legend(loc = "upper right")
    a.set_xlim(datetime.date(inidate.year, 1, 1), datetime.date(inidate.year, 12, 31))
    a.set_ylim(0, 16.5)
    a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
    a.set_ylabel("$10^6$ km²")
    a.set_title("Damped anomaly persistence forecast of\ndaily Arctic sea ice extent\ninitialized on " + str(inidate))
    fig.autofmt_xdate(rotation=45)
    fig.savefig("./figs/forecast_order" + str(order) + "_" + 
                str(inidate) + ".png")
    

# Verification and bias correction
fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 300)# dpi)

ax = axes[0]
ax.grid()
if hemi == "north":
    ax.set_ylim(0.0, 8.0)
elif hemi == "south":
    ax.set_ylim(15.0, 22.0)
    
ax.plot(forecast_year, forecast_mean, color = [1, 0.5, 0.0], marker = "s", 
        label = "Hindcasts")
ax.plot(forecast_year, verif_mean, color = [0.0, 0.0, 0.0],  marker = "s", 
        label = "Verification data")
ax.set_ylabel("10$^6$ km$^2$")
ax.set_title("September mean sea ice extent")
ax.legend()



ax = axes[1]
ax.grid()
ax.plot((-1e9, 1e9), (-1e9, 1e9), color = "k", lw = 1, label = "y = x")

ax.plot((forecast_mean[-1], forecast_mean[-1]), (-1e9, 1e9), "red", 
        label = forecast_year[-1])



ax.set_title("Verification and bias-correction")
if hemi == "north":
    ax.set_xlim(2.0, 8.0)
    ax.set_ylim(2.0, 8.0)
elif hemi == "south":
    ax.set_xlim(17.0, 20.0)
    ax.set_ylim(17.0, 20.0)
ax.set_xlabel("Forecast [10$^6$ km$^2$]")
ax.set_ylabel("Verification [10$^6$ km$^2$]")
ax.scatter(forecast_mean, verif_mean, 200, marker = "s", color = "green", \
           alpha = 0.5, label = "Hindcasts")
[ax.text(   forecast_mean[j], verif_mean[j], str(forecast_year[j]), 
         color = "white", ha = "center", va = "center", fontsize = 5) \
    for j in range(len(forecast_mean)  -1)]
    
# Regress forecasts on verification to determine bias-corrected forecast
x = np.array(forecast_mean[:-1])
y = np.array(verif_mean[:-1])
n = len(x)
xbar = np.mean(x)
ybar = np.mean(y)
xtil = x - xbar
ytil = y - ybar
ahat = np.sum(xtil * ytil) / np.sum(xtil ** 2)
bhat = ybar - ahat * xbar
yhat = ahat * x + bhat
res = y - yhat
se2 = 1.0 / (n - 2) * np.sum(res ** 2)

# Detrend
xd = x - np.polyval(np.polyfit(np.arange(len(x)), x, 2), np.arange(len(x)))
yd = y - np.polyval(np.polyfit(np.arange(len(y)), y, 2), np.arange(len(y)))

# Prediction Interval
def spred(xin):
    return np.sqrt(se2 * (1 + 1 / n + (xin - xbar) ** 2 / np.sum(xtil ** 2)))

xx = np.arange(0.0, 10.0)
fit = ahat * xx + bhat
ax.plot(xx, fit, color = "green", label = "Regression")
ax.fill_between(xx, fit - 1.96 * spred(xx), fit + 1.96 * spred(xx), 
                color = "green", alpha = 0.2, lw = 0, 
                label = "Prediction\n95% confidence interval")

# Display correlations
ax.text(5.5, 3.5, "$r$ = " + str(np.round(np.corrcoef(x, y)[0, 1], 2)) +
        "\n(detrended: " + str(np.round(np.corrcoef(xd, yd)[0, 1], 2)) + ")")

# Print re-processed forecast
ax.text(5.5, 2.5    , str(forecast_year[-1]) + " forecast:\n" + 
        str(np.round(ahat * forecast_mean[-1] + bhat, 2)) + 
        " [" + str(np.round(ahat * forecast_mean[-1] + bhat - 
                            1.96 * spred(x[-1]), 2)) +  " - " +
               str(np.round(ahat * forecast_mean[-1] + bhat + 
                            1.96 * spred(x[-1]), 2)) + "]\n" + "$10^6$ km$^2$")

ax.legend(loc = "upper left")

fig.savefig("./figs/forecast_mean.png")




# Presentation of forecast as a PDF
fig, ax = plt.subplots(1, 1, figsize = (5, 3), dpi = 300)
ax.grid()
if hemi == "north":
    ax.set_xlim(2.0, 10.0)
elif hemi == "south":
    ax.set_xlim(15.0, 22.0)
ax.set_ylim(-0.2, 1.0)
ax.set_xlabel("September mean sea ice extent [10$^6$ km$^2$]")

mu = ahat * forecast_mean[-1] + bhat
sig= spred(x[-1])

# All time minimum until now
alltimemin = np.min(verif_mean[:-1])

# Terciles
terciles = np.percentile(verif_mean[:-1], [100 / 3, 200 / 3])


xx = np.linspace(0.0, 30.0, 10000)
ax.set_ylabel("Density [10$^6$ km$^2$]$^{-1}$")
ax.plot(xx, scipy.stats.norm.pdf(xx, mu, sig), color = "k", label = "Forecast PDF")

# All time min
ax.plot((alltimemin, alltimemin), (-10, 10), "r", label = "All-time min.")
ax.arrow(alltimemin, - 0.07, -0.4, 0.0, color = "r", head_width = 0.05)
# Print CDF
cdf = scipy.stats.norm.cdf(alltimemin, mu, sig) * 100.0
ax.text(alltimemin, -0.1, str(np.round(cdf, 1)) + " % ", va = "top", ha = "right", color = "red", alpha = 0.8)

#ax.plot((np.nanmax(verif_mean), np.nanmax(verif_mean)), (-10, 10), "g", label = "All-time maximum")
#ax.plot((np.nanmean(verif_mean), np.nanmean(verif_mean)), (-10, 10), "g", label = "Mean")
#ax.fill_between(np.linspace(0.0, alltimemin, 10000), 
#                scipy.stats.norm.pdf(np.linspace(0.0, alltimemin, 10000), mu, sig), color = "orange", alpha = 0.5)
# First tercile
ax.fill_between(np.linspace(0.0, terciles[0], 10000), 
                scipy.stats.norm.pdf(np.linspace(0.0, terciles[0], 10000), \
                                     mu, sig), color = "red", alpha = 0.5, lw = 0.0, \
                                     label = "1st tercile (" + \
                                     str(np.round(scipy.stats.norm.cdf(terciles[0], mu, sig) * 100.0, 1)) + " %)")
# Second tercile
ax.fill_between(np.linspace(terciles[0], terciles[1], 10000), 
                scipy.stats.norm.pdf(np.linspace(terciles[0], terciles[1], \
                                    10000), mu, sig), color = "black",
                                                 alpha = 0.1, lw = 0.0,
                                                 label = "2nd tercile (" \
                                                 + str(np.round((scipy.stats.norm.cdf(terciles[1], mu, sig) - scipy.stats.norm.cdf(terciles[0], mu, sig)) * 100, 1)) + " %)")
# Third tercile
ax.fill_between(np.linspace(terciles[1], 30, 10000), 
                scipy.stats.norm.pdf(np.linspace(terciles[1], 30, 10000), 
                                     mu, sig), color = "blue", alpha = 0.5, lw = 0.0, 
                                     label = "3rd tercile (" + str(np.round((1 - scipy.stats.norm.cdf(terciles[1], mu, sig)) * 100, 1)) + " %)")
ax.legend(ncol = 1)
ax.set_title("Forecast PDF (initialized " + str(inidate) + ")")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("./figs/presentation.png")
