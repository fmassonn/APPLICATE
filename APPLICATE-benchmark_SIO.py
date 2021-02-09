# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:48:32 2020

@author: massonnetf
"""

#!/usr/bin/python
#
# Real-time predictions of sea ice extent using damped anomaly persistence
# forecasting. See below for the forecasting scheme.

# More info can also be found at 
# https://nsidc.org/sites/nsidc.org/files/webform/APPLICATE-benchmark.pdf
#
# Author F. Massonnet (June 2020) as part of the APPLICATE project and 
#        participation to Sea Ice Outlook


# Imports of modules
# ------------------

import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '' or os.name == "posix":
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import wget
import csv
import pandas as pd
import calendar
import datetime
import matplotlib.dates  as mdates
import scipy.stats

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
hemi = "south"
diag = "extent"

# Image resolution
dpi = 150

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
yearb = rawdata[0][0].year
yeare = rawdata[-1][0].year
nyear = yeare - yearb + 1
# Number of days per year
nday  = 365

# Create data array
data = np.full(nyear * nday, np.nan)

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
        
        data[row * nday + col] = r[1]
    
stop()

# Forecast function
# -----------------
        
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
       sys.exit("System cannot forecast from a 29th of February")
        
    # Get year, month and day of initial condition
    iniyear = inidate.year
    inimon  = inidate.month
    iniday  = inidate.day
    
    # Index of the initial day. To avoid problems with leap years, 
    # only the month
    # and day are considered (the year is arbitrarily set to a non-leap one)
    
    inidoy  = (datetime.date(2005, inidate.month, inidate.day) - 
               datetime.date(2005, 1, 1)).days

    # First and last years used to construct the background
    # Choosing yearebk < iniyear allows to put one self in operational context 
    yearbbk = yearb
    yearebk = iniyear - 1
    
    if yearebk >= iniyear:
        print("STOP: Interval for training model ends after initial time!")
        print("      Don't use of future data in retrospective predictions")
        sys.exit()
    
    # Construction of the background and its uncertainty
    background = np.full((nyear, nday), np.nan)
    std_bck    = np.full((nyear, nday), np.nan) 
    
    # We do this day by day
    for d in range(nday):
        # If yearb = 1979, yearbbk = yearb = 1979, iniyear = 2000, 
        # and yearebk = iniyear - 1 = 1999
        # then "series" is the data sub-sampled on 0:21, so the 21 first values
        # (Python ignores the last index). That makes effectiely the 1979-1999
        # period used to train the model.
        series = data[(yearbbk - yearb):(yearebk - yearb  + 1), d]
        years  = np.arange(yearb, yeare + 1)[(yearbbk - yearb):\
                          (yearebk- yearb  + 1)]   
        
        # Ignore nans (they occur quite frequently in the first decade)
        # by just popping them out
        years = years[  ~np.isnan(series)]
        series = series[~np.isnan(series)]
        
        # Create background for all years including that of initialization
        p, cov = np.polyfit(years, series, order, cov = True)
        background[:, d] = np.polyval(p, np.arange(yearb, yeare + 1))
     
        # Uncertainty estimates
        # Here a little bit of explanation is necessary.
        # Polynomial regression of order "q" takes the form
        #      T' * A = Y_hat 
        # where
        #          t_1^q  t_1^(q-1) ... t_1  1
        #      T=  t_2^q  t_2^(q-1) ... t_2  1
        #          ...
        #          t_M^q  t_M^(q-1) ... t_M  1
        # and
        #      A = [a_0 a_1 a_2 ... a_q]'
        #     is the vector of coefficients of the regression 
        #      (the ' denotes transpose)
        #     and Y_hat is the M-dimensional vector that we hope is 
        #     close to the data we fit
    
        # np.polyfit returns "p" and "C", such that
        # p = the A that minimizes the error
        # C   is the covariance matrix of p

        # Now, we know through statistics that if A ~ N(mu, Sigma)
        # then b + X * A ~ N(b + X * mu, X * Sigma * X')

        # Thus, Y_hat (the vector resulting from the fit at the sample points) follows
        #
        # Y_hat ~ N( T' * p, T' * C * T)
        
        XX = np.matrix([np.arange(yearb, yeare + 1) ** (order - i) \
                        for i in range(order + 1)]).transpose()
        covfit = XX * np.matrix(cov) * XX.transpose()
        std_bck[:, d] = np.array([np.sqrt(covfit[i, i]) \
               for i in range(len(XX))])
        
        del years, series
    
    # Creation of anomalies
    # Anomalies are the raw values minus the background
    data_ano = data - background
    
 
    # Computation of autocorrelation of anomalies at day d with inidoy
    r = np.full(nday, np.nan)
    for d in range(nday):
        x1tmp = data_ano[(yearbbk - yearb):(yearebk- yearb  + 1), d]
        x2tmp = data_ano[(yearbbk - yearb):(yearebk- yearb  + 1), inidoy]
        
        # Pop up nans
        x1 = x1tmp[~np.isnan(x1tmp * x2tmp)]
        x2 = x2tmp[~np.isnan(x1tmp * x2tmp)]
        
        # Correlate
        r[d] = np.corrcoef(x1, x2)[0, 1]
        del x1tmp, x2tmp, x1, x2
    
    # Construction of forecasts
    # -------------------------
    
    forecast       = list()
    std_forecast   = list()
    dates_forecast = list()
    verif_data     = list()
    
    for d in range(inidoy, nday):
        myforecast = r[d] * data_ano[iniyear - yearb, inidoy] + \
                        background[iniyear - yearb, d]
        forecast.append(myforecast)
        
        mystdforecast = np.abs(r[d] * std_bck[iniyear - yearb, inidoy] \
                               - std_bck[iniyear - yearb, d])
        
        std_forecast.append(mystdforecast)
        
        # Create datetime axis for forecast
        tmp = datetime.date(2005, 1, 1) + timedelta(days = d)
        dates_forecast.append(datetime.date(iniyear, tmp.month, tmp.day))
        
        # Output the verification data as well
        verif_data.append(data[iniyear - yearb, d])
        
        del tmp, myforecast, mystdforecast
    
    forecast     = np.array(forecast)
    std_forecast = np.array(std_forecast)
    verif_data   = np.array(verif_data)
    
    
    # Return values
    return dates_forecast, forecast, std_forecast, verif_data
    


# Forecasting
if mode == "oper":
    if freq == "monthly":
        startdates = [datetime.date(yeare, m, 1) for m in range(1, \
                     rawdata[-1][0].month + 1)]
    elif freq == "daily":
        # All days from January 1 of this year to now
        startdates = [datetime.date(yeare, 1, 1) + \
                  timedelta(days = d) for d in range((rawdata[-1][0] - \
                           datetime.date(yeare, 1, 1)).days + 1)]
        startdates = [startdates[-1]]
    else:
        sys.exit("Frequency not valid")
elif mode == "econ":
    startdates = [datetime.date(y, rawdata[-1][0].month, rawdata[-1][0].day) \
                  for y in np.arange(1994, yeare - 1)]
else: 
    if freq == "daily":
        startdates = [datetime.date(int(mode), 1, 1) + timedelta(days = d) \
              for d in range((datetime.date(int(mode), 8, 31) - \
                              datetime.date(int(mode), 1, 1)).days)]
    elif freq == "monthly":
        startdates = [datetime.date(int(mode), m, 1) for m in range(1, 9)]
    else:
        sys.exit("Frequency not valid")
# Remove 29th of February
startdates = [s for s in startdates if not (s.month == 2 and s.day == 29)]


# First year for bias correction
# To allow background estimation on a daily basis, years > 1988 only work
# Having 2 data points for linear trend, 3 for quadratic, etc., is further
# required to estimate the background.

year_ppb = 1988 + order + 1 

# Forecast probability of various events happening
# Will be a list with initial date and matching forecast probabilities
# for sea ice extent being in certain percentiles 
# It will also have the vector of the events being realized or not

pe = list()

# Next, we loop over all initial times and issue a lead-time dependent forecast
for inidate in startdates:
    
    # Year, month and day of initial condition
    iniyear = inidate.year
    inimon  = inidate.month
    iniday  = inidate.day
    
    # Create folder for figures if it does not exist
    if not os.path.isdir("./figs/" + str(iniyear)):
        os.makedirs("./figs/" + str(iniyear))
        
    # Issue the raw forecast
    print("Raw forecasting " + str(inidate))
    dates_forecast, forecast, std_forecast, verif_data = \
           damped_anomaly_persistence_forecast(inidate)

    # Compute target statistic
    forecast_mean = np.mean([f for d, f in zip(dates_forecast, forecast) \
                             if d.month == 9 and d.year == iniyear])
    forecast_year = iniyear 

    # Start forecast verification and recalibration for that day
    
    hindcast_mean       = list()
    hindcast_year       = list()
    verif_hindcast_mean = list()

    for year in np.arange(year_ppb, iniyear): # thus not including iniyear
        inidate_hindcast = datetime.date(year, inimon, iniday)
        
        # Reforecast
        print("  Hindcasting " + str(inidate_hindcast))
        dates_hindcast, hindcast, std_hindcast, verif_hindcast = \
              damped_anomaly_persistence_forecast(inidate_hindcast)
              
        # Extract target diagnostic
        hindcast_mean.append(np.mean([f for d, f in \
                                      zip(dates_hindcast, hindcast) \
                                      if d.month == 9]))
        hindcast_year.append(year)


        # In verif data as well
        verif_hindcast_mean.append(np.mean([v for d, v in \
                    zip(dates_hindcast, verif_hindcast) if d.month == 9]))
    
    
    # Prepare plots
    # -------------
    plt.close("all") 
    fig, ax= plt.subplots(1, 1, figsize = (5, 5), dpi = dpi)
    
    a = ax
    a.grid()
    
    # Plot observed
    a.plot([r[0] for r in rawdata], [r[1] for r in rawdata] , \
           color = [0.0, 0.0, 0.0], label = "Observed")
    a.plot(dates_forecast, forecast, "lightblue", label = "Best estimate")
    
    # Plot uncertainty
    myzs = [50, 95, 99] # quantiles
    zs = [scipy.stats.norm.ppf((100 - (100 - q) / 2 ) / 100) for q in myzs]
    
    for j, z in enumerate(zs):
        a.fill_between(dates_forecast, forecast - z * std_forecast, forecast \
                       + z * std_forecast, color = "blue", 
                     alpha = 0.8 / (j + 1), edgecolor = "none", zorder = 0, 
                     label = str(myzs[j]) + "% pred. interval", lw = 0)
    
    
    # Add all-time extremea until then
    alltimemin = np.nanmin(data[:iniyear-yearb, :])
    a.plot((-1e9, 1e9), (alltimemin, alltimemin), "r--", \
           label = "Minimum on record")
    
    a.legend(loc = "upper right", fontsize = 6)
    a.set_xlim(datetime.date(inidate.year, 1, 1), \
               datetime.date(inidate.year, 12, 31))
    a.set_ylim(0, 16.5)
    a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
    a.set_ylabel("$10^6$ km²")
    a.set_title("Damped anomaly persistence forecast of\ndaily " + \
                hemi_region[hemi] +  \
                "sea ice extent\ninitialized on " + str(inidate))
    ax.text(ax.get_xlim()[1], 0.0,  "\nProduced  " +
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
            " | @FMassonnet @applicate_eu", 
            rotation =90, ha = "left", va = "bottom", fontsize = 5)
    fig.autofmt_xdate(rotation = 45)
    fig.tight_layout()
    fig.savefig("./figs/" + str(iniyear) + "/forecast_order" + str(order) + \
                "_" + str(inidate) + ".png")
    
    
    # Verification and bias correction.
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 300)# dpi)
    
    ax = axes[0]
    ax.grid()
    if hemi == "north":
        ax.set_ylim(0.0, 8.0)
    elif hemi == "south":
        ax.set_ylim(15.0, 22.0)
        
    ax.plot(hindcast_year, hindcast_mean, color = [1, 0.5, 0.0], marker = "s", 
            label = "Hindcasts")
    ax.plot(hindcast_year, verif_hindcast_mean, color = [0.0, 0.0, 0.0], \
            marker = "s", 
            label = "Verification data")
    ax.set_ylabel("10$^6$ km$^2$")
    ax.set_title("September mean sea ice extent")
    ax.legend()
    
    
    
    ax = axes[1]
    ax.grid()
    ax.plot((-1e9, 1e9), (-1e9, 1e9), color = "k", lw = 1, label = "y = x")
    
    ax.plot((forecast_mean, forecast_mean), (-1e9, 1e9), "red", 
            label = forecast_year)
        
    ax.set_title("Verification and recalibration")
    if hemi == "north":
        ax.set_xlim(2.0, 8.0)
        ax.set_ylim(2.0, 8.0)
    elif hemi == "south":
        ax.set_xlim(17.0, 20.0)
        ax.set_ylim(17.0, 20.0)
    ax.set_xlabel("Forecast [10$^6$ km$^2$]")
    ax.set_ylabel("Verification [10$^6$ km$^2$]")
    ax.scatter(hindcast_mean, verif_hindcast_mean, 200, marker = "s", \
               color = "green", \
               alpha = 0.5, label = "Hindcasts")
    [ax.text(   hindcast_mean[j], verif_hindcast_mean[j],  \
             str(hindcast_year[j]), 
             color = "white", ha = "center", va = "center", fontsize = 5) \
        for j in range(len(hindcast_mean)  -1)]
        
    # Regress forecasts on verification to determine recalibrated forecast
    x = np.array(hindcast_mean)
    y = np.array(verif_hindcast_mean)
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
    xd = x - np.polyval(np.polyfit(np.arange(len(x)), x, 2), \
                        np.arange(len(x)))
    yd = y - np.polyval(np.polyfit(np.arange(len(y)), y, 2), \
                        np.arange(len(y)))
    
    # Prediction Interval function . Eq. 6.22 of
    # Daniel S. Wilks, Statistical methods in the atmospheric sciences, \
    # 2nd ed, International geophysics series, v. 91 
    # (Amsterdam ; Boston: Academic Press, 2006).
    def spred(xin):
        return np.sqrt(se2 * (1 + 1 / n + (xin - xbar) ** 2 / \
                              np.sum(xtil ** 2)))
    
    # Sample variable to show PDFs etc.
    xx = np.linspace(0.0, 30.0, 10000)
    
    fit = ahat * xx + bhat
    ax.plot(xx, fit, color = "green", label = "Regression")
    ax.fill_between(xx, fit - 1.96 * spred(xx), fit + 1.96 * spred(xx), 
                    color = "green", alpha = 0.2, lw = 0, 
                    label = "Prediction\n95% confidence interval")
    
    # Display correlations
    ax.text(5.5, 3.5, "$r$ = " + str(np.round(np.corrcoef(x, y)[0, 1], 2)) +
        "\n(detrended: " + str(np.round(np.corrcoef(xd, yd)[0, 1], 2)) + ")")
    
    # Print re-processed forecast
    if hemi == "north":
        xtext, ytext = 5.5, 2.5
    elif hemi == "south":
        xtext, ytext = 19.0, 17.8
        
    ax.text(xtext, ytext    , str(forecast_year) + " forecast:\n" + 
            str(np.round(ahat * forecast_mean + bhat, 2)) + 
            " [" + str(np.round(ahat * forecast_mean + bhat - 
                                1.96 * spred(forecast_mean), 2)) +  " - " +
                   str(np.round(ahat * forecast_mean + bhat + 
                                1.96 * spred(forecast_mean), 2)) + \
                                "]\n" + "$10^6$ km$^2$")
    
    ax.legend(loc = "upper left")
    
    fig.savefig("./figs/" + str(iniyear) + "/verif-postproc_order" + \
                str(order) + "_" + 
                    str(inidate) + ".png")
    
    
    
    # Presentation of the outlook
    # ---------------------------
    # The outlook is a way to communicate the probability that certain events
    # happen. Events are defined by a vector of percentiles
    # 
    # For a n-long vector of percentiles, the forecast probabilities of the 
    # following n + 1 events will be plotted
    # 1) below lowest percentile. If that percentile is 0, it means 
    #    breaking record low
    # 2) between first and second percentile
    # ...
    # n) between n-1 th and nth percentile
    # n+1) above highest percentile. If that percentile, it means 
    # breaking record high))
    
    percentiles = [0.0, 5.0, 10.0, 30.0, 70.0, 100.0]
    colors = [plt.cm.RdYlBu(int(255 / 100 * 0.5 * (percentiles[j - 1] + \
                        percentiles[j]))) for j in range(1, len(percentiles))]

    fig, ax = plt.subplots(1, 1, figsize = (5, 3), dpi = 300)
    ax.grid()
    if hemi == "north":
        ax.set_xlim(2.0, 10.0)
    elif hemi == "south":
        ax.set_xlim(15.0, 22.0)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel(str(iniyear) + " September mean sea ice extent [10$^6$ km$^2$]")
    
    # Best estimate
    mu = ahat * forecast_mean + bhat
    sig= spred(forecast_mean)

    ax.set_ylabel("Density [10$^6$ km$^2$]$^{-1}$")
    # Plot PDF
    ax.plot(xx, scipy.stats.norm.pdf(xx, mu, sig), color = "k", \
            label = "Forecast PDF")
    
    
    # Define thresholds from the observed data based on provided percentiles
    mythreshs   = np.percentile(verif_hindcast_mean, percentiles)
    
    
    for j in range(1, len(mythreshs)):
        xxtmp = np.linspace(mythreshs[j - 1], mythreshs[j], 10000)
        ax.fill_between(xxtmp, scipy.stats.norm.pdf(xxtmp, mu, sig), 
                    color = colors[j - 1], alpha = 0.8 , lw = 0.0, \
                     label = "Obs. " + str(int(percentiles[j - 1])) + 
                     "-" + str(int(percentiles[j])) + " %")
        
        # Plot bars for extremes if requested
        if percentiles[j - 1] == 0.0:
            ax.plot((mythreshs[j - 1], mythreshs[j - 1]), (-1e9, 1e9), \
                    color = "red", label = "Lowest record to date")
            ax.arrow(mythreshs[j - 1], - 0.07, -0.4, 0.0, color = "red", 
                     head_width = 0.05)
            probmin = scipy.stats.norm.cdf(mythreshs[j - 1], mu, sig)
            ax.text(mythreshs[j - 1], - 0.1, str(np.round(probmin * 100, 1)) + 
            " % ", va = "top", ha = "right", color = "red", alpha = 0.8)
        if percentiles[j] == 100.0:
            ax.plot((mythreshs[j], mythreshs[j]), (-1e9, 1e9), \
                    color = "blue", label = "Highest record to date")
            ax.arrow(mythreshs[j], - 0.07, 0.4, 0.0, color = "blue", 
                     head_width = 0.05)
            probmax = 1.0 - scipy.stats.norm.cdf(mythreshs[j], mu, sig)
            ax.text(mythreshs[j], - 0.1, " " + \
                    str(np.round(probmax * 100, 1)) + 
            " % ", va = "top", ha = "left", color = "blue", alpha = 0.8)
   
    # Plot verification data if available
    if not mode == "oper":
        truth = np.mean([r[1] for r in rawdata if r[0].year == iniyear \
                         and r[0].month == 9])
        ax.plot((truth, truth), (0, 1e9), color = "green", \
                linestyle = "--", label = "Observed")
        
    # Display trend extrapolation forecast
    trend_fc = np.polyval(np.polyfit(range(len(verif_hindcast_mean)), \
                verif_hindcast_mean, order), len(verif_hindcast_mean) + 1)
    ax.plot((trend_fc, trend_fc), (0, 1e9), color = "orange",
            label = str(type_trend[order]) + " extrapolation")\
    # Time-stamp the figure
    ax.text(ax.get_xlim()[1], -0.2,  "\nProduced " +
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
            " | @FMassonnet @applicate_eu", 
            rotation =90, ha = "left", va = "bottom", fontsize = 5)
    
    ax.legend(ncol = 1, fontsize = 6)
    ax.set_title("Forecast and associated event probabilities\n(initialized "\
                                                        + str(inidate) + ")")
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    fig.savefig("./figs/" + str(iniyear) + "/outlook_order" + str(order) + \
                "_" + str(inidate) + ".png")
    if inidate == startdates[-1]:
        if mode == "oper":
            fig.savefig("./webpages/operational_pdf.png")
        else:
            fig.savefig("./webpages/hindcast_" + mode + "_pdf.png")

    plt.close(fig)




    # Time series of probability of various events happening
    # ------------------------------------------------------
    cdfs = scipy.stats.norm.cdf(mythreshs, mu, sig)
    for j in range(len(percentiles)):
        myprobs = np.append(np.append(np.array(cdfs[0]), np.diff(cdfs)), \
                          np.array(1.0 - cdfs[-1]))
        
    if mode == "oper":
        pe.append([inidate, myprobs])
    else:
        truth = np.mean([r[1] for r in rawdata if r[0].year == iniyear \
                         and r[0].month == 9])
        realized = 1 * np.array([truth < mythreshs[0]] + \
                        [(truth >= mythreshs[j] and truth < mythreshs[j + 1]) \
                        for j in range(len(mythreshs) - 1)] + \
                         [truth >= mythreshs[-1]])
        pe.append([inidate, myprobs, realized])
        
        if np.sum(realized) != 1:
            sys.exit("Two exclusive events cannot realize")
    
    fig, ax = plt.subplots(1, 1, figsize = (6, 4), dpi = dpi)
    ax.grid()
  

    for j in range(len(percentiles)):
        if j == 0:
            if percentiles[j] == 0:
                mylabel = "... breaking record low to date"
                mycol = (1.0, 0.0, 0.0, 1.0)
            else:
                mylabel = "... < obs. " + \
                  str(int(percentiles[j])) +  " %"
                mycol = (1.0, 0.0, 0.0, 1.0)
            
            ax.plot([x[0] for x in pe], [100.0 * x[1][j] for x in pe], 
                color = mycol, label = mylabel)
            
            # Also plot the following range
            mylabel = "... in obs. " + str(int(percentiles[j])) + \
            "-" + str(int(percentiles[j + 1])) + " %"
            mycol = colors[j]
            ax.plot([x[0] for x in pe], [100.0 * x[1][j + 1] for x in pe], 
                color = mycol, label = mylabel)
            
            
        
        elif j == len(percentiles) - 1:
            if percentiles[j] == 100:
                mylabel = "... breaking record high to date"
                mycol = (0.0, 0.0, 1.0, 1.0)
            else:
                mylabel = "... > obs. " + \
                str(int(percentiles[j])) +  " %"
                mycol = (0.0, 0.0, 1.0, 1.0)
                
            ax.plot([x[0] for x in pe], [100.0 * x[1][j] for x in pe], 
                color = mycol, label = mylabel)
        else:
            mylabel = "... in obs. " + str(int(percentiles[j])) + \
            "-" + str(int(percentiles[j + 1])) + " %"
            mycol = colors[j]
            ax.plot([x[0] for x in pe], [100.0 * x[1][j + 1] for x in pe], 
                color = mycol, label = mylabel)
                     
    

    # Add target
    ax.fill_between((datetime.date(iniyear, 9, 1), \
                    datetime.date(iniyear, 10, 1)), (100, 100), \
                    color = "black", alpha = 0.2, lw = 0, \
                    label = "Target month")

    
    ax.set_title("Forecast probabilities of\n" + str(iniyear) + \
                 " September mean sea ice extent ...")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("%")
    ax.set_xlabel("Initialization date")

    ax.set_xlim(datetime.date(iniyear, 1, 1), datetime.date(iniyear, 12, 31))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
    
    ax.text(ax.get_xlim()[1], -0.2,  "\nProduced " +
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
            " | @FMassonnet @applicate_eu", 
            rotation =90, ha = "left", va = "bottom", fontsize = 5)
    
    fig.autofmt_xdate(rotation = 45)
    ax.legend(loc = "upper right", fontsize = 6)
    ax.set_facecolor([0.9, 0.9, 0.9])
    fig.tight_layout()
    fig.savefig("./figs/" + str(iniyear) + "/events_order" + str(order ) + \
                "_" + str(inidate) + ".png")
    if inidate == startdates[-1]:
        if mode == "oper":
            fig.savefig("./webpages/operational_outlook.png")# latest available
        else:
            fig.savefig("./webpages/hindcast_" + mode + "_outlook.png")

myevents = list()
if percentiles[0] == 0.0:
    myevents.append('"lower than min. on record"')
else:
    myevents.append("< " + str(percentiles[0]) + " % percentile")
    
for j in range(0, len(percentiles) - 1):
    myevents.append(str(percentiles[j]) + " % < $x$ $\leq$ " + str(percentiles[j + 1]) + " %")
    
if percentiles[-1] == 100.0:
    myevents.append("higher than max. on record")
else:
    myevents.append("$\geq$ " + str(percentiles[-1]) + " % percentile")
    
    
if mode == "econ":

    for j in range(len(myevents)):

        fig, ax = plt.subplots(1, 1, figsize = (6, 3), dpi = dpi)        
        
        # Plot forecast probabilities
        for k in range(len(pe)):
            if pe[k][2][j] == 1:
                color = "green"
            else:
                color = "red"
            ax.bar(pe[k][0].year, 100.0 * pe[k][1][j], color = color)
            
            ax.set_title("Forecast probabilities of event " + str(myevents[j]))
            ax.set_ylabel("%")
        
        # If I say that event E has probability p to occur then I'm ready to
        # give someone a little less than p (say 0.95 p). If I'm wrong, the person
        # keeps the money. If I'm right, she gives me 100
        
        ax.grid()
        ax.set_axisbelow(True)
        fig.savefig("./econ" + str(j) + ".png")
    
        moneymade = np.sum([- 0.95 * 100.0 * pe[k][1][j] + 100.0 * \
                            pe[k][2][j] for k in range(len(pe))])
                    
        print("Money made: " + str(np.round(moneymade)) + " €")
