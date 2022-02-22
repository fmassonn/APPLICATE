#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:22:33 2021

@author: massonnetf
"""

# To-Do
# - allow Nan in the data
# - allow unevenly spaced data in time
# - check if forecast gives expected behavior on one case
# - What happens when attempting to predict yesterday or today?
# - check agreement with Charctic's page
# - remove hard-coded leadTimes variable in forecast()

import calendar
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import os
import scipy.stats
import sys
import wget

np.random.seed(3)

type_trend = ["Climatology", "Linear", "Quadratic", "Cubic"]
order = 2

#global hemi; hemi = "north"
global dateRef; dateRef = datetime.date(1900, 1, 1)

def downloadData(hemi = "north", dataSet = "OSISAF-v2p1"):
    
   
    # Retrieving the data source file
    # -------------------------------

    hemi_region = {"south": "Antarctic",
                   "north": "Arctic"   ,
                  }
    hemshort = hemi[0] + hemi[-1]
    
    if dataSet == "NSIDC-G02135":
        rootdir = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/" + hemi + \
                   "/daily/data/"
        filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"
    elif dataSet == "OSISAF-v2p1":
        rootdir = "ftp://osisaf.met.no/prod_test/ice/index/v2p1/" + hemshort + "/"
        filein = "osisaf_" + hemshort + "_sie_daily.txt"
    else:
        sys.exit("Dataset unknown")
    
    if os.path.exists("./data/" + filein):
        os.remove("./data/" + filein)
        wget.download(rootdir + filein, out = "./data/")
    else:
        wget.download(rootdir + filein, out = "./data/")



def loadData(hemi = "north", dataSet = "OSISAF-v2p1"):
    # Reading the data
    # ----------------
    
    # Reading and storing the data. We are going to archive the daily extents
    # in a 1-D numpy array. Missing data are still recorded but as NaNs
    # Creating NaNs for missing data is useful because it makes the computation
    # of autocorrelation easier later on.
    # The 29th of February of leap years are excluded for 
    # ease of analysis
    hemshort = hemi[0] + hemi[-1]
    
    if dataSet == "NSIDC-G02135":
        filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"
        delimiter = ","
        nbIgnoreLines = 1
    elif dataSet == "OSISAF-v2p1":
        filein = "osisaf_" + hemshort + "_sie_daily.txt"
        delimiter = " "
        nbIgnoreLines = 7
        
    # Index for looping through rows in the input file
    j = 0
    
    

    rawData = list()
    
    
    if dataSet == "NSIDC-G02135":
      with open("./data/" + filein, 'r') as csvfile:
        obj = csv.reader(csvfile, delimiter = delimiter)
        for row in obj:
          if j <= nbIgnoreLines:
            pass
          else:
            thisDate = datetime.date(int(row[0]), int(row[1]), int(row[2]))
            thisValue = float(row[3])
            timeElapsed = (thisDate - dateRef).days
            
            if thisDate.month != 2 or thisDate.day != 29:
              rawData.append(
              [timeElapsed, thisDate, thisValue])
            
          j += 1
            
    elif dataSet == "OSISAF-v2p1":
      with open("./data/" + filein, 'r') as csvfile:
        obj = csv.reader(csvfile, delimiter = delimiter)
        for row in obj:
          if j <= nbIgnoreLines:
              pass
          else:
            if row[5]!= "MISSING":
              thisDate = datetime.date(int(row[1]), int(row[2]), int(row[3]))
              thisValue = float(row[4]) / 1e6              
              timeElapsed = (thisDate - dateRef).days

            # Only append if not 29 Feb
              if thisDate.month != 2 or thisDate.day != 29:
                rawData.append(
                [timeElapsed, thisDate, thisValue])
          j += 1   
  
            

        
        
    # Now that we have the raw dates, we can create 
    # a list of items for each date, even those for which there is no data.
        

    # Create list of all dates except 29th of Feb between the first and
    # last dates of rawData
    
    thisDate = rawData[0][1]
    allDates = list()
    
    while thisDate <= rawData[-1][1]:
        
        if thisDate.day != 29 or thisDate.month != 2:
            
            allDates.append(thisDate)

        thisDate += datetime.timedelta(days = 1)
    
    
    
    # Finally, go throught allDates and dump rawData if exists for that date
    counterRaw = 0
    outData = list()
    import time
    
    for d in allDates:
        
        timeElapsed = (d - dateRef).days
        if rawData[counterRaw][1] == d:
            #If there is a match, record it
            thisValue = rawData[counterRaw][2]
            counterRaw += 1
        else:

            thisValue = np.nan
        
        outData.append([timeElapsed, d, thisValue])

    return outData



# # Detect first and last years of the sample
# yearb = rawdata[0][0].year
# yeare = rawdata[-1][0].year
# nyear = yeare - yearb + 1
# # Number of days per year
# nday  = 365

# # Create data array
# data = np.full(nyear * nday, np.nan)

# # Fill it: loop over the raw data and store the extent value
# for r in rawdata:
    
#     # Ignore if 29th February
#     # Day of year
#     doy = int(r[0].strftime('%j'))
#     if not (calendar.isleap(r[0].year) and doy == 60):
#         # Match year
#         row = r[0].year - yearb
#         # Match day of year. Number of days counted from 1 to 365
#         # If leap year and after 29th Feb, subtract 1 to column to erase 29th Feb

#         if calendar.isleap(r[0].year) and doy > 60:
#             doy -= 1
#         # To match Pythonic conventions    
#         col = doy - 1
        
#         data[row * nday + col] = r[1]
        
def loadVerifData(hemi = "north"):
    pass    
        
def dampedAnomalyForecast(time, series, leadTimes):
    """
    

    Parameters
    ----------
    time : NUMPY ARRAY of integers
           Time coordinates
        
    series : NUMPY ARRAY
           Dataset used to train the model
           
    leadTimes : NUMPY ARRAY of integers
           Array of time coordinates at which the forecast is to be made

    Returns
    -------
    NUMPY ARRAY of length len(targetTime) with the forecast


    A damped anomaly persistence forecast is a simple persistence forecast of
    a time series of anomalies [1], starting from the latest data available, 
    and damped towards zero by having this latest anomaly multiplied by the 
    lead-time decreasing autocorrelation estimated from the training data. 
    
    [1] the use of a damped anomaly persistence forecast implies that
    anomalies must have been estimated beforehand.
    
    """
    
    # Checks
    if len(time) != len(series):
        sys.exit("Unequal lengths)")
    
   
    nowAnomaly = series[-1]
                 
    # estimation of auto-correlation of anomalies 
    # Add 0-th lag autocorrel!
    autocorrel = list()
    
    for lag in leadTimes:
        if lag == 0:
            correl = 1
        else:
            tmpSeries1 = series[:-lag]
            tmpSeries2 = series[lag: ]
            
            # Product is used to identify Nans
            tmpSeries1NoNan = tmpSeries1[~np.isnan(tmpSeries1 * tmpSeries2)]
            tmpSeries2NoNan = tmpSeries2[~np.isnan(tmpSeries1 * tmpSeries2)]
            
            correl = np.corrcoef(tmpSeries1NoNan, tmpSeries2NoNan)[0, 1]
        
        autocorrel.append(correl)
        

    autocorrel = np.array(autocorrel)
    

    forecast = autocorrel * nowAnomaly
    
    # Contribution of anomaly term to forecast std       
    
    return forecast, autocorrel

def createSyntheticData(time):

    nt = len(time)
    
    series = 3.0 * np.sin(2 * np.pi * np.arange(nt) / 365) + \
            0.1 * np.convolve(np.random.randn(nt), np.ones(100), "same")
    
    return series


def computeAnomalies(time, series, order, \
                               periodicity = None):
    """
    
    Decomposition of a time series in terms of a background contribution
    (forcing) and anomalies
    
    Parameters
    ----------
    time : NUMPY ARRAY of integers
           Time coordinates, measured as units of times elapsed since a
           reference date (e.g;, number of days since 1900-01-01)
           Need not be evenly spaced.
    series : NP.ARRAY
        array of values to be anomalized
    order = order of detrending (linear, quadratic)
    periodicity: INT
        possible periodicity in the input data (e.g., 365 for daily data of
        a geophysical variable with annual cycle). If not None, the
        anomalies are computed relative to the
        
    backgroundAtTime: NP.ARRAY
        array of times at which the background needs to be estimated
        (extrapolated, interpolated)

    Returns
    -------
    None.

    """
    
    background = np.full(series.shape, np.nan)
    
    if periodicity is not None:
        
        jt = 0
        while jt < periodicity:
    
            t0 = time[jt]
            # Find matching times (inclluding of course the first one)
            indices = [jtt for jtt, t in enumerate(time) \
                       if (t - t0) % periodicity == 0]

    
            tmpSeries = series[indices]

            tmpTime   = time[indices]
            
            # Estimate background for input time
            # First we need to remove the NaNs

            tmpTimeNoNan = tmpTime[ ~ np.isnan(tmpSeries)]
            tmpSeriesNoNan = tmpSeries[ ~ np.isnan(tmpSeries)]
  
            p = np.polyfit(tmpTimeNoNan, tmpSeriesNoNan, order)
            tmpBackground = np.polyval(p, \
                                       tmpTime)
            
            background[indices] = tmpBackground
            
            jt += 1
                
    else:
        print("NOT CODED")
        stop()
    
    outAnomalies = series - background
    

    
    return outAnomalies

def extrapolateBackground(time, series, order, extrapTime, periodicity = None):
    """
    extrapTime: NUMPY ARRAY of integers
                Time coordinates at which to compute the extrapolation
                Must include initial time
    """
    

    outBackground = np.full(len(extrapTime), np.nan)
    stdBackground = np.full(len(extrapTime), np.nan)

    if periodicity is not None:
        for k, e in enumerate(extrapTime):
            
            # Find the time instances and relevant indices that correspond
            # to the extrapolation time
            tmpList = [[t, series[j]] for j, t in enumerate(time) if \
                       (e - t) % periodicity == 0]
            
            tmpTime   = np.array(tmpList)[:, 0]
            tmpSeries = np.array(tmpList)[:, 1]

            tmpTimeNoNan = tmpTime[~ np.isnan(tmpSeries)]
            tmpSeriesNoNan = tmpSeries[~ np.isnan(tmpSeries)]

            p, cov = np.polyfit(tmpTimeNoNan, tmpSeriesNoNan, \
                                order, cov = True)
                
            outBackground[k] = np.polyval(p, e)
            
            XX = np.matrix([e ** (order - i) \
                        for i in range(order + 1)])
            
            covfit = XX * np.matrix(cov) * XX.transpose()
     
    
            stdBackground[k] = np.array([np.sqrt(covfit[i, i]) \
               for i in range(len(XX))])
                
            #print(k)
            #print(stdBackground[k])
         
            #print(series)
            #print("--")
            #if k >=45:
            #    stop()
    
    else:
        sys.exit("Not coded yet")


    return outBackground, stdBackground
            


# FORECAST
# --------

def forecast(hemi = "north", dateInit = None, getData = True, \
             dataSet = "OSISAF-v2p1", verif = False):
    
    """
    dateInit defines the initialization date. If None, takes the
    latest date of the latest available input data. 
    In all case, no data after dateInit
    is used to make the forecast
    
    dateInit has to be a valid date among the data used to train the
    prediction system. The following cases won't work:
        - dateInit greater than the latest date of available data
        - dateInit less than the earliest date of available data
        - dateInit corresponding to a date for which there was no data
    
    """

    # Set dateInit if not given
    # First, load the data

    if dateInit is None:
        if getData:
            downloadData(hemi = hemi, dataSet = dataSet)
            
        rawData = loadData(hemi = hemi, dataSet = dataSet)
        dateInit = rawData[-1][1] # datetime.date(2020, 6, 1)
    else:
        if getData:
            downloadData(hemi = hemi, dataSet = dataSet)
            
        rawData = loadData(hemi = hemi, dataSet = dataSet)
        

        
    # At this stage dateInit has a value. We can thus create variables
    # with initialization year and month and day
    
    # We stop if the date Init is not in the range of dates
    if not dateInit in [r[1] for r in rawData]:
        sys.exit("(forecast) STOP, dateInit not in range")

    yearInit = dateInit.year
    



    # Define target period over which an outlook will be computed
    # -----------------------------------------------------------
    d = 0
    keepGoing = True
    while keepGoing:
        leadDate = dateInit + datetime.timedelta(days = d)
        
        if hemi == "north":
            if leadDate.month == 9 and leadDate.day == 30:
                targetDateMin = datetime.date(leadDate.year, 9, 1)
                targetDateMax = leadDate
                keepGoing = False
    
        elif hemi == "south":
            if leadDate.month == 2 and leadDate.day == 28:
            #if leadDate.month == 9 and leadDate.day == 30:
                #targetDateMin = datetime.date(leadDate.year, 9, 1)
                targetDateMin = datetime.date(leadDate.year, 2, 1)
                targetDateMax = leadDate

                keepGoing = False
        
      
        else:
            sys.exit("Hemisphere unknown")
        d += 1

    

    # Remove all data past dateInit to make sure we are not using future
    # data to train the model. Just before doing this, we record the 
    # verification outlook to output it if asked

    if verif:
        verifOutlook = np.nanmean(np.array([r[2] for r in rawData if \
            r[1] >= targetDateMin and r[1] <= targetDateMax]))
    else:
        verifOutlook = np.nan
        
        
    # Subset the data to make sure we don't use future data
    rawData = [r for r in rawData if r[1] <= dateInit]
    
    time   = np.array([r[0] for r in rawData])
    dates  =          [r[1] for r in rawData]
    series = np.array([r[2] for r in rawData])
    

    # Computation of anomalies of the historical data
    anomalies = computeAnomalies(time, series, order, periodicity = 365)
    
    # Computation of the background
    background = series - anomalies
    
    
    
    # !! Warning always include the initial time (0) in the set of leadTimes
    # This is needed to compute the forecast variance, as it depends on the
    # background estimate at initial time.
    leadTimes = np.arange(0, 365)
    
    datesLeadTimes = [dateInit + datetime.timedelta(days = \
                                    float(l)) for l in leadTimes]
    
    backgroundForecast, backgroundStd = extrapolateBackground(time, \
                                                     series, order, \
                                    leadTimes + time[-1], periodicity = 365)
    
        
    anomalyForecast, autocorrel =  dampedAnomalyForecast(time, anomalies, \
                                    leadTimes)
    
    forecast = anomalyForecast + backgroundForecast
        
    outlook = np.mean([z[0] for z in zip(forecast, datesLeadTimes) \
                       if z[1] >= targetDateMin and z[1] <= targetDateMax])
            

    # Plots
    # -----
    
    fig, ax = plt.subplots(3, 1, dpi = 300, figsize = (8, 12))
    
        
    # Show background forecast
    ax[1].plot(dates, background, "brown", label = "Background")
    ax[1].plot(datesLeadTimes, backgroundForecast, lw = 0.5, color = "brown",\
                  label = "Background forecast")
    ax[1].fill_between(datesLeadTimes, backgroundForecast -  \
                               1.96 * backgroundStd, \
                                       backgroundForecast + \
                               1.96 * backgroundStd, \
                                      color = "brown", alpha = 0.3, lw = 0, \
                                          label = "95% conf. interval")    
    
        
    
    
    # Show initial time
    ax[2].plot(dates, anomalies, color = "darkgreen", label = "Anomalies")
    
    ax[2].plot(datesLeadTimes, anomalyForecast, lw = 0.5 , \
                 color = "darkgreen", \
                 label = "Damped anomaly forecast")
    
    # Compute forecast std
    forecastStd = np.sqrt((autocorrel * backgroundStd[0] - backgroundStd) ** 2)
    
    # Show raw forecast
    ax[0].plot(dates, series, color = "k", label = "Raw data")
    ax[0].plot(datesLeadTimes, forecast, lw = 0.5, color = "k", \
                 label = "Damped anomaly persistence forecast")
    ax[0].fill_between(datesLeadTimes,       forecast -  1.96 * forecastStd, \
                                             forecast +  1.96 * forecastStd, \
                                             color = "k", alpha = 0.4, \
                                             label = "95% conf. interval", \
                                             lw = 0)
    ax[0].plot(dates + datesLeadTimes, np.nanmin(series) * \
               np.ones(len(dates + datesLeadTimes)), color = "r", \
                   linestyle = "--", \
               label = "All time minimum so far")
    
    
    for j, a in enumerate(ax):
        if j != 2:
            a.set_ylim(0, 20)
            #a.set_ylim(14, 15)
        else:
            a.set_ylim(-3.0, 3.0)
            #a.set_ylim(0.0, 1.0)
        a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        a.grid()
        a.set_xlim(dateInit + datetime.timedelta(days = - 365), \
                   datesLeadTimes[-1])
        a.set_axisbelow(True)
        a.set_ylabel("Million km$^2$")
        
        myTicks = list(set([datetime.date(d.year, d.month, 1) for d in datesLeadTimes]))
        a.set_xticks(myTicks)
        
        for label in a.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(45)
            
        a.fill_between((targetDateMin, targetDateMax), (-1e9, -1e9), (1e9, 1e9), \
                       alpha = 0.2, label = "Target period")
        #print(targetDateMin, targetDateMax)
    
        a.legend(loc = "upper left")
    

    
    fig.suptitle("Sea ice extent forecast")
    fig.tight_layout()
    directory = "./figs/" + str(yearInit) + "/" + str(dateInit) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + "/rawOutlook_order" + \
            str(order) + "_" + \
                dataIn + "_" +\
                str(dateInit) + "_" + hemi + ".png")
    plt.close(fig)

    return outlook, verifOutlook


# Run forecast
# ------------
#
# Product
#dataIn = "OSISAF-v2p1"
dataIn = "NSIDC-G02135"
# Define initialization date (set to None for latest in sample)
#dateInit = None
dateInit = datetime.date(2022, 2, 19)

# Set hemisphere
hemi = "south"#hemi = "north"

# Run the raw forecast
thisOutlook, _ = forecast(hemi = hemi, dateInit = dateInit, dataSet = dataIn)


print("Raw outlook for initial date " + str(dateInit) + ": " + \
      str(np.round(thisOutlook, 2)) + " million km2")

# Post-processing
# ---------------

# Load verification outlooks
# --------------------------
# These are not necesarily the same data as the data used to train the model
# In this case they are, but this might change (e.g., NSIDC is the target but
# OSISAF is the training data)
#verifData = loadData(hemi = "north")

# Run hindcasts
# -------------

# Get initialization day and month to run hindcasts
# If dateInit was set to None, fetch the information from the loaded Data
if dateInit is None:
    dateInit = loadData(hemi, dataSet = dataIn)[-1][1]

print(dateInit)


dayInit   = dateInit.day
monthInit = dateInit.month
yearInit  = dateInit.year


allOutlooks = list()
allVerifOutlooks = list()
allYears = list()

for year in np.arange(1989, yearInit):
    outlook, verifOutlook = forecast(hemi = hemi, dateInit = \
              datetime.date(year, monthInit, dayInit), \
              getData = False, dataSet = dataIn, verif = True)
    print(str(year) + ": " + str(np.round(outlook, 2)) + " million km2 " + \
          "(verification: " + str(np.round(verifOutlook, 2)) + ")")
    
    allOutlooks.append(outlook)
    allVerifOutlooks.append(verifOutlook)
    allYears.append(year)
    


# Verification and bias correction.
# ---------------------------------
fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 300)# dpi)
    
ax = axes[0]
ax.set_title("Raw hindcasts")
ax.grid()
        
ax.plot(allYears, allOutlooks, color = [1, 0.5, 0.0], marker = "s", 
        label = "Hindcasts")
ax.plot(allYears, allVerifOutlooks, color = [0.0, 0.0, 0.0], \
        marker = "s", 
        label = "Verification data")
ax.set_ylabel("10$^6$ km$^2$")
ax.set_ylim(bottom = 0.0)
ax.legend()
    
    
    
ax = axes[1]
ax.grid()


    

     
ax.set_xlim(np.min([thisOutlook] + allOutlooks) * 0.9, np.max([thisOutlook] + allOutlooks) * 1.1)
ax.set_ylim(np.min(allVerifOutlooks) * 0.9, np.max(allVerifOutlooks) * 1.1)
ax.set_xlabel("Forecast [10$^6$ km$^2$]")
ax.set_ylabel("Verification [10$^6$ km$^2$]")
ax.scatter(allOutlooks, allVerifOutlooks, 200, marker = "s", \
           color = "green", \
           alpha = 0.5, label = "Hindcasts")
[ax.text(   allOutlooks[j], allVerifOutlooks[j],  \
         str(allYears[j]), 
         color = "white", ha = "center", va = "center", fontsize = 5) \
    for j in range(len(allOutlooks))]
    

# Regress forecasts on verification to determine recalibrated forecast
x = np.array(allOutlooks)
y = np.array(allVerifOutlooks)
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



# Print re-processed forecast


xtext = np.max(allOutlooks) * 0.7
ytext = np.min(allOutlooks) * 1.1
# Display correlations
ax.set_title("Verification and recalibration\n" + 
            "$r$ = " + str(np.round(np.corrcoef(x, y)[0, 1], 2)) +
         " (detrended: " + str(np.round(np.corrcoef(xd, yd)[0, 1], 2)) + ")" + \
             "\nThis forecast: " + \
        str(np.round(ahat * thisOutlook + bhat, 2)) + \
        " [" + str(np.round(ahat * thisOutlook + bhat - \
                            1.96 * spred(thisOutlook), 2)) +  " - " + \
                str(np.round(ahat * thisOutlook + bhat + \
                            1.96 * spred(thisOutlook), 2)) + \
                            "]" + " x $ 10^6$ km$^2$")



lims = ax.get_xlim()

ax.plot((-1e9, 1e9), (-1e9, 1e9), color = "k", lw = 1, label = "y = x")
ax.plot((thisOutlook, thisOutlook), (-1e9, 1e9), "red", 
        label = "Raw outlook")

ax.set_xlim(lims[0], lims[1])

ax.legend(loc = "upper left")

directory = "./figs/" + str(yearInit) + "/" + str(dateInit) + "/"
if not os.path.exists(directory):
        os.makedirs(directory)

fig.savefig(directory + "/verif-postproc_order" + \
            str(order) + "_" + \
                dataIn + "_" +\
                str(dateInit) + "_" + hemi + ".png")



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

percentiles = [0.0, 10.0, 33.33, 66.67, 100.0]
colors = [plt.cm.RdYlBu(int(255 / 100 * 0.5 * (percentiles[j - 1] + \
                     percentiles[j]))) for j in range(1, len(percentiles))]

fig, ax = plt.subplots(1, 1, figsize = (5, 3), dpi = 300)
ax.grid()
# if hemi == "north":
#     ax.set_xlim(2.0, 10.0)
# elif hemi == "south":
#     ax.set_xlim(15.0, 22.0)

ax.set_xlabel("Target mean sea ice extent [10$^6$ km$^2$]")

# Best estimate
mu = ahat * thisOutlook + bhat
sig= spred(thisOutlook)

ax.set_ylabel("Density [10$^6$ km$^2$]$^{-1}$")


# Define thresholds from the observed data based on provided percentiles
mythreshs   = np.percentile(allVerifOutlooks, percentiles)


for j in range(1, len(mythreshs)):
    xxtmp = np.linspace(mythreshs[j - 1], mythreshs[j], 10000)

    
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
   
    prob= scipy.stats.norm.cdf(mythreshs[j], mu, sig) - \
                scipy.stats.norm.cdf(mythreshs[j - 1], mu, sig)
                
    ax.fill_between(xxtmp, scipy.stats.norm.pdf(xxtmp, mu, sig), 
                color = colors[j - 1], alpha = 0.8 , lw = 0.0, \
                  label = "in obs. " + str(int(percentiles[j - 1])) + 
                  "-" + str(int(percentiles[j])) + " %" + " [" + str(np.round(prob * 100, 1))+ " %]")
        
        
# Display trend extrapolation forecast
trend_fc = np.polyval(np.polyfit(range(len(allVerifOutlooks)), \
            allVerifOutlooks, order), len(allVerifOutlooks) + 1)
ax.plot((trend_fc, trend_fc), (0, 1e9), color = "orange",
        label = str(type_trend[order]) + " extrapolation")\

ax.set_xlim(0.0, np.max(allVerifOutlooks) * 1.2)
# Plot PDF
pdfFit = scipy.stats.norm.pdf(xx, mu, sig)
ax.plot(xx, pdfFit, color = "k", \
        label = "Forecast PDF")
maxYvalue = np.max(scipy.stats.norm.pdf(xx, mu, sig))
    
    
    
# Time-stamp the figure
ax.text(ax.get_xlim()[1], -0.2,  "\nProduced " +
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
        " | @FMassonnet @applicate_eu" + "\nInput data: " + dataIn, 
        rotation =90, ha = "left", va = "bottom", fontsize = 5)

ax.legend(ncol = 1, fontsize = 6)
ax.set_title("Forecast and associated event probabilities\n(initialized "\
                                                    + str(dateInit) + ")")
ax.set_axisbelow(True)
ax.set_ylim(-0.3, np.max(pdfFit) * 1.1 )

directory = "./figs/" + str(yearInit) + "/" + str(dateInit) + "/"
if not os.path.exists(directory):
    os.makedirs(directory)
        
fig.tight_layout()

fig.savefig(directory + "/outlook_order" + \
            str(order) + "_" + \
                dataIn + "_" +\
                str(dateInit) + "_" + hemi + ".png")
plt.close(fig)




# Time series of probability of various events happening
# ------------------------------------------------------
#cdfs = scipy.stats.norm.cdf(mythreshs, mu, sig)
#for j in range(len(percentiles)):
#    myprobs = np.append(np.append(np.array(cdfs[0]), np.diff(cdfs)), \
#                      np.array(1.0 - cdfs[-1]))
  
        
#pe = list()

# if mode == "oper":
#     pe.append([inidate, myprobs])
# else:
#     truth = np.mean([r[1] for r in rawdata if r[0].year == iniyear \
#                      and r[0].month == 9])
#     realized = 1 * np.array([truth < mythreshs[0]] + \
#                     [(truth >= mythreshs[j] and truth < mythreshs[j + 1]) \
#                     for j in range(len(mythreshs) - 1)] + \
#                      [truth >= mythreshs[-1]])
#     pe.append([inidate, myprobs, realized])
    
#     if np.sum(realized) != 1:
#         sys.exit("Two exclusive events cannot realize")

# fig, ax = plt.subplots(1, 1, figsize = (6, 4), dpi = dpi)
# ax.grid()
  

# for j in range(len(percentiles)):
#     if j == 0:
#         if percentiles[j] == 0:
#             mylabel = "... breaking record low to date"
#             mycol = (1.0, 0.0, 0.0, 1.0)
#         else:
#             mylabel = "... < obs. " + \
#               str(int(percentiles[j])) +  " %"
#             mycol = (1.0, 0.0, 0.0, 1.0)
        
#         ax.plot([x[0] for x in pe], [100.0 * x[1][j] for x in pe], 
#             color = mycol, label = mylabel)
        
#         # Also plot the following range
#         mylabel = "... in obs. " + str(int(percentiles[j])) + \
#         "-" + str(int(percentiles[j + 1])) + " %"
#         mycol = colors[j]
#         ax.plot([x[0] for x in pe], [100.0 * x[1][j + 1] for x in pe], 
#             color = mycol, label = mylabel)
        
        
    
#     elif j == len(percentiles) - 1:
#         if percentiles[j] == 100:
#             mylabel = "... breaking record high to date"
#             mycol = (0.0, 0.0, 1.0, 1.0)
#         else:
#             mylabel = "... > obs. " + \
#             str(int(percentiles[j])) +  " %"
#             mycol = (0.0, 0.0, 1.0, 1.0)
            
#         ax.plot([x[0] for x in pe], [100.0 * x[1][j] for x in pe], 
#             color = mycol, label = mylabel)
#     else:
#         mylabel = "... in obs. " + str(int(percentiles[j])) + \
#         "-" + str(int(percentiles[j + 1])) + " %"
#         mycol = colors[j]
#         ax.plot([x[0] for x in pe], [100.0 * x[1][j + 1] for x in pe], 
#             color = mycol, label = mylabel)
                 


# # Add target
# ax.fill_between((datetime.date(iniyear, 9, 1), \
#                 datetime.date(iniyear, 10, 1)), (100, 100), \
#                 color = "black", alpha = 0.2, lw = 0, \
#                 label = "Target month")


# ax.set_title("Forecast probabilities of\n" + str(iniyear) + \
#              " September mean sea ice extent ...")
# ax.set_ylim(0.0, 100.0)
# ax.set_ylabel("%")
# ax.set_xlabel("Initialization date")

# ax.set_xlim(datetime.date(iniyear, 1, 1), datetime.date(iniyear, 12, 31))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))

# ax.text(ax.get_xlim()[1], -0.2,  "\nProduced " +
#         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
#         " | @FMassonnet @applicate_eu", 
#         rotation =90, ha = "left", va = "bottom", fontsize = 5)

# fig.autofmt_xdate(rotation = 45)
# ax.legend(loc = "upper right", fontsize = 6)
# ax.set_facecolor([0.9, 0.9, 0.9])
# fig.tight_layout()
# fig.savefig("./figs/" + str(iniyear) + "/events_order" + str(order ) + \
#             "_" + str(inidate) + ".png")
# if inidate == startdates[-1]:
#     if mode == "oper":
#         fig.savefig("./webpages/operational_outlook.png")# latest available
#     else:
#         fig.savefig("./webpages/hindcast_" + mode + "_outlook.png")

# myevents = list()
# if percentiles[0] == 0.0:
# myevents.append('"lower than min. on record"')
# else:
# myevents.append("< " + str(percentiles[0]) + " % percentile")

# for j in range(0, len(percentiles) - 1):
# myevents.append(str(percentiles[j]) + " % < $x$ $\leq$ " + str(percentiles[j + 1]) + " %")

# if percentiles[-1] == 100.0:
# myevents.append("higher than max. on record")
# else:
# myevents.append("$\geq$ " + str(percentiles[-1]) + " % percentile")


# if mode == "econ":

# for j in range(len(myevents)):

#     fig, ax = plt.subplots(1, 1, figsize = (6, 3), dpi = dpi)        
    
#     # Plot forecast probabilities
#     for k in range(len(pe)):
#         if pe[k][2][j] == 1:
#             color = "green"
#         else:
#             color = "red"
#         ax.bar(pe[k][0].year, 100.0 * pe[k][1][j], color = color)
        
#         ax.set_title("Forecast probabilities of event " + str(myevents[j]))
#         ax.set_ylabel("%")
    
#     # If I say that event E has probability p to occur then I'm ready to
#     # give someone a little less than p (say 0.95 p). If I'm wrong, the person
#     # keeps the money. If I'm right, she gives me 100
    
#     ax.grid()
#     ax.set_axisbelow(True)
#     fig.savefig("./econ" + str(j) + ".png")

#     moneymade = np.sum([- 0.95 * 100.0 * pe[k][1][j] + 100.0 * \
#                         pe[k][2][j] for k in range(len(pe))])
                
#     print("Money made: " + str(np.round(moneymade)) + " €")











