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

import calendar
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import os
import sys
import wget

np.random.seed(3)

order = 0

global hemi; hemi = "north"
global dateRef; dateRef = datetime.date(1900, 1, 1)

def downloadData(hemi = "north"):
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


def loadData(hemi = "north"):
    # Reading the data
    # ----------------
    
    # Reading and storing the data. We are going to archive the daily extents
    # in a 1-D numpy array. Missing data are still recorded but as NaNs
    # Creating NaNs for missing data is useful because it makes the computation
    # of autocorrelation easier later on.
    # The 29th of February of leap years are excluded for 
    # ease of analysis
    
    filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"    
        
    # Index for looping through rows in the input file
    j = 0
    
    outData = list()
    rawData = list()
    with open("./data/" + filein, 'r') as csvfile:
      obj = csv.reader(csvfile, delimiter = ",")
      nd = obj.line_num - 2 # nb data
      for row in obj:
        if j <= 1:
          pass
          #print("Ignore, header")
        else:
            thisDate = datetime.date(int(row[0]), int(row[1]), int(row[2]))
            timeElapsed = (thisDate - dateRef).days
            thisValue = float(row[3])
            
            # Only append if not 29 Feb
            if thisDate.month != 2 or thisDate.day != 29:
                rawData.append(
                    [timeElapsed, thisDate, thisValue]
                    )
            
        j = j + 1
        
        
    # Now that we have the raw dates, we can create 
    # a list of items for each date, even those for which there is no data.
        
    outData = list()
    

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
                
    else:
        sys.exit("Not coded yet")

    return outBackground, stdBackground
            


# FORECAST
# --------

def forecast(hemi = "north", dateInit = None, getData = False):
    
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

    

    if dateInit is None:
        if getData:
            downloadData(hemi = hemi)
            
        rawData = loadData(hemi = hemi)
        dateInit = rawData[-1][1] # datetime.date(2020, 6, 1)
        timeInit = rawData[-1][0]
    else:
        if getData:
            downloadData(hemi = hemi)
            
        rawData = loadData(hemi = hemi)
        
        # Remove all data past dateInit
        
        rawData = [r for r in rawData if r[1] <= dateInit]
        timeInit = rawData[-1][0]
    
    
    
    if not dateInit in [r[1] for r in rawData]:
        sys.exit("(forecast) STOP, dateInit not in range")
        

    
    leadTimes = np.arange(0, 365, 2)

    time   = np.array([r[0] for r in rawData if r[1] <= dateInit])
    dates  =          [r[1] for r in rawData if r[1] <= dateInit]
    series = np.array([r[2] for r in rawData if r[1] <= dateInit])


    # Define target period over which an average will be computed
    # -----------------------------------------------------------
    d = 0
    keepGoing = True
    while keepGoing:
        leadDate = dateInit + datetime.timedelta(days = d)
        
        if hemi == "north":
            if leadDate.month == 9 and leadDate.day == 1:
                targetDateMin = leadDate
                targetDateMax = datetime.date(leadDate.year, 9, 30)
                keepGoing = False
    
        elif hemi == "south":
            if leadDate.month == 2 and leadDate.day == 1:
                targetDateMin = leadDate
                targetDateMax = datetime.date(leadDate.year, 2, 28)
                keepGoing = False
        
      
        else:
            sys.exit("Hemisphere unknown")
        d += 1


    # Computation of anomalies of the historical data
    anomalies = computeAnomalies(time, series, order, periodicity = 365)
    
    # Computation of the background
    background = series - anomalies
    
    
    
    # !! Warning always include the initial time (0) in the set of leadTimes
    # This is needed to compute the forecast variance, as it depends on the
    # background estimate at initial time.
    
    
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
    
    fig, ax = plt.subplots(3, 1, dpi = 300, figsize = (4, 9))
    
        
    # Show background forecast
    ax[1].plot(dates, background, "brown", label = "Background")
    ax[1].plot(datesLeadTimes, backgroundForecast, lw = 0.5, color = "brown",\
                  label = "Background forecast")
    ax[1].fill_between(datesLeadTimes, backgroundForecast -  backgroundStd, \
                                             backgroundForecast + \
                                                 backgroundStd, \
                                      color = "brown", alpha = 0.3, lw = 0)    
    
        
    
    
    # Show initial time
    ax[2].plot(dates, anomalies, color = "darkgreen", label = "Anomalies")
    
    ax[2].plot(datesLeadTimes, anomalyForecast, lw = 0.5 , \
                 color = "darkgreen", \
                 label = "Damped anomaly forecast")
    
    # Compute forecast std
    forecastStd = np.sqrt((autocorrel * backgroundStd[0] - backgroundStd) ** 2)
    
    ax[0].plot(dates, series, color = "k", label = "Raw data")
    ax[0].plot(datesLeadTimes, forecast, lw = 0.5, color = "k", \
                 label = "Damped anomaly persistence forecast")
    ax[0].fill_between(datesLeadTimes,       forecast -  1.96 * forecastStd, \
                                             forecast +  1.96 * forecastStd, \
                                             color = "k", alpha = 0.4, lw = 0)
    
        
    
    
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
        print(targetDateMin, targetDateMax)
    
        a.legend()
    
    fig.suptitle("Sea ice extent forecast")
    fig.tight_layout()
    fig.savefig("./fig.png")


    return outlook


# Load verification data
# ----------------------
# These are not necesarily the same data as the data used to train the model
verifData = loadVerifData(hemi = "north")

# Run hindcasts
# -------------
#for year in np.arange(1989, 2020):
#    outlook = forecast(dateInit = datetime.date(year, 2, 8))
#    print(str(year) + ": " + str(outlook))














