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

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

order = 2

def dampedAnomalyForecast(time, series, leadTimes,\
                               tMin = None, tMax = None):
    """
    

    Parameters
    ----------
    time : NUMPY ARRAY of integers
           Time coordinates
        
    series : NUMPY ARRAY
           Dataset used to train the model
    tMin   : int
           Time coordinate defining the beginning of the training period
           (included)
    tMax   : int
           Time coordinate defining the end of the training period
           (excluded)
           
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

    # Optional arguments
    if tMin is None:
        tMin = np.min(time)
    if tMax is None:
        tMax = np.max(time)
    
    # Restrict series to training period
    seriesTraining = series[(time >= tMin) * (time < tMax)]
    timeTraining   = time[(time >= tMin) * (time < tMax)]
   
    nowAnomaly = seriesTraining[-1]
                 
    # estimation of auto-correlation of anomalies 
    autocorrel = np.array([np.corrcoef(seriesTraining[:- lag], seriesTraining[lag:])[0, 1] \
                           for lag in np.arange(1, len(seriesTraining) - 2)])
                    
    
    targetTimes = timeTraining[-1] + leadTimes
        
    forecast = autocorrel[leadTimes] * nowAnomaly
    
    # Contribution of anomaly term to forecast std       
    
    return forecast, autocorrel[leadTimes]

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
           Time coordinates
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
        for j in np.arange(periodicity):
            tmpseries = series[j::periodicity]
            tmptime   = time[j::periodicity]
            
            # Estimate background for input time
            p = np.polyfit(tmptime, tmpseries, order)
            tmpBackground = np.polyval(p, \
                                       tmptime)
            
            background[j::periodicity] = tmpBackground
                
    else:
        tmpseries = series
        tmptime   = time
        p = np.polyfit(tmptime, tmpseries, order)
        tmpBackground = np.polyval(p, \
                                  tmptime)
        background = tmpBackground
    
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
            tmplist = [[t, series[j]] for j, t in enumerate(time) if \
                       (e - t)%periodicity == 0]
            
            tmptime   = np.array(tmplist)[:, 0]
            tmpseries = np.array(tmplist)[:, 1]

            p, cov = np.polyfit(tmptime, tmpseries, order, cov = True)
            outBackground[k] = np.polyval(p, e)
            
            XX = np.matrix([e ** (order - i) \
                        for i in range(order + 1)])
            
            covfit = XX * np.matrix(cov) * XX.transpose()
     
            stdBackground[k] = np.array([np.sqrt(covfit[i, i]) \
               for i in range(len(XX))])
                
    else:
        sys.exit("Not coded yet")

    return outBackground, stdBackground
            
time = np.arange(365*30)

series = createSyntheticData(time)



anomalies = computeAnomalies(time, series, order, periodicity = 365)

background = series - anomalies

fig, ax = plt.subplots(3, 1, dpi = 300, figsize = (6, 8))





timeInit = time[-1]

# !! Warning always include the initial time (0) in the set of leadTimes
# This is needed to compute the forecast variance, as it depends on the
# background estimate at initial time.

leadTimes = np.arange(0, 120, 10)

backgroundForecast, backgroundStd = extrapolateBackground(time, series, order, \
                                           leadTimes + time[-1], periodicity = 365)
    
# Show background forecast
ax[1].plot(time, background, "brown", label = "Background")
ax[1].plot(timeInit + leadTimes, backgroundForecast, lw = 0.5, color = "brown",\
              label = "Background forecast")
ax[1].fill_between(timeInit + leadTimes, backgroundForecast -  backgroundStd, \
                                         backgroundForecast +  backgroundStd, \
                                         color = "brown", alpha = 0.3, lw = 0)    

    
anomalyForecast, autocorrel =  dampedAnomalyForecast(time, anomalies, \
                                leadTimes)

forecast = anomalyForecast + backgroundForecast
    

# Show initial time
ax[2].plot(time, anomalies, color = "darkgreen")

ax[2].plot(timeInit + leadTimes, anomalyForecast, lw = 0.5 , \
             color = "darkgreen", \
             label = "Damped anomaly forecast")

# Compute forecast std
forecastStd = np.sqrt((autocorrel * backgroundStd[0] - backgroundStd) ** 2)

ax[0].plot(time, series, color = "k", label = "Raw data")
ax[0].plot(timeInit + leadTimes, forecast, lw = 0.5, color = "k", \
             label = "Damped anomaly persistence forecast")
ax[0].fill_between(timeInit + leadTimes, forecast -  forecastStd, \
                                         forecast +  forecastStd, \
                                         color = "k", alpha = 0.3, lw = 0)    
    


for a in ax:
    a.set_ylim(-6, 6)
    a.grid()
    a.set_xlim(timeInit - 365, timeInit + 2 * leadTimes[-1])
    a.legend()
    
fig.savefig("./fig.png")