#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:36:12 2021

@author: massonnetf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

prop = fm.FontProperties(fname="/Users/massonnetf/Library/Fonts/ProximaNova-Regular.otf")

fac = 6 / 30 # The data was accumulated for 5-day periods (Mohamed e-mail 17 June 2021) so we multiply by fac to get daily counts

def c(xin, direction = "to", ref = 1, shift = 0.5, alpha = 10):
    if type(xin) is int or type(xin) is np.int64 or type(xin) is np.float64:
        xin = [xin]
    # This function maps the scalar line to a compressed version by respecting
    # linearity until "ref", then compressing by a factor log x/ref in a 
    # given basis alpha. There is also a shift by ref finally.
    # by putting "direction to "form", the opposite operation is done.

    xout = list()
    
    for x in xin:
        if direction == "to":
            if x <= ref:
                xout.append(x + shift)
            else:
                xout.append(shift + ref * ( 1 + np.log(x / ref) / np.log(alpha)))
        elif direction == "from":
            if x <= ref:
                xout.append(x - ref)
            else:
                xout.append(ref * alpha ** (x / ref - 2))
    xout = np.array(xout)

    if len(xout) == 1:
        xout = xout[0]
    return xout
    

datadir = "./data/ECMWF_Mohamed_new/"


# January
month = ["january", "july"]

fig, ax = plt.subplots(1, 1, dpi = 300, figsize = (9,6))

ax.set_frame_on(False)
#                     WINTER                   SUMMER
#           SATELLITE    CONVENTIONAL
#            USED         USED
colors = [["#3e6589", "#052542"], ["#F1BB46", "#CA4E00"]]

year = 2019

for jm, mon in enumerate(month):
    for jt, typ in enumerate(["sat", "conv"]):
    
      for j, usage in enumerate(["used", "all"]):
        lat = list()
        den = list()
    
        with open(datadir + "density_" + typ + "_" + usage + "_" + \
                  mon + str(year) + ".txt", "r") as f:
          for row in f:
            lattmp = float(row.split(" ")[0])
            dentmp = float(row.split(" ")[1])
            lat.append(lattmp)
            den.append(dentmp * fac) 
          
        lat = np.array(lat)
        den = np.array(den)
       
    
        #ax.set_xscale("log")
        #ax.barh(lat, width = (-1) ** jt * (np.log10(den)+shift) ,
        if usage == "used":
            ax.barh(lat  + (-1) ** jt * 1.0, width = \
                (-1) ** (jm + 1) * (c(den) - c(0)),
                height = 1.8,  \
                label = usage + "-" + typ, 
             facecolor = colors[jm][jt], edgecolor = colors[jm][jt] , lw = 0.5, \
                 left = (-1) ** (jm + 1) * c(0) )
        elif usage == "all":
            ax.barh(lat  + (-1) ** jt * 1.0, width = \
                (-1) ** (jm + 1) * (c(den) - c(0)) ,
                #ax.barh(lat, (-1) ** jt * den ,
                height = 1.8,  \
                label = usage + "-" + typ, hatch = "////////",
             facecolor = "white", edgecolor = colors[jm][jt], lw = 0.1, \
                 left = (-1) ** (jm + 1) * c(0) , zorder = -5, alpha = 0.8)
        if jt == 0 and j == 0:
            for k in range(len(lat)):
                if lat[k] > 0:
                  ax.text(0, lat[k], str(int(lat[k] - 2.5)) + \
                          "-" + str(int(lat[k] + 2.5)), \
                              va = "center", ha = "center", color = [0.5, 0.5, 0.5], fontproperties=prop)
    #print(den[lat>=60])
    del lat, den

ax.text(0.0, 91, "Latitude °N", ha = "center", fontproperties=prop)


# WINTER Legend

ax.fill_between((-c(150000), -c(85000)), (18, 18), (20, 20), \
                color = colors[0][0], lw = 0.1  )
ax.fill_between((-c(85000), -c(50000)), (18, 18), (20, 20),  \
                facecolor = "white", edgecolor = colors[0][0], \
                    hatch = "////////", lw = 0.1  )    
ax.text(-c(160000), 18, "Satellite", fontweight = "bold", \
         color = colors[0][0], fontsize = 12, ha = "right", fontproperties=prop)
ax.fill_between((-c(150000), -c(85000)), (15, 15), (17, 17),  \
                lw = 0.1 ,color = colors[0][1] )    
ax.fill_between((-c(85000), -c(50000)), (15, 15), (17, 17),  \
                facecolor = "white", edgecolor = colors[0][1], \
                    hatch = "////////", lw = 0.1  )    
ax.text(-c(160000), 15, "Conventional", fontweight = "bold", \
         color = colors[0][1], fontsize = 12, ha = "right", fontproperties=prop)
ax.text(-c(50000), 8, "WINTER", fontweight = "bold", \
         color = [0.2, 0.2, 0.2], fontsize = 18, ha = "right", fontproperties=prop)

ax.text(-c(160000), 21, "Used", rotation = 90, \
        fontproperties = prop, va = "bottom", color = [0.2, 0.2, 0.2])

ax.text(-c(80000), 21, "All", rotation = 90, \
        fontproperties = prop, va = "bottom", color = [0.2, 0.2, 0.2])

    
# SUMMER legend
ax.fill_between((c(50000), c(90000)), (18, 18), (20, 20), \
                color = colors[1][0], lw = 0.1  )
ax.fill_between((c(90000), c(150000)), (18, 18), (20, 20),  \
                facecolor = "white", edgecolor = colors[1][0], \
                    hatch = "////////" , lw = 0.1  )    
ax.text(c(160000), 18, "Satellite", fontweight = "bold", \
         color = colors[1][0], fontsize = 12, ha = "left", fontproperties=prop)
ax.fill_between((c(50000), c(90000)), (15, 15), (17, 17),  \
                lw = 0.1, color = colors[1][1] )    
ax.fill_between((c(90000), c(150000)), (15, 15), (17, 17),  \
                facecolor = "white", edgecolor = colors[1][1], \
                    hatch = "////////", lw = 0.1 )    
ax.text(c(160000), 15, "Conventional", fontweight = "bold", \
         color = colors[1][1], fontsize = 12, ha = "left", fontproperties=prop)
ax.text(c(60000), 8, "SUMMER", fontweight = "bold", \
         color = [0.2, 0.2, 0.2], fontsize = 18, ha = "left", fontproperties=prop)

ax.text(c(50000), 21, "Used", rotation = 90, \
        fontproperties = prop, va = "bottom", color = [0.2, 0.2, 0.2])

ax.text(c(100000), 21, "All", rotation = 90, \
        fontproperties = prop, va = "bottom", color = [0.2, 0.2, 0.2])

ax.set_ylabel("Latitude [°N]", fontproperties=prop)
ax.set_xlabel("Observation density (# / 1000 km$^{2}$)", fontproperties=prop)


myTicks = np.array([ 0, 1, 10, 100, 1000, 10000, 10000]) 

ax.set_xticks(np.concatenate((-c(myTicks), \
                                c(myTicks))))


ax.set_xticklabels([str(t) for t in myTicks] + \
                    [str(t) for t in myTicks] )

for tick in myTicks[1:]:
    ax.plot( (c(tick), c(tick)), (0, 90), \
            "grey", zorder = - 10, lw = 0.5)
    
    ax.plot( (- c(tick), - c(tick)), (0, 90), \
            "grey", zorder = - 10, lw = 0.5)
    

for label in ax.get_xticklabels():
    label.set_fontproperties(prop)
    
#ax.set_xlim(-c(myTicks[-1]), c(myTicks[-1]))
ax.set_xlim(-c(5000000), c(5000000))
ax.plot((0.0, 0.0), (0.0, 90), "w-")
ax.set_ylim(0.0, 90.0)
#ax.legend()

ax.get_xaxis().tick_bottom()
ax.get_yaxis().set_visible(False)
xmin, xmax = ax.get_xaxis().get_view_interval()
ymin, ymax = ax.get_yaxis().get_view_interval()
ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))

fig.tight_layout()
fig.savefig("./fig.png")
fig.savefig("./fig.pdf")

