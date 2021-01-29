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


datadir = "./data/ECMWF_Mohamed/"


# January
month = ["january", "july"]

fig, ax = plt.subplots(1, 1, dpi = 300, figsize = (5,5))

ax.set_frame_on(False)
#                     WINTER                   SUMMER
#           SATELLITE    CONVENTIONAL
#            USED         USED
colors = [["#3e6589", "#052542"], ["#F1BB46", "#CA4E00"]]
shift = 35
exp = 1

for jm, mon in enumerate(month):
    for jt, typ in enumerate(["sat", "conv"]):
    
      for j, usage in enumerate(["used"]):
        lat = list()
        den = list()
    
        with open(datadir + "density_" + typ + "_" + usage + "_" + \
                  mon + "2020.txt", "r") as f:
          for row in f:
            lattmp = float(row.split(" ")[0])
            dentmp = float(row.split(" ")[1])
            lat.append(lattmp)
            den.append(dentmp)
          
        lat = np.array(lat)
        den = np.array(den)
       
    
        print(den)
        #ax.set_xscale("log")
        #ax.barh(lat, width = (-1) ** jt * (np.log10(den)+shift) ,
        ax.barh(lat  + (-1) ** jt * 1.0, width = (-1) ** (jm + 1) * den ** exp ,
        #ax.barh(lat, (-1) ** jt * den ,
                height = 2,  \
                label = usage + "-" + typ, 
             facecolor = colors[jm][jt], edgecolor = "white", lw = 0.5, \
                 left = (-1) ** (jm + 1) * shift)
    
        if jt == 0 and j == 0:
            for k in range(len(lat)):
                if lat[k] > 0:
                  ax.text(0, lat[k], str(int(lat[k] - 2.5)) + \
                          " - " + str(int(lat[k] + 2.5)), \
                              va = "center", ha = "center", color = [0.5, 0.5, 0.5], fontproperties=prop)
    del lat, den

ax.text(0.0, 91, "Latitude °N", ha = "center", fontproperties=prop)

ax.fill_between((-175, -165), (38, 38), (40, 40),  color = colors[0][0] )    
ax.text(-180, 38.0, "Satellite", fontweight = "bold", \
        color = colors[0][0], fontsize = 12, ha = "right", fontproperties=prop)
ax.fill_between((-175, -165), (35, 35), (37, 37),  color = colors[0][1] )    
ax.text(-180, 35.0, "Conventional", fontweight = "bold", \
        color = colors[0][1], fontsize = 12, ha = "right", fontproperties=prop)
ax.text(-180, 30.0, "WINTER", fontweight = "bold", \
        color = colors[0][0], fontsize = 18, ha = "right", fontproperties=prop)

ax.fill_between((150, 160), (38, 38), (40, 40),  color = colors[1][0] )    
ax.text(170, 38.0, "Satellite", fontweight = "bold", \
        color = colors[1][0], fontsize = 12, ha = "left", fontproperties=prop)
ax.fill_between((150, 160), (35, 35), (37, 37),  color = colors[1][1] )    
ax.text(170, 35.0, "Conventional", fontweight = "bold", \
        color = colors[1][1], fontsize = 12, ha = "left", fontproperties=prop)

ax.text(145, 30.0, "SUMMER", fontweight = "bold", \
        color = colors[1][0], fontsize = 18, ha = "left", fontproperties=prop)

#ax.grid()
ax.set_ylabel("Latitude [°N]", fontproperties=prop)
ax.set_xlabel("Observation density (km$^{-2}$)", fontproperties=prop)
ax.set_xticks(np.concatenate((-(shift + np.array([300, 200, 100, 0])), \
                                shift + np.array([0, 100, 200, 300]))))
ax.set_xticklabels( ["300", "200", "100", "0", "0", "100", "200", "300"])



for label in ax.get_xticklabels():
    label.set_fontproperties(prop)
    
ax.set_xlim(-350, 350)
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

