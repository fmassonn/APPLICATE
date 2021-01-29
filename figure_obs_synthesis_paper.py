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
mon = "july"

fig, ax = plt.subplots(1, 1, dpi = 300, figsize = (5 ,5))
ax.set_frame_on(False)

#           SATELLITE                   CONVENTIONAL
#           ALL        USED           ALL       USED
colors = [["#C9DAF9", "#256AE5"], ["#D2B29B", "#704523"] ]
shift = 0.7
exp = 0.2
for jt, typ in enumerate(["sat", "conv"]):

  for j, usage in enumerate(["all", "used"]):
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
    ax.barh(lat, width = (-1) ** jt * den ** exp ,
    #ax.barh(lat, (-1) ** jt * den ,
            height = 3.5,  \
            label = usage + "-" + typ, 
         facecolor = colors[jt][j], edgecolor = "", \
             left = (-1) ** jt * shift )

    if jt == 0 and j == 0:
        for k in range(len(lat)):
            if lat[k] > 0:
              ax.text(0.0, lat[k], str(int(lat[k] - 2.5)) + \
                      " - " + str(int(lat[k] + 2.5)), \
                          va = "center", ha = "center", color = [0.5, 0.5, 0.5], fontproperties=prop)
    del lat, den
ax.text(0.0, 98.0, mon.capitalize() + " 2020", fontproperties = prop, ha = "center", fontsize = 20)
ax.text(0.0, 91, "Latitude °N", ha = "center", fontproperties=prop)
ax.text(4.8, 93.0, "SATELLITE", fontweight = "bold", \
        color = colors[0][1], fontsize = 12, ha = "right", fontproperties=prop)
ax.text(-4.8, 93.0, "CONVENTIONAL", fontweight = "bold", \
        color = colors[1][1], fontsize = 12, ha = "left", fontproperties=prop)

#ax.grid()
ax.set_ylabel("Latitude [°N]", fontproperties=prop)
ax.set_xlabel("Observation density (km$^{-2}$)", fontproperties=prop)
ax.set_xticks(np.concatenate((-(shift + np.array([1000, 100, 10, 1]) ** exp), \
                                shift + np.array([1, 10, 100, 1000]) ** exp)))
ax.set_xticklabels( ["1000", "100", "10", "1", "1", "10", "100", "1000"])

ax.text(shift, 87.5, " assimilated", va = "center", ha = "left", color = colors[0][0], \
        fontproperties = prop, fontsize = 12)
    
ax.text(shift + 3.5 , 87.5, "all", va = "center", color = colors[0][1], \
        fontproperties = prop, fontsize = 12)

for label in ax.get_xticklabels():
    label.set_fontproperties(prop)
    
ax.set_xlim(-6.5, 6.5)
ax.plot((0.0, 0.0), (0.0, 90), "w-")
ax.set_ylim(0.0, 90.0)
#ax.legend()

ax.get_xaxis().tick_bottom()
ax.get_yaxis().set_visible(False)
xmin, xmax = ax.get_xaxis().get_view_interval()
ymin, ymax = ax.get_yaxis().get_view_interval()
ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))

fig.tight_layout()
fig.savefig("./fig" + mon + ".png")

