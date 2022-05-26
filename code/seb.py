#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:10:45 2022

@author: tommatthews
"""

import datetime,core, pandas as pd, numpy as np, matplotlib.pyplot as plt

# Params
pc_alb_snow=95
pc_alb_ice=5

# Filenames
din="../data/"
din_clean="../data/cleaned/"
metafile=din+"meta.csv"

# Reading and allocating, etc.
meta=pd.read_csv(metafile)
nl=len(meta)
fig,ax=plt.subplots(1,1)
# Begin 
for i in range(nl):
    fname=din_clean+meta["station"].iloc[i]+".csv"
    data_i=pd.read_csv(fname,parse_dates=True,\
                         index_col="date")
        
    ds=(data_i.index[1]-data_i.index[0]).total_seconds()
        
        
    # Compute the snow and ice albedo 
    albedo_snow=np.nanpercentile(data_i["albedo"].values[:],pc_alb_snow)
    albedo_ice=np.nanpercentile(data_i["albedo"].values[:],pc_alb_ice)
    
    # Compute roughness length for momentum
    z0_m=core.Z0(data_i["albedo"].values[:],albedo_snow,albedo_ice)
    # ax.plot(data_i.index,z0_m)
    
    
    # Compute SEB
    shf, lhf, swn, lwn, seb, melt, subl =\
    core.SEB(data_i["t"].values[:],data_i["ts"].values[:],data_i["q"].values[:],\
             data_i["qs"].values[:],data_i["rho"].values[:],data_i["u"].values[:],\
                 data_i["p"].values[:],data_i["sin"].values[:],\
                    data_i["sout"].values[:],data_i["lin"].values[:],\
                        data_i["lout"].values[:],z0_m[:],data_i["zu"].values[:],\
                            data_i["zt"].values[:],data_i["zt"].values[:],ds)

    ax.plot(shf)