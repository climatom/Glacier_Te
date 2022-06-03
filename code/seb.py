#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:10:45 2022

@author: tommatthews
"""

from matplotlib.dates import DateFormatter
import datetime,core, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize

font = {'family' : 'normal',
        'size'   : 6}

plt.rc('font', **font)


    
# Params
pc_alb_snow=95
pc_alb_ice=5
k=75 # Percent data availability to compute daily values
mov=5 # Ndays in moving average
date_form = DateFormatter("%m/%y")
tvers="teq"

# Filenames
din="../data/"
din_clean="../data/cleaned/"
metafile=din+"meta.csv"
outdir="../data/modelled/"

# Reading and allocating, etc.
meta=pd.read_csv(metafile)
seb_meta={}
nl=len(meta)
log={}

# Init plots
fig,ax=plt.subplots(1,1)

# Begin 
for i in range(nl):
    fname=din_clean+meta["station"].iloc[i]+"_hour.csv"
    data_i=pd.read_csv(fname,parse_dates=True,\
                         index_col="date")
    
    # Identify time step (3600 sec for hourly)
    ds=(data_i.index[1]-data_i.index[0]).total_seconds()
                
    # Compute the snow and ice albedo 
    albedo_snow=np.nanpercentile(data_i["albedo"].values[:],pc_alb_snow)
    albedo_ice=np.nanpercentile(data_i["albedo"].values[:],pc_alb_ice)
    
    # Compute roughness length for momentum
    z0_m=core.Z0(data_i["albedo"].values[:],albedo_snow,albedo_ice)
    # ax.plot(data_i.index,z0_m)
        
    # Compute SEB
    mod={}
    seb_meta[meta["station"].iloc[i]]={}
    # SEB(ta,ts,qa,qs,rho,u,p,sin,sout,lin,lout,z0_m,zu,zt,zq,ds)
    mod["shf"],mod["lhf"],mod["swn"],mod["lwn"],mod["seb"],mod["melt"],mod["subl"] =\
    core.SEB(data_i["t"].values[:],data_i["ts"].values[:],data_i["q"].values[:],\
             data_i["qs"].values[:],data_i["rho"].values[:],data_i["u"].values[:],\
                 data_i["p"].values[:],data_i["sin"].values[:],\
                    data_i["sout"].values[:],data_i["lin"].values[:],\
                        data_i["lout"].values[:],z0_m[:],data_i["zu"].values[:],\
                            data_i["zt"].values[:],data_i["zt"].values[:],ds)
        
   # Also copy across some met
    mod["t"]=data_i["t"]
    mod["teq"]=data_i["teq"]
    mod["q"]=data_i["q"]
    mod["u"]=data_i["u"]   
    mod["tdep"]=mod["shf"]+mod["lhf"]+mod["lwn"] 
    mod["sin"]=data_i["sin"]
    mod["lin"]=data_i["lin"]
    mod["sout"]=data_i["sout"]
    mod["lout"]=data_i["lout"]
    mod["rh"]=data_i["rh"]
    mod=pd.DataFrame(mod,index=data_i.index)
    mod.to_csv(outdir+meta["station"].iloc[i]+".csv")
    # seb_meta[meta["station"].iloc[i]]["obs_per_day"]=int(ds/3600.*24)
    # mod=pd.DataFrame(mod,index=data_i.index)
    # log[meta["station"][i]]=mod
    
    


    
