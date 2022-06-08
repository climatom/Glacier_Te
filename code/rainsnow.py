#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:34:35 2022

@author: k2147389
"""

from simpledbf import Dbf5
import pandas as pd, numpy as np, utils as ut
from haversine import haversine_vector, Unit

dmax=10.
dzmax=100.
din="../../RainSnow/"
fin=din+"glacier/rgi60_global.dbf"
fin2=din+"stations/jennings_et_al_2018_file1_station_locs_elev.csv"
fin3=din+"stations/jennings_et_al_2018_file2_ppt_phase_met_observations.csv"
data=Dbf5(fin)
data = data.to_dataframe()
sel=['CENLON', 'CENLAT','ZMIN', 'ZMAX', 'ZMED', 'SLOPE']
data[sel].to_csv(din+"glacier_meta.csv")
stats=pd.read_csv(fin2)
nstat=len(stats)
subset=[]
for ii in range(nstat):
    dist=ut.distance(stats["Latitude"].iloc[ii],\
                                stats["Longitude"].iloc[ii],\
                                data["CENLAT"].values[:],\
                                data["CENLON"].values[:])
    dz=np.abs(stats["Elevation"].iloc[ii]-data["ZMIN"])
    idx=np.logical_and(dist<=dmax,dz<=dzmax)
    if idx.any():
        subset.append(stats["Station_ID"].iloc[ii])
    
idx=[True if stats["Station_ID"][ii] in subset else False for ii in range(nstat)]
subset=stats.loc[idx]

# To do -- 
# Extract stations in subset and write to csv files (inc. te)
# Fit function as defined here:
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008GL033295