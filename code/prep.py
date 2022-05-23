#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processes all met data, including:
    
    ...
"""

import os, datetime, pandas as pd, numpy as np, utils as ut
import matplotlib.pyplot as plt

emiss=0.985 #(warren, 1999 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501920/)
boltz=5.67*10**-8
din="/Users/k2147389/Desktop/Papers/Glacier_Te/Public/data/"
metafile=din+"meta.csv"
    

meta=pd.read_csv(metafile)
nl=len(meta)
sin_check=[]
# Plots of -- T/RH/WS/SIN/LIN/SOUT/LOUT/P
fig,ax=plt.subplots(4,2)
fig.set_size_inches(8,10)
plot_list=["t","rh","u","sin","sout","lin","lout","p"]
cs=["red","blue","green","orange","cyan","yellow","pink","black"]
for i in range(nl):
    fname=din+meta["station"].iloc[i]+".csv"
    data_i=pd.read_csv(fname,parse_dates=True,\
                         index_col="date")
    # Ensure floats
    for c in data_i.columns: data_i[c]=data_i[c].astype(float)
    # Convert time to UTC
    data_i.index=data_i.index+datetime.timedelta\
        (hours=meta["time_corr"].iloc[i])
        
    # Check s/sout and correct if necessary:
    # toa=ut.sin_toa(doy=dhour.index.dayofyear.values,hour=dhour.index.hour.values,\
    #                lat=float(meta["lat"].iloc[i]),\
    #                lon=float(meta["lon"].iloc[i]))
    data_i["sin"].loc[data_i["sin"]<0]=0
    data_i["sin"].loc[data_i["sin"]<data_i["sout"]]=\
        data_i["sout"].loc[data_i["sin"]<data_i["sout"]]
    data_i["sin"].loc[data_i["sin"]>1400]=np.nan
    data_i["sout"].loc[np.isnan(data_i["sin"])]=np.nan
        
    # # Check and correct Lout if necessary:
    lout_theory=emiss*boltz*273.15**4
    data_i["lout"].loc[data_i["lout"]>lout_theory*1.05]=np.nan
    data_i["lin"].loc[data_i["lout"]>lout_theory*1.05]=np.nan
    
    # Correct the RH (ensuring all are fraction)
    if np.max(data_i["rh"])>10:
        data_i["rh"]=data_i["rh"]/100.
        data_i["rh"].loc[data_i["rh"]>1]=1
        data_i["rh"].loc[data_i["rh"]<0]=0
        
        
        
    # Resample to hour
    dhour={}
    for v in data_i.columns:
        isnull=np.isnan(data_i[v])*1.
        isnull=isnull.resample("H").sum()
        dhour[v]=data_i[v].resample("H").mean()
        dhour[v].loc[isnull>0]=np.nan
    dhour=pd.DataFrame(dhour,index=isnull.index)

    # Compute profile
    # sin_prof=dhour["sin"].groupby(dhour.index.hour).apply(resample_func)
    # sin_check.append(sin_prof)
    # ax.plot(sin_prof)
    # print("Processed %s"%fname)
    
    # Fix air pressure -- if it was measured, use the LTM; otherwise use
    # the ISA
    if np.isnan(data_i["p"]).all():
        zi=meta["elev"].iloc[i]
        data_i["p"]=ut.ISA(zi/1000.)
    else:
        data_i["p"].loc[np.isnan(data_i["p"])]=np.nanmean(data_i["p"])


    # Now compute daily means
    dday={}
    for v in data_i.columns:
        isnull=np.isnan(data_i[v])*1.
        isnull=isnull.resample("H").sum()
        dday[v]=data_i[v].resample("H").mean()
        dday[v].loc[isnull>0]=np.nan
    dday=pd.DataFrame(data_i,index=isnull.index)
    
    # Plot daily means
    for j in range(len(plot_list)):
        ax.flat[j].plot(dday[plot_list[j]],label=meta["station"].iloc[i],\
                        linewidth=0.5,color=cs[i],alpha=0.15)
        if j == (len(plot_list))-1 and i == nl-1:
            ax.flat[j].legend()
    print("Processed %s"%fname)
    print("...pc 99.9 rh = %.3f %%"%np.nanpercentile(data_i["rh"],99.9))
    
    ## Write out
    if not os.path.isdir(din+"cleaned/"):
        os.makedirs(din+"cleaned/")
    dday.to_csv(din+"cleaned/"+meta["station"][i]+"_day.csv")    