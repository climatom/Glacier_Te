#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processes all met data, including:
    
    ...
"""

import os, datetime, pandas as pd, numpy as np, utils as ut
import matplotlib.pyplot as plt

emiss=0.98 #(warren, 1999 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501920/)
boltz=5.67*10**-8
din="../data/"
metafile=din+"meta.csv"
minpc=75 # percent complete in day/window for non-nan
date_parser = lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H:%M")

meta=pd.read_csv(metafile)
nl=len(meta)
sin_check=[]
# Plots of -- T/RH/WS/SIN/LIN/SOUT/LOUT/P
# fig,ax=plt.subplots(4,2)
# fig.set_size_inches(8,10)
# fig2,ax2=plt.subplots(1,1)
plot_list=["t","rh","u","sin","sout","lin","lout","p"]
cs=["red","blue","green","orange","cyan","yellow","pink","black"]
for i in range(nl):
    
    fig,ax=plt.subplots(1,1)
    
    fname=din+meta["station"].iloc[i]+".csv"
    data_i=pd.read_csv(fname,parse_dates=True,\
                         index_col="date",date_parser=date_parser)
    # Ensure floats
    for c in data_i.columns: data_i[c]=data_i[c].astype(float)
    # Convert time to UTC
    data_i.index=data_i.index+datetime.timedelta\
        (hours=meta["time_corr"].iloc[i])
        
    # Figure out freq (obs/hour)
    f=1./((data_i.index[1]-data_i.index[0]).total_seconds()/3600.)
    
    # Check index is correct
    data_i=data_i.sort_index()
        
    # Check s/sout and correct if necessary:
    # toa=ut.sin_toa(doy=dhour.index.dayofyear.values,hour=dhour.index.hour.values,\
    #                 lat=float(meta["lat"].iloc[i]),\
    #                 lon=float(meta["lon"].iloc[i]))
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
    
    # Compute vapor pressure (scratch var)
    ei=ut.satvp_huang(data_i["t"].values[:])
    
    # Mixing ratio (scratch var)
    mix=ut.mix(data_i["p"],ei)
    
    # Virtual temp (scratch var)
    vi=ut.virtual(data_i["t"]+273.15,mix)
    
    # Density (store)
    data_i["rho"]=ut.rho(data_i["p"]*100.,vi)
   
    
    # Compute specific humidity 
    data_i["q"]=ut.satvp_huang(data_i["t"].values[:])*0.622/(data_i["p"]*100.)*\
        data_i["rh"].values[:]
    # print("Max shum is %.2f g/kg"%(np.nanmax(data_i["q"])*1000.))
    
    # Compute the equivalent temperature (k)
    data_i["teq"]=ut.Teq(data_i["t"].values[:]+273.15,data_i["q"].values[:])
    
        
    # Calc albedo
    window_len=int(f*24)
    min_periods=int(f*24*minpc/100.)
    if "albedo" not in data_i.columns:
        data_i["albedo"]=\
            data_i["sout"].rolling(window_len,min_periods=min_periods,center=True).sum()/\
            data_i["sin"].rolling(window_len,min_periods=min_periods,center=True).sum()             
    # data_i["acc_alb"]=data_i["sout"].rolling(int(f*24)).sum(center=True)/\
    #     data_i["sin"].rolling(int(f*24)).sum(center=True)
    # ax2.plot(data_i.index,data_i["albedo"],color=cs[i],alpha=0.2)
    # print("...Min alb = %.2f, max = %.2f"%\
    #   (data_i["albedo"].min(), data_i["albedo"].max()))
        
    # Calc surface temp
    data_i["ts"]=np.power(data_i["lout"]/(boltz*emiss),0.25)-273.15
    data_i["ts"].loc[data_i["ts"]>0]=0.0
    # ax2.plot(data_i.index,data_i["ts"],color=cs[i],alpha=0.2)
    
    # Calc surface vapur pressure
    data_i["qs"]=ut.satvp_huang(data_i["ts"].values[:])*0.622/(data_i["p"]*100.)
    
    # Interpolate gaps in measurement height
    data_i["zu"].interpolate(inplace=True)
    data_i["zt"].interpolate(inplace=True)
    data_i["zq"]=data_i["zt"].values[:]
    
    # Do not trust measurements < 0.5 m above surface
    dnotrust_t=data_i["zt"]<0.5
    dnotrust_u=data_i["zu"]<0.5
    data_i["t"].loc[dnotrust_t]=np.nan
    data_i["q"].loc[dnotrust_t]=np.nan
    data_i["u"].loc[dnotrust_u]=np.nan
    
    
    # Change t and ts to K
    data_i["t"]=data_i["t"]+273.15
    data_i["ts"]=data_i["ts"]+273.15

    # Now compute daily means
    dday={}
    for v in data_i.columns:
        isnull=np.isnan(data_i[v])*1.
        isnull=isnull.resample("H").sum()
        dday[v]=data_i[v].resample("H").mean() 
        dday[v].loc[isnull>0]=np.nan
    dday=pd.DataFrame(data_i,index=isnull.index)
    
    # Plot daily means
    # for j in range(len(plot_list)):
    #     ax.flat[j].plot(dday[plot_list[j]],label=meta["station"].iloc[i],\
    #                     linewidth=0.5,color=cs[i],alpha=0.15)
    #     if j == (len(plot_list))-1 and i == nl-1:
    #         ax.flat[j].legend()

    # print("...pc 99.9 rh = %.3f %%"%np.nanpercentile(data_i["rh"],99.9))
    ax.plot(data_i.index,data_i["t"].values[:],linewidth=0.2)
    fig.savefig(din+"scratch/"+meta["station"][i]+".png")
    
    ## Write out
    if not os.path.isdir(din+"cleaned/"):
        os.makedirs(din+"cleaned/")
    dday.to_csv(din+"cleaned/"+meta["station"][i]+"_day.csv")   
    data_i.to_csv(din+"cleaned/"+meta["station"][i]+".csv")   
    print("Processed %s"%fname)