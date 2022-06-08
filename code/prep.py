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
    
    if "ice.csv" in fname or "kerst" in fname or "kan" in fname:
        # Raw files don't have day/month/year, so process differently
        data_i=pd.read_csv(fname)
        nobs=len(data_i)
        dates=np.array([datetime.datetime(year=data_i["year"].iloc[ii],month=1,\
                       day=1,hour=0)+\
            datetime.timedelta(days=float(data_i["dayofyear"].iloc[ii])-1) \
                for ii in range(nobs)])
        dates=[dates[ii]+datetime.timedelta(hours=float(data_i["hour"].iloc[ii])) \
               for ii in range(nobs)]
        data_i.index=dates
        data_i.index.name="date"
    else:
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
    data_i["sin"].loc[data_i["sin"]<0]=0
    data_i["sin"].loc[data_i["sin"]<data_i["sout"]]=\
        data_i["sout"].loc[data_i["sin"]<data_i["sout"]]
    data_i["sin"].loc[data_i["sin"]>1400]=np.nan
    data_i["sout"].loc[np.isnan(data_i["sin"])]=np.nan
        
    # # Check and correct Lout if necessary:
    lout_theory=emiss*boltz*273.15**4
    data_i["lout"].loc[data_i["lout"]>lout_theory*1.1]=np.nan
    data_i["lin"].loc[data_i["lout"]>lout_theory*1.1]=np.nan
    # Also correct lin if greater than blackbody radiator?
    # lin_theory=(data_i["t"]+273.15)**4*boltz
    # data_i["lin"].loc[data_i["lin"]>lin_theory]=np.nan
    
    # Correct the RH (ensuring all are fraction)
    if np.max(data_i["rh"])>10:
        data_i["rh"]=data_i["rh"]/100.
        data_i["rh"].loc[data_i["rh"]>1]=1
        data_i["rh"].loc[data_i["rh"]<0]=0

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
    
    # Resample to hour
    dhour={}
    for v in data_i.columns:
        # isnull=np.isnan(data_i[v])*1.
        # isnull=isnull.resample("H").sum()
        dhour[v]=data_i[v].resample("H").mean()
        # dhour[v].loc[isnull>0]=np.nan
        # 
    dhour=pd.DataFrame(dhour,index=dhour[v].index)
    
    # Create index for padding the hourly
    ref_idx=pd.date_range(start=dhour.index[0],end=dhour.index[-1],freq="H")
    dhour=dhour.reindex(ref_idx)

    
    # ****  Calc albedo for the *hourly* data
    # window_len=int(f*24)
    # min_periods=int(f*24*minpc/100.)
    window_len=24
    min_periods=window_len*minpc/100.
    dhour["albedo"]=\
    dhour["sout"].rolling(window_len,center=True).sum()/\
    dhour["sin"].rolling(window_len,center=True).sum()            
    nvalid=(dhour["sin"]+dhour["sout"]).rolling(window_len,center=True).\
                count()
    dhour["albedo"].loc[nvalid<min_periods]=np.nan

    # Now compute daily means
    dday={}
    for v in data_i.columns:
        isnull=np.isnan(data_i[v])*1.
        isnull=isnull.resample("D").sum()
        dday[v]=data_i[v].resample("D").mean() 
        dday[v].loc[isnull>0]=np.nan
    dday=pd.DataFrame(dday,index=isnull.index)
     
    ## Write out
    if not os.path.isdir(din+"cleaned/"):
        os.makedirs(din+"cleaned/")
    dday.to_csv(din+"cleaned/"+meta["station"][i]+"_day.csv",index_label="date")  
    dhour.to_csv(din+"cleaned/"+meta["station"][i]+"_hour.csv",index_label="date")  
    data_i.to_csv(din+"cleaned/"+meta["station"][i]+".csv",index_label="date")     
    print("Processed %s"%fname)
