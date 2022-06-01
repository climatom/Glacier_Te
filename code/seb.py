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


# Define here the optimizing function for tdep
def tdep_f(coefs,t,tdep):
    #tdep(t,ttip,a,b)
    ttip=coefs[0]
    a=coefs[1]
    b=coefs[2]
    idx=np.logical_and(~np.isnan(t),~np.isnan(tdep))
    err=np.nanmean(np.abs(core.tdep(t[idx],ttip,a,b)-tdep[idx]))
    
    return err

# Ditto for eti
def eti_f(coefs,t,swn,melt):
    
    # eti(t,tf,sf,ttip,swn):
    tf=coefs[0]
    sf=coefs[1]
    ttip=coefs[2]
    idx=np.logical_and(~np.isnan(t),~np.isnan(swn))
    err=np.nanmean(np.abs(core.eti(t[idx],tf,sf,ttip,swn[idx])-melt))
    
    return err
    
    
        

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

# Reading and allocating, etc.
meta=pd.read_csv(metafile)
seb_meta={}
nl=len(meta)
fig,ax=plt.subplots(1,1)
log={}
err_log=np.zeros((nl,2))*np.nan
err_log_2=np.zeros(err_log.shape)*np.nan
r2_log=np.zeros(err_log.shape)
r2_log_2=np.zeros(err_log.shape)*np.nan
# Begin 
for i in range(nl):
    fname=din_clean+meta["station"].iloc[i]+".csv"
    data_i=pd.read_csv(fname,parse_dates=True,\
                         index_col="date")
    
    # Identify time step
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
    mod["tdep"]=mod["shf"]+mod["lhf"]+mod["lwn"]   
    seb_meta[meta["station"].iloc[i]]["obs_per_day"]=int(ds/3600.*24)
    mod=pd.DataFrame(mod,index=data_i.index)
    log[meta["station"][i]]=mod



# Iterate again and process to daily -- only summing days when we have at 
# least k % data availability 
print("Computations complete. Processing results to daily...")
vs=["shf","lhf","swn","lwn","tdep","melt","subl"]
vmet=["t","teq"]
cs=["green","blue","orange","purple","k","red","cyan"]
# Plot raw energy terms
fig,ax=plt.subplots(2,4)
fig.set_size_inches(8,5)
fig.set_dpi(300)
# Plot t/te vs tdep
# fig2,ax2=plt.subplots(2,4)
# fig2.set_size_inches(8,5)
# fig2.set_dpi(300)
day={}
# Begin 
for i in range(nl):
    day_i={}
    mod=log[meta["station"][i]]
    ci=0
    
    # Iterate over energy terms 
    for v in vs:
        n=seb_meta[meta["station"].iloc[i]]["obs_per_day"]
        min_hours=int(n*(k/100.))
        nval=mod[v].resample("D").count().astype(float)
        mean=mod[v].resample("D").mean(); mean[nval<min_hours]=np.nan
        total=mod[v].resample("D").sum(); total[nval<min_hours]=np.nan
        if v!= "melt" and v != "subl":
            day_i[v]=mean
            ax.flat[i].plot(day_i[v].rolling(mov,center=True).mean(),\
                            linewidth=0.4,color=cs[ci])

        else:
            day_i[v]=total

        ci+=1
        
    # Now over met terms
    for mv in vmet:
        n=seb_meta[meta["station"].iloc[i]]["obs_per_day"]
        min_hours=int(n*(k/100.))
        nval=mod[mv].resample("D").count().astype(float)
        mean=mod[mv].resample("D").mean(); mean[nval<min_hours]=np.nan
        day_i[mv]=mean; 
    
    
    # Pop in date-aware frame        
    day_i=pd.DataFrame(day_i,index=day_i[v].index)
    ax.flat[i].set_ylim(-200,280)
    ax.flat[i].grid()
    date_form = DateFormatter("%m-%y")
    ax.flat[i].xaxis.set_major_formatter(date_form)
    ax.flat[i].xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.flat[i].title.set_text(meta["station"][i])
    ax.flat[i].tick_params(rotation = 45,size=6)
    if i !=0 and i != 4:
        ax.flat[i].set_yticklabels([])
    else:
        ax.flat[i].set_ylabel("W m$^{-2}$")
        

    # ax2.flat[i].scatter(day_i[tvers].values[:],day_i["tdep"].values[:],s=2)
    day[meta["station"].iloc[i]]=day_i
    idx=np.logical_and(np.logical_and(~np.isnan(day_i["t"]),\
                                      ~np.isnan(day_i["teq"])),\
                                      ~np.isnan(day_i["tdep"]))
    y=day_i["tdep"].values[idx] 
    y2=day_i["melt"].values[idx] 
    x2=day_i["swn"].values[idx]
    ts=["t","teq"]
    nt=len(ts)
    for ti in range(nt):
        x=day_i[ts[ti]].values[idx]
        sol=minimize(fun=tdep_f,
                                x0=np.array([273.15,-50,10]),
                                args=(x,y),
                                bounds=((None,None),(None,None),(None,None)))
        pred=core.tdep(x,sol.x[0],sol.x[1],sol.x[2])
        err_log[i,ti]=np.mean(np.abs(pred-y))
        r2_log[i,ti]=np.corrcoef(pred,y)[0,1]**2

        # coefs = tf,sf,ttip
        sol2=minimize(fun=eti_f,
                                x0=np.array([5,0.5,273.]),
                                args=(x,x2,y2),
                                bounds=((None,None),(None,None),(None,None)))        
    
        #eti(t,tf,sf,ttip,swn)
        pred=core.eti(x,sol.x[0],sol.x[1],sol.x[2],x2)
        err_log_2[i,ti]=np.mean(np.abs(pred-y2))
        r2_log_2[i,ti]=np.corrcoef(pred,y2)[0,1]**2
    # ax2.flat[i].set_xlim(270,310)
    
    
    # refx=np.linspace(np.nanmin(day_i[tvers]),np.nanmax(day_i[tvers]),500)
    # refy=core.tdep(refx,sol.x[0],sol.x[1],sol.x[2])
    # ax2.flat[i].plot(refx,refy,color='red')
    # ax2.flat[i].set_ylim(-220,110)
    # ax2.flat[i].set_xlim(250,310)
    # assert i != 2
    # print(sol.x)
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.02)
fig.savefig(din+"scratch/"+"SEB.png")



    
