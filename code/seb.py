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
fig2,ax2=plt.subplots(2,4)
fig2.set_size_inches(8,5)
fig2.set_dpi(300)
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
        

    ax2.flat[i].scatter(day_i["teq"].values[:],day_i["tdep"].values[:],s=2)
    day[meta["station"].iloc[i]]=day_i
    
    x=day_i[tvers].values[:]
    y=day_i["tdep"].values[:]
    idx=np.logical_and(~np.isnan(x),~np.isnan(y))
    x=x[idx]
    y=y[idx]
    sol=minimize(fun=tdep_f,
                            x0=np.array([273.15,-50,10]),
                            args=(x,y),
                            bounds=((250,310),(None,200),(None,None)))
                            # options={'disp':True}) 
    
    # ax2.flat[i].set_xlim(270,310)
    
    
    refx=np.linspace(np.nanmin(day_i["teq"]),np.nanmax(day_i["teq"]),500)
    refy=core.tdep(refx,sol.x[0],sol.x[1],sol.x[2])
    refy=core.tdep(refx,277,-150,10)
    ax2.flat[i].plot(refx,refy,color='red')
    pred=core.tdep(x,sol.x[0],sol.x[1],sol.x[2])
    print("R for %s = %.2f"%(tvers, (np.corrcoef(x,y)[0,1])))
    # assert i != 2
    # print(sol.x)
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.02)
fig.savefig(din+"scratch/"+"SEB.png")


# Optimize
# tdep_f(coefs,t,tdep)
# coefs = [ttip,a,b]
init_guess=[]
out=minimize(fun=tdep_f,
                        x0=np.array([280,-10,10]),
                        args=(day_i["t"].values[:],day_i["tdep"].values[:]),
                        bounds=((250,310),(-100,100),(1,100)),
                        method="Nelder-Mead",
                        options={'disp':True}) 
    
