#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:43:47 2022

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
def eti_f(coefs,t,swn,melt,ttip):
    
    # eti(t,tf,sf,ttip,swn):
    tf=coefs[0]
    sf=coefs[1]
    # ttip=coefs[2]
    idx=np.logical_and(~np.isnan(t),~np.isnan(swn))
    err=np.nanmean(np.abs(core.eti(t[idx],tf,sf,ttip,swn[idx])-melt[idx]))
    
    return err
    
# Filenames
din="../data/"
din_mod="../data/modelled/"
metafile=din+"meta.csv"
vs=["shf","lhf","swn","lwn","t","u","melt","subl","tdep","teq"]
plot_vs=["shf","lhf","swn","lwn","melt"]
cs=["green","blue","orange","purple","k"]
k=75. # Percent data availability to compute stats

#============================================================================#
# Process to daily
#============================================================================#

# Reading and allocating, etc.
meta=pd.read_csv(metafile)
seb_meta={}
nl=len(meta)
day={}
fig,ax=plt.subplots(2,4)
fig.set_size_inches(8,5)
# Begin 
for i in range(nl):
    
    # Process to daily means/totals
    day_i={}
    mod=pd.read_csv(din_mod+meta["station"].iloc[i]+".csv",parse_dates=True,\
                    index_col=0)        
    mod["tdep"]=mod["shf"]+mod["lhf"]+mod["lwn"]
    plot_i=0
    for v in vs:
        min_hours=int(24*(k/100.))
        nval=mod[v].resample("D").count().astype(float)
        mean=mod[v].resample("D").mean(); mean[nval<min_hours]=np.nan
        total=mod[v].resample("D").sum(); total[nval<min_hours]=np.nan
        if v!= "melt" and v != "subl":
            day_i[v]=mean
            
        else:
            day_i[v]=total
        if v in plot_vs:
            ax.flat[i].plot(day_i[v].index,day_i[v].rolling(5).mean(),\
                color=cs[plot_i],linewidth=0.5)
            plot_i+=1


    # Pop in date-aware frame     
    day_i=pd.DataFrame(day_i,index=day_i[v].index)
               
    # Write out
    day_i.to_csv(din_mod+meta["station"].iloc[i]+"_day.csv")
    


# Format plot
for i in range(nl):
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
        
           
plt.tight_layout()
fig.subplots_adjust(wspace=0.02,hspace=0.3)
fig.savefig(din+"scratch/"+"SEB.png")

#============================================================================#
# Fit and eval mods (t vs teq), etc.
#============================================================================#
ts=["t","teq"]
# store coefs
ttip=np.zeros((nl,2))*np.nan
a=np.zeros((nl,2))*np.nan
b=np.zeros((nl,2))*np.nan
# tf=np.zeros((nl,2))*np.nan
# sf=np.zeros((nl,2))*np.nan
# ttip2=np.zeros((nl,2))*np.nan
# ... and performance 
rtdep=np.zeros((nl,2))*np.nan
# reti=np.zeros((nl,2))*np.nan

for i in range(nl):
    day_i=pd.read_csv(din_mod+meta["station"].iloc[i]+"_day.csv")
   
    # Find index for valid data (for model fitting)   
    idx=np.logical_and(\
        np.logical_and(~np.isnan(day_i["t"]),~np.isnan(day_i["teq"])),\
                        np.logical_and(~np.isnan(day_i["tdep"]),\
                                      ~np.isnan(day_i["swn"])))
        
    melt_t=np.nanmin(day_i["t"].loc[day_i["melt"]>0])
    melt_teq=np.nanmin(day_i["teq"].loc[day_i["melt"]>0])
    
    y=day_i["tdep"].values[idx] 
    y2=day_i["melt"].values[idx] 
    x2=day_i["swn"].values[idx]
    nt=len(ts)
    melt_ts=[melt_t,melt_teq]
    
    # Iterate over t and teq to fit models
    for ti in range(nt):     
        x=day_i[ts[ti]].values[idx]
        
        # Optimize tdep function       
        sol=minimize(fun=tdep_f,
                                x0=np.array([273.15,-50,10]),
                                args=(x,y))
                                # bounds=((None,None),(None,None),(None,None)),
                                # method="cg")
        pred=core.tdep(x,sol.x[0],sol.x[1],sol.x[2])   
        ttip[i,ti]=sol.x[0]; a[i,ti]=sol.x[1]; b[i,ti]=sol.x[2]
        rtdep[i,ti]=np.corrcoef(pred,y)[0,1]**2
        
   
        # # Repeat, ETI model
        # sol2=minimize(fun=eti_f,
        #                         x0=np.array([0.01,0.1,]),
        #                         args=(x,x2,y2,melt_ts[ti]))
        #                         # bounds=((None,None),(None,None),(250,300)),
        #                         # method="cg")
        # #eti(t,tf,sf,ttip,swn)
        # pred2=core.eti(x,sol2.x[0],sol2.x[1],melt_ts[ti],x2)
        # tf[i,ti]=sol2.x[0]; sf[i,ti]=sol2.x[1]; ttip2[i,ti]=melt_ts[ti]
        # reti[i,ti]=np.corrcoef(pred2,y2)[0,1]**2