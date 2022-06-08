#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:43:47 2022

@author: tommatthews
"""
from matplotlib.dates import DateFormatter
import datetime,core, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

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
vs=["shf","lhf","swn","lwn","t","u","melt","subl","tdep","teq","rh","tlat"]
plot_vs=["shf","lhf","swn","lwn","melt"]
cs=["green","blue","orange","purple","k"]
k=75. # Percent data availability to compute stats
bootyes_glob=False # Run bootstraps on long datasets?

#============================================================================#
# Process to daily
#============================================================================#

# Reading and allocating, etc.
meta=pd.read_csv(metafile)
seb_meta={}
nl=len(meta)
day={}
fig,ax=plt.subplots(2,5)
fig.set_size_inches(15,5)
fig.set_dpi(400)
# Begin 
for i in range(nl):
    
    # Process to daily means/totals
    day_i={}
    mod=pd.read_csv(din_mod+meta["station"].iloc[i]+".csv",parse_dates=True,\
                    index_col=0)        
    mod["tdep"]=mod["shf"]+mod["lhf"]+mod["lwn"]
    mod["tlat"]=mod["teq"]-mod["t"]
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
    if i !=0 and i != 5:
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
# Bootstrap params
nboot=1000
samp_size=30
min_size=samp_size*10.
# store coefs
log_perf={}

ttip=np.zeros((nl,len(ts)))*np.nan
a=np.zeros((nl,len(ts)))*np.nan
b=np.zeros((nl,len(ts)))*np.nan
# ... and performance 
rtdep=np.zeros((nl,len(ts)))*np.nan
nboot=1000
samp_size=90
min_size=samp_size*3.
# Init coefs
coef_st=[np.array([273.15,-200,8]),np.array([280,-200,8])]# ,\
#          np.array([5,-50,10])]
    
fig,ax=plt.subplots(2,5)
fig2,ax2=plt.subplots(2,5)
axs=[ax,ax2]
keys=[] # Stores the bootstrapped locs  
for i in range(nl):
    day_i=pd.read_csv(din_mod+meta["station"].iloc[i]+"_day.csv")
    log_perf[meta["station"].iloc[i]]=np.zeros((nboot,len(ts)))*np.nan
   
    # Find index for valid data (for model fitting)   
    idx=np.logical_and(\
        np.logical_and(~np.isnan(day_i["t"]),~np.isnan(day_i["teq"])),\
                        ~np.isnan(day_i["tdep"])) 
    y=day_i["tdep"].values[idx] 
    if len(y)>min_size and bootyes_glob: 
        bootyes=True
        keys.append(meta["station"].iloc[i])
    else: bootyes=False
    
    nt=len(ts)
    # Iterate over t and teq to fit models
    for ti in range(nt):     
        x=day_i[ts[ti]].values[idx]
        ref=np.random.choice(len(y)) 
        p0=[np.mean(x),np.percentile(y,25),10]
        if bootyes:           
            # Bootstrap if big dataset
            for bi in range(nboot):
                idxi=np.random.choice(ref,samp_size)
                xi=x[idxi]
                yi=y[idxi]     

                # Optimize tdep function       
                sol=minimize(fun=tdep_f,
                                        x0=coef_st[ti],
                                        args=(xi,yi))

                                        # bounds=((None,None),(None,None),(None,None)),
                                        # method="cg")
                pred=core.tdep(xi,sol.x[0],sol.x[1],sol.x[2]) 
                log_perf[meta["station"].iloc[i]][bi,ti]\
                    =np.corrcoef(pred,yi)[0,1]**2
            print("Finished bootstrap for var %s and loc %s "%\
                  (ts[ti],meta["station"].iloc[i]))
            # if meta["station"].iloc[i] == "qqg" and ti == 1:
            #     assert 1==2 
   
        # Optimize tdep function (global)    
        # sol=minimize(fun=tdep_f,
        #              x0=coef_st[ti],
        #              args=(x,y))
        sol2=curve_fit(core.tdep,x,y,p0=p0)
        pred2=core.tdep(x,sol2[0][0],sol2[0][1],sol2[0][2])
                    # bounds=((None,None),(None,None),(None,None)),
                                        # method="cg")
        
        # pred=core.tdep(x,sol.x[0],sol.x[1],sol.x[2]) 
        rtdep[i,ti]=np.corrcoef(pred2,y)[0,1]**2     
        
        axs[ti].flat[i].scatter(x,y,s=0.5)
        axs[ti].flat[i].scatter(x,pred2,s=0.5)
        
        
# Plot boot perf
if bootyes_glob:
    fig,ax=plt.subplots(4,2)
    cs=["green","purple"]
    for i in range(len(keys)):
        for ti in range(len(ts)):
            yi=log_perf[keys[i]][:,ti]
            yi[np.abs(yi)<0.0001]=np.nan
            yi=yi[~np.isnan(yi)]
            if len(yi)<2: continue
            ax.flat[i].hist(yi,bins=15,color=cs[ti],density=True)
            ax.flat[i].set_ylim(0,10)