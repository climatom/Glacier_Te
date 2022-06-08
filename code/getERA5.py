#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:42:55 2022

@author: k2147389
"""

import os,datetime,cdsapi, pandas as pd, numpy as np
from calendar import monthrange
c = cdsapi.Client(quiet=False,debug=True)

din="../data/"
metafile=din+"meta.csv"
scratch_dir=din+"scratch/"
meta=pd.read_csv(metafile)
meta["start"]=pd.to_datetime(meta.start,format='%d/%m/%Y %H:%M')
meta["end"]=pd.to_datetime(meta.end, format='%d/%m/%Y %H:%M')
nl=len(meta)
for i in range(nl):
    loc=meta["station"].iloc[i]

    lat_l=np.round(float(meta["lat"].iloc[i])-1)
    lat_u=np.round(float(meta["lat"].iloc[i])+1)
    lon_l=np.round(float(meta["lon"].iloc[i])-1)
    lon_u=np.round(float(meta["lon"].iloc[i])+1)  
    plevs=["%.0f" %ii for ii in range(meta["lower_p"].iloc[i],\
                                     meta["upper_p"].iloc[i]+50,50)]
    times=["%.02d:00"%ii for ii in range(24)]
    area=[lat_u,lon_l,lat_l,lon_u]
    start=meta["start"].iloc[i]
    end=meta["end"].iloc[i]
    end_month_days=monthrange(end.year,end.month)[1]
    
    _start=datetime.datetime(year=start.year,month=start.month,day=1,hour=0)
    _end=datetime.datetime(year=end.year,month=end.month,day=end_month_days,
                           hour=23)    
    date_range=pd.date_range(start=_start,end=_end,freq="M")
    oname_base=scratch_dir+"era5_"+loc 
    file_count=0
    for d in date_range:
        oname=oname_base+"_%.0f.nc"%file_count 
        if os.path.isfile(oname): file_count+=1; continue        
        days=["%.0f"%ii for ii in range(1,monthrange(d.year,d.month)[1]+1)]
        print("\n\nGetting...\nyear:%.0f\nmonth:%02d\n\n"%(d.year,d.month))
        
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'geopotential', 'specific_humidity', 'temperature',
                ],
                'pressure_level': plevs,
                'year': '%.0f'%d.year,
                'month': '%.02d'%d.month,
                'day': days,
                'time': times,
                'area': area,
                'format': 'netcdf',
            },
            oname)
            
        file_count+=1