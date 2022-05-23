#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:14:20 2022

@author: k2147389
"""
import pandas as pd, numpy as np

# Constants 
c=2*np.pi # Radians in circle
r_mean=149.6 # average earth-sun distance (million km)
g=9.80665 # Gravitatonal acceleration (m/s/s)
R=287.05287 # Gas constant
e0=610.8 #satvp at 

## Functions
# Solar geom.
def deg2rad(deg):
    rad=deg/(np.pi*180.)
    return rad

# Insolation
def decl(lat,doy):
  dec=deg2rad(23.44)*np.cos(c*(doy-172.)/365.25)
  return dec

def sin_elev(dec,hour,lat,lon=0):
  lat=deg2rad(lat)
  out=np.sin(lat)*np.sin(dec) - np.cos(lat)*np.cos(dec) * \
      np.cos(c*hour/24.+deg2rad(lon))
  return out

def sun_dist(doy):
  m=c*(doy-4.)/365.25
  v=m+0.0333988*np.sin(m)+0.0003486*np.sin(2.*m)+0.0000050*np.sin(3.*m)
  r=149.457*((1.-0.0167**2)/(1+0.0167*np.cos(v)))
  return r

def sin_toa(doy,hour=12,lat=0,lon=0):
  dec=decl(lat,doy)
  _sin_elev=sin_elev(dec,hour,lat,lon)
  r=sun_dist(doy)
  s=1366.*((r_mean/r)**2)
  toa=_sin_elev*s
  toa[toa<0]=0.
  return toa

# Air press
def ISA(z,ps=1013.,Tb=288.15):
    # Note enter geopotentia height in km
    p=ps*(1+6.5/Tb*z)**-(g/(0.0065*R))
    return p
    
def satvp_huang(tc):
    tc=np.atleast_1d(tc)
    vp=np.ones(len(tc))*np.nan
    vp[tc>0]=np.exp(34.494-4924.99/(tc+237.1))/(tc+105)**1.57
    vp[tc<=0]=np.exp(43.494-6545.8/(tc+278.))/(tc+868)**2
    return vp
    