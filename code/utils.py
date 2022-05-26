#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:14:20 2022

@author: k2147389
"""
import pandas as pd, numpy as np
from numba import jit, int32, float32, float64

# Constants 
c=2*np.pi # Radians in circle
r_mean=149.6 # average earth-sun distance (million km)
g=9.80665 # Gravitatonal acceleration (m/s/s)
R=287.05287 # Gas constant
e0=610.8 #satvp at 
epsilon=0.622 # Ratio of gas constants for dry/moist air 
rd=287.053 # Gas constant for dry air (J/K/kg)
e0=611.0 # Constant to evaluate vapour pressure in Clasius Clapeyron equation (Pa)
boltz=5.67*10**-8 # Stefan Boltzmann constant

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

@jit("float64[:](float64[:])")
def satvp_huang(tc):
    vp=np.ones(len(tc))*np.nan
    vp[tc>0]=np.exp(34.494-4924.99/(tc[tc>0]+237.1))/(tc[tc>0]+105)**1.57
    vp[tc<=0]=np.exp(43.494-6545.8/(tc[tc<=0]+278.))/(tc[tc<=0]+868)**2
    return vp


def mix(p,e):

    """
    This function computes the mixing ratio
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)  : air pressure
        - e (Pa)  : vapour pressure 
        
    Out:
        
        - mr (kg vapour/kg dry air)   : mixing ratio
    """

    mr=epsilon*e/(p-e)
    
    return mr


def virtual(t,mr):
    
    """
    This function computes the virtual temperature
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (K)    : air pressure
        - mr (kg vapour/kg dry air)   : mixing ratio
        
    Out:
        
        - tv (K)   : virtual temperature
    """    
    
    tv=t*(1+mr/epsilon)/(1.+mr)
    
    return tv


def rho(p,tv):
    
    """
    Computes the air density
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)  : air pressure
        - tv (K)  : virtual temperature
        
    Out:
    
        - rho (kg/m^3) : air density
        
    """    
    
    rho=np.divide(p,np.multiply(rd,tv))
    
    return rho