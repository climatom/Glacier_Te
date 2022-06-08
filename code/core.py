
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Core SEB functions
"""
import numpy as np, pandas as pd
from numba import jit, float64, float32, int32
import numba as nb
import scipy.optimize as optimize
from scipy.optimize import minimize
import warnings

#_____________________________________________________________________________#
# Define constants. 


# NOTE -- some 'legacy' constants may be included here -- called by functions
# below, but not necessrily in the SEB routine used by Matthews et al. (2020)
#_____________________________________________________________________________#

k=0.40 # Von Karman Constant
z0_m = 5*10**-4# Roughness length for snow (m; Oke )
v=1.46*10**-5 # kinematic viscosity of air  *** CHECK
epsilon=0.622 # Ratio of gas constants for dry/moist air 
rd=287.053 # Gas constant for dry air (J/K/kg)
g=9.81 # Acceleration due to gravity
cp=1005.7 # Specific heat capacity of dry air at constant pressure (J/kg)
Le=2.501*10**6 # Latent heat of vaporization (J/kg)
Ls=2.834*10**6 # Latent heat of sublimation (J/kg)
Lf=3.337*10**5 # Latent heat of fusion (J/kg)
Rv=461.0 # Gas constant for water vapour (J/K/kg)
e0=611.0 # Constant to evaluate vapour pressure in Clasius Clapeyron equation (Pa)
maxiter=10 # Max iterations allowed in the shf/MO scheme 
maxerr=1 # Maximum error allowed for the shf/MO scheme to settle (%)
# Beer-Lambert function (see Wheler and Flowers, 2011; units = 1/m)
rho_i=910.0 # Density of ice (kg/m^3)
boltz=5.67*10**-8 # Stefan Boltzmann constant
emiss=0.98 # Thermal emissivity of ice (dimensionless)
ratio=True # True=assume z0_t and z0_q are fixed fraction of z0_m
# (otherwise use the expresssions of Andreas (see ROUGHNESS))
roughness_ratio=10 # Divide z0_m by this to find z0_h and z0_q (if ratio=True)
zsnow=2.6/1000. # Now in m (same for below)
zice=7.4/1000.



#_____________________________________________________________________________#
# Begn function definitions

# Turbulent heat fluxes
#_____________________________________________________________________________#


@jit(float64[:](float64[:],float64,float64),nopython=True)
def Z0(albedo,albedo_snow,albedo_ice):
    """
    Computes the roughness length using albedo to interpolate between 
    the snow/ice roughness lengths
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - albedo (dimensionless): sout/sin
        - albedo_snow ("")      : albedo of snow
        - albedo_ice  ("")      : albedo of sice
    Out:
    
        - z0                    : roughness length (m)
        
    Note: zsnow and zice are global variables 
   
    """
    n=len(albedo)
    z0=np.zeros(n)*np.nan
    for i in range(n):
        if albedo[i]>albedo_snow: albedo[i]=albedo_snow
        if albedo[i]<albedo_ice: albedo[i]=albedo_ice
        
        z0[i]=zsnow+(zice-zsnow)*\
            (1-(albedo[i]-albedo_ice)/(albedo_snow-albedo_ice))
    
    return z0

@jit(nb.typeof((1.,1.,1.,))(float64),nopython=True)
def STAB(zl):
    
    """
    Computes the stability functions to permit deviations from the neutral 
    logarithmic profile
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - zl (dimensionless): ratio of ~measurement height to the MO length
        
    Out:
    
        - corr_m (dimensionless) : stability correction for momentum
        - corr_h (dimensionless) : stability correction for sensible heat
        - corr_q (dimensionless) : stability correction for vapour
        
    Note: for the stable case (positive lst), the stability functions of 
    Holtslag and de Bruin (1988) are used; for unstable conditions 
    (negative lst), the functions of Dyer (1974) are applied.
   
    
    """
    # Coefs
    a=1.; b=0.666666; c=5; d=0.35
    
    # Stable case
    if zl >0:
        
       corr_m=(zl+b*(zl-c/d)*np.exp(-d*zl)+(b*c/d))*-1
       
       corr_h=corr_q=(np.power((a+b*zl),(1/b)) + \
       b*(zl-c/d)*np.exp(-d*zl)+(b*c)/d-a)*-1
        
    # Unstable
    elif zl<0:
        
        corr_m=np.power((1-16.*zl),-0.25)
        corr_h=corr_q=np.power((1-16.*zl),-0.5)
    
    # Neutral    
    else: corr_m=corr_h=corr_q=0.0
               
        
    return corr_m, corr_h, corr_q

@jit(nb.typeof((1.,1.,1.,1.))(float64,float64,float64,float64),nopython=True)
def USTAR(u,zu,lst,z0_m):
    
    """
    Computes the friction velocity.
    
    NOTE: this function also returns the stability corrections!
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - u (m/s) : wind speed at height zu
        - zu (m)  : measurement height of wind speed
        - lst (m) : MO length
        - z0_m (m): roughness length of momentum
        
    Out:
    
        - ust (m/s) : friction velocity
        
    """

    corr_m,corr_h,corr_q=STAB((zu-z0_m)/lst)    
    ust=k*u/(np.log(zu/z0_m)-corr_m)
    
    return ust,corr_m,corr_h,corr_q



@jit(nb.typeof((1.,1.,1.))(float64,float64),nopython=True)
def ROUGHNESS(ust,z0_m):
    
    """
    Computes the roughness lenghths for heat and vapour
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ust (m/s) : friction velocity
        - z0_m (m): roughness length of momentum
        
    Out:
    
        - z0_h (m) : roughness length for heat
        - z0_q (m) : roughness length for vapour
        
    Note: when stable, 
    """
    # Reynolds roughness number
    re=(ust*z0_m)/v
    
    # Use the fixed ratio method if requested. Then return (so rest of code
    # will be skipped)
    if ratio:
        z0_h=z0_m/roughness_ratio
        z0_q=z0_m/roughness_ratio
        return z0_h, z0_q, re
    
    assert re <= 1000, "Reynolds roughness number >1000. Cannot continue..."
    
    # Equation terms 
    f=np.array([1,np.log(re),np.power(np.log(re),2)])  
       
    # Coefficients
    # heat
    h_coef=np.zeros((3,3))
    h_coef[0,:]=[1.250,0.149,0.317]
    h_coef[1,:]=[0,-0.550,-0.565]
    h_coef[2,:]=[0,0,-0.183]
    # vapour
    q_coef=np.zeros((3,3))
    q_coef[0,:]=[1.610,0.351,0.396]
    q_coef[1,:]=[0,-0.628,-0.512]
    q_coef[2,:]=[0,0,-0.180]    
    
    # Use roughness to determine which column in coefficient array we should 
    # use
    if re<=0.135: col=0
    elif re>0.135 and re<2.5: col=1
    else: col=2
        
    # Compute stability functions with the dot product
    z0_h=np.exp(np.dot(f,h_coef[:,col]))*z0_m
    z0_q=np.exp(np.dot(f,q_coef[:,col]))*z0_m
    
    return z0_h, z0_q, re

@jit(nb.float64(float64,float64,float64,float64),nopython=True)
def MO(t,ust,h,rho):
    
    """
    Computes the MO length. 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (K)       : air temperature 
        - ust (m/s)   : friction velocity
        - h (W/m^2)   : sensible heat flux
        - rho (kg/m^3): air density 
        
    Out:
    
        - lst (m)    : MO length
        
    
    Note that we use the air temperature, not the virtual temperature 
    (as Hock and Holmgren [2005]). Lst is positive when h is positive -- 
    that is, when shf is toward the surface (t-ts > 0). This is "stable". 
    When lst is negative, shf is away from the surface and the boundary layer
    is "unstable".
        
    """
    
    lst=t*np.power(ust,3) / (k * g * (h/(cp*rho)))
    
    return lst
        
@jit(nb.float64(float64,float64,float64,float64,float64,float64,float64),\
     nopython=True)
def SHF(ta,ts,zt,z0_t,ust,rho,corr_h):
    
    """
    Computes the sensible heat flux.     
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)       : air temperaure
        - ts (K)       : surface temperature
        - zt (m)       : measurment height for air temperature
        - z0_h (m)     : roughness length for heat 
        - ust (m/s)    : friction velocity
        - rho (kg/m^3) : air density 
        - corr_h (none): stability correction 
        
    Out:
    
        - shf (W/m^2)  : sensible heat flux
        
    """    
    
    if (ta-ts)==0: shf=0;# print("Skipped --> t equal")
    else: shf = rho * cp * ust * k * (ta-ts) / (np.log(zt/z0_t) - corr_h)

    return shf

@jit(nb.float64(float64,float64,float64,float64,float64,float64,float64,float64),\
     nopython=True)
def LHF(ts,qa,qs,zq,z0_q,ust,rho,corr_q): 
    
    """
    Computes the latent heat flux
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ts (K)            : surface temperature
        - qa (kg/kg)        : air specific humdity (at zq)
        - qs (kg/kg)        : air specific humidity (immediately above surface)
        - p pressure (Pa)   : air pressure
        - zq (m)            : measurment height for humidity
        - z0_q (m)          : roughness length for water vapour
        - ust (m/s)         : friction velocity
        - rho (kg/m^3)      : air density 
        - corr_q (none)     : stability correction 
        
    Out:
    
        - lhf (W/m^2)   : latent heat flux
    """
    
    # The direction of the latent heat flux determines whether latent heat 
    # of evaporation or sublimation is applied. The former is only used when 
    # flux is directed toward the surface (qa > qs) and when ts = 273.15.
    L=Ls
    #print qa,qs,ts
    if qa > qs and ts == 273.15:
        
        L = Le
    if qa-qs==0: lhf=0    
    else: lhf= rho * L * k * ust * (qa-qs) / ( np.log(zq/z0_q) - corr_q )
    
    return lhf

    
@jit(nb.typeof((1.,1.,1.))(float64,float64,float64,float64,float64,float64,\
                           float64,float64,float64,float64,float64),\
                           nopython=True) 
def NEUTRAL(ta,ts,rho,qa,qs,u,p,z0_m,zu,zt,zq):
    
    """
    This is a convenience function that computes the turbulent heat fluxes 
    assuming a netural boundary layer. 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)            : air temperaure
        - ts (K)            : surface temperature
        - rho (kg/m^3)      : air density 
        - qa (kg/kg)        : air specific humdity (at zq)
        - u (m/s)           : wind speed at height zu
        - p (Pa)            : air pressure 
        - z0_m (m)          : roughness length of momentum
        
    Out:
        - shf (W/m^2)   : sensible heat flux
        - lhf (W/m^2)   : latent heat flux
        - qs (kg/kg)    : surface (saturation) specific humidity 
        
    """    
    
    ust = k*u/(np.log(zu/z0_m))
    z0_t, z0_q, re = ROUGHNESS(ust,z0_m)
    shf = SHF(ta,ts,zt,z0_t,ust,rho,0)
    lhf = LHF(ts,qa,qs,zq,z0_q,ust,rho,0) 

    return ust, shf, lhf

@jit(nb.typeof((1.,1.))(float64,float64,float64,float64,float64,float64,\
                           float64,float64,float64,float64,float64),\
                           nopython=True) 
def ITERATE(ta,ts,qa,qs,rho,u,p,z0_m,zu,zq,zt):
    
    """
    This function coordinates the iteration required to solve the circular
    problem of computing MO and shf (which are inter-dependent).
    
    NOTE: if the iteration doesn't converge, the function returns the 
    turbulent heat fluxes under a neutral profile.
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)            : air temperaure
        - ts (K)            : surface temperature
        - qa (kg/kg)        : air specific humdity (at zq)
        - rho (kg/m^3)      : air density
        - u (m/s)           : wind speed (at zu)
        - p pressure (Pa)   : air pressure
        - z0_m (m)          : roughness length of momentum
        
    Out:
    
        - lhf (W/m^2)   : latent heat flux
    """
    
    # Compute the turbulent heat fluxes assuming a neutral profile:
    ust, shf, lhf = NEUTRAL(ta,ts,rho,qa,qs,u,p,z0_m,zu,zt,zq)

    # Iteratively recompute the shf until delta <= maxerr OR niter >= maxiter
    delta=999
    i = 0
    while delta > maxerr and i < maxiter:
        
        if shf == 0: i = maxiter; break
        
        # MO (with old ust)
        lst=MO(ta,ust,shf,rho)
        
        # Ust
        ust,corr_m,corr_h,corr_q = USTAR(u,zu,lst,z0_m)
        
        # Roughness
        z0_h, z0_q, re = ROUGHNESS(ust,z0_m)
        
        # SHF (using the stability corrections returned above)
        shf_new = SHF(ta,ts,zt,z0_h,ust,rho,corr_h)
        
        # Difference?
        delta = np.abs(1.-shf/shf_new)*100.; #print i, delta
        
        # Update old 
        shf = shf_new*1
        
        # Increase i
        i+=1
    
    # Loop exited
    if i >= maxiter: # Use fluxes computed under assumption of neutral profile
        return shf, lhf #, (i, re, ta-ts, corr_m) 
    
    else: # Compute LHF using the last estimate of z0_q and corr_q
        lhf = LHF(ts,qa,qs,zq,z0_q,ust,rho,corr_q)
        
        return shf_new, lhf #, (i, re, ta-ts, corr_m) 
   

#_____________________________________________________________________________#
# Radiative fluxes
#_____________________________________________________________________________#
@jit(float64(float64,float64),nopython=True)
def SW(sin,sout):
    
    """
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - sin (W/m^2)       : incident shorwave flux 
        - sout (W/m^2)      : reflected shorwave flux 
     
    Out:
    
        - swn (W/m^2)       : net shortwave flux absorbed at the surface  
    """
    swn=sin-sout
    return swn

@jit(float64(float64,float64),nopython=True)
def LW(lin,lout):
    
    """
    This function computes the net longwave flux at the (ice) surface.
       
    Inputs/Outputs (units: explanation): 
    
    In:
        - lin (W/m^2)       : incident longwave flux
        - lout (W/m^2)      : incident longwave flux
        
    Out: 
        - lwn (W/m^2)       : net longwave flux at the surface
        
    """    
    
    lwn = lin - lout    
    return lwn

        

#_____________________________________________________________________________#
# Coordinating/summarising functions
#_____________________________________________________________________________# 

@jit(nb.typeof((1.,1.))(float64,float64,float64,float64),nopython=True) 
def MELT_SUB(ts,lhf,seb,ds):
    
    """
    This function computes mass loss via melt and sublimation. 
    
    Notes: If seb is +ve and ts == 273.15: melt = seb/Lf. Otherwise, melt = 0
           If lhf is -ve: sublimation = -lhf/Ls. Else if lhf is +ve 
           and ts < 273.15: sublimation = -lhf/Le
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ts (K)         : surface temperature
        - lhf (W/m^2)    : latent heat flux
        - seb (W/m^2)    : surface energy balance
        - ds (s)         : model time step
     
    Out:
        - melt (mm we)   : melt 
        - sub (mm we)    : sublimation (+ve); resublimation (-ve)
             
    """    
    
    # Melt
    if ts == 273.15:
        melt=np.max(np.array([0,seb/Lf]))*ds
    else: melt = 0
    
    # Sublimation
    if lhf <=0:
        sub=-lhf/Ls * ds 
    else:
        if ts <0: # Resublimation
            sub=-lhf/Ls * ds
        else: sub=-lhf/Le # Condensation

    return melt, sub


@jit(nb.typeof((np.array([1.,]),np.array([1.,]), np.array([1.,]),\
                np.array([1.,]),np.array([1.,]),np.array([1.,]),np.array([1.,])))\
     (float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],\
     float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],\
     float64[:],float64[:],float64[:],float64),nopython=True)                                               
def SEB(ta,ts,qa,qs,rho,u,p,sin,sout,lin,lout,z0_m,zu,zt,zq,ds):
    
    """
    This function computes the surface energy balance
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)         : air temperature at zt 
        - qa (kg/kg)     : specific humidity at zq
        - rho (kg/m^3)   : air density 
        - u (m/s)        : wind speed at zu
        - p pressure (Pa): air pressure
        - sin (W/m^2)    : incident shortwave radiation
        - sout (W/m^2)   : reflected shortwave radiation        
        - lin (W/m^2)    : incident longwave radiation
        - lout (W/m^2)   : emitted longwave radiation
        - z0_m (m)       : roughness length of momentum
        - ds (s)         : time between measurements (s)
        - zu (m)         : rwind sensor height
        - zq (m)         : humidity sensor height
        - zt (m)         : temperature sensor height
                           
    Out:
        - shf (W/m^2)    : surface sensible heat flux
        - lhf (W/m^2)    : surface latent heat flux
        - swn (W/m^2)    : surface shortwave heat flux
        - lwn (W/m^2)    : surface longwave heat flux
        - seb_log (W/m^2): surface energy balance

        
    """

    # Preallocate for output arrays
    nt=len(ta)
    shf_log=np.zeros(nt)*np.nan
    lhf_log=np.zeros(nt)*np.nan
    swn_log=np.zeros(nt)*np.nan
    lwn_log=np.zeros(nt)*np.nan
    seb_log=np.zeros(nt)*np.nan
    melt_log=np.zeros(nt)*np.nan
    sub_log=np.zeros(nt)*np.nan

    
    # Begin iteration
    for i in range(nt):
        # print(i)
        
        # Skip all computations if there any NaNs at this time-step 
        if np.isnan(ta[i]) or np.isnan(sin[i]) or np.isnan(ts[i]) or \
            np.isnan(lin[i]): continue 
        
        
        # Compute the turbulent heat fluxes 
        if u[i] == 0:
            shf = 0
            lhf = 0
        else:
            shf, lhf = ITERATE(ta[i],ts[i],qa[i],qs[i],rho[i],u[i],p[i],\
                                    z0_m[i],zu[i],zq[i],zt[i]) 
    
        # Compute the radiative heat fluxes
        lwn = LW(lin[i],lout[i]); 
        swn=SW(sin[i],sout[i])
        
        # Initial SEB 
        seb=shf+lhf+swn+lwn

        # Compute melt and sublimation here -- melt energy returned by 
        # SEB_WF is different from the SEB (because cc was accounted for)
        melt_log[i],sub_log[i] = MELT_SUB(ts[i],lhf,seb,ds)   
               
                                  
        # Log seb 
        seb_log[i]=seb # same between WF and non-WF
        # Log shf
        shf_log[i]=shf
        # Log lhf
        lhf_log[i]=lhf
        # Log lwn
        lwn_log[i]=lwn
        # Log lhf
        swn_log[i]=swn        
        
        if np.abs(shf) > 1000. or np.abs(lhf)>1000\
            or np.abs(swn)>2000 or np.abs(lwn>1000):
                print("Suspicious values encountered!!")
                print("SHF/LHF/SWN/LWN-->")
                print(shf,lhf,swn,lwn)
                print("T/TS/Q/QS/U/P/RHO/Z0m/ZU/ZT")
                print(ta[i],ts[i],qa[i],qs[i],u[i],p[i],rho[i],z0_m[i],zu[i],\
                      zt[i])
    return shf_log, lhf_log, swn_log, lwn_log, seb_log, melt_log,sub_log



#_____________________________________________________________________________#
# Empirical functions
#_____________________________________________________________________________# 
@jit(float64[:](float64[:],float64,float64,float64),nopython=True)
def tdep(t,ttip,a,b):
    """
    This function computes the stemperature-dependent energy flux (from
    Giesen and Oerlemans = (2012); see Eq. 3 here: 
        https://tc.copernicus.org/articles/6/1463/2012/tc-6-1463-2012.pdf)
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (k or C)      : air temperature (/temp-like array)
        - ttip (k or C)   : temperature beyond which to expect linear f
        - a (W/m**2)      : energy flux at ttip
        - b (W/m**2/C)    :change in energy flux/degree 
    Out:
        - tdep (W/m**2)   : temperature-dependent heat flux
        
    """
    _tdep=np.zeros(len(t))*np.nan
    _tdep[t<ttip]=a
    _tdep[t>=ttip]=a+b*(t[t>=ttip]-ttip)

    return _tdep
    


@jit(float64[:](float64[:],float64,float64,float64,float64[:]),\
     nopython=True)
def eti(t,tf,sf,ttip,swn):
     
    """
    The enanced temperature-index model from Pellicciotti et al. (2005).
    See Eq. 3 here(ignoring backslashes):
    https://www.cambridge.org/core/journals/journal-of-glaciology/\
        article/an-enhanced-temperatureindex-glacier-melt-model-including-\
            the-shortwave-radiation-balance-development-and-testing-for-h\
                aut-glacier-darolla-switzerland/E96A8B8D2903523DE6DBF\
                    88E2E06E6D9
                    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (k or C)              : air temperature (/temp-like array)
        - tf (mm we/t/C)          : temperature melt factor
        - sf (mm we/t/[W/m**2])   : shortwave melt factor
        - ttip (K or C)           : melt threshold (K or C; must be same as t)
        - swn (W/m**2)            : net shortwave radiation
    Out:
        - m (mm/t)                : melt per timestep (t)              
    """

    m=np.zeros(len(t))*np.nan
    m[t<=ttip]=0
    m[t>ttip]=t[t>ttip]*tf+sf*swn[t>ttip]
    
    return m
    

