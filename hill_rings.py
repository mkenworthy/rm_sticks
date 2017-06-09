# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:18:44 2017

@author: demooij
"""

import numpy as np
from scipy import ndimage
from scipy.misc import imrotate
import matplotlib.pyplot as plt
import time
from astropy.io import fits
from numba import jit

@jit
def make_lineprofile(npix,rstar,xc,vgrid,A,veq,linewidth):
    """
    returns the line profile for the different points on the star
    as a 2d array with one axis being velocity and other axis position
    on the star
    npix - number of pixels along one axis of the star (assumes solid bosy rotation)
    rstar - the radius of the star in pixels
    xc - the midpoint of the star in pixels
    vgrid - the velocity grid for the spectrum you wish to make (1d array in km/s)
    A - the line depth of the intrinsic profile - the bottom is at (1 - A) is the max line depth (single value)
    veq - the equatorial velocity (the v sin i for star of inclination i) in km/s (single value)
    linewidth - the sigma of your Gaussian line profile in km/s (single value)
    """
    vc=(np.arange(npix)-xc)/rstar*veq
    vs=vgrid[np.newaxis,:]-vc[:,np.newaxis]
    profile=1.-A*np.exp( -(vs*vs)/2./linewidth**2)
    return profile

@jit
def make_star(npix0,osf,xc,yc,rstar,u1,u2):
    """ 
    makes a circular disk with limb darkening
    returns a 2D map of the limb darkened star
    npix0 - number of pixels along one side of the output square array
    osf - the oversampling factor (integer multiplier)
    xc - x coordinate of the star
    yc - y coordinate of the star
    rstar - radius of the star in pixels
    u1 and u2 - linear and quadratic limb darkening coefficients
    """
    npix=int(np.floor(npix0*osf))
    map=np.mgrid[:npix,:npix].astype('float64')
    map[0]-=xc*osf
    map[1]-=yc*osf
    r2=(map[0]*map[0]+map[1]*map[1])/(rstar*osf)**2
    mu=np.sqrt(1.-r2)
    star=1-u1*(1.-mu)-u2*(1.-mu**2)
    star[np.isnan(star)]=0.
    star=star.reshape((npix0,osf,npix0,osf))
    star=star.mean(axis=3).mean(axis=1)
    return star

@jit
def make_map(npix,npix_star,dRRs,angle):
    """
    makes the stick ring and rotates it to a given angle
    npix - number of pixels in the ring map
    npix_star - number of pixels in the stellar map
    dRRs - the half width of the ring in pixels
    angle - the rotation from the vertical axis of the rings
    output is a 2d square image with ones where the ring is and zeroes outside
    """
    map=np.zeros((npix,npix))
    yc=npix/2
    xc=npix/2
    map[np.round(xc-dRRs).astype(int):np.round(yc+dRRs).astype(int),:]=1.
    rmap=ndimage.interpolation.rotate(map,angle,reshape=False,cval=0.,order=1)

    return rmap


@jit
def overlay_ring(npix_star,b,dr,angle,map0,OSF):
   """
   Makes a stick-like object and overlays on the ring system.
   npix_star - number of pixels in the stellar map
   dRRs - the half width of the ring in pixels
   angle - the rotation from the vertical axis of the rings
   output is a 2d square image with ones where the ring is and zeroes outside
   """
   map=np.zeros((npix_star*OSF,npix_star*OSF))#map0.copy()
   minr=max([0,(b-dr)*OSF])
   maxr=min([(npix_star*OSF)-1,(b+dr)*OSF])
   if (minr < (npix_star*OSF)) & (maxr > 0):
      iminr=int(minr)
      imaxr=int(maxr)
      fl=1.-(minr-iminr)
      fr=maxr-imaxr
      map[:,iminr:imaxr+1]=1.
      map[:,iminr]=fl
      map[:,imaxr]=fr
      #print ("%7.2f  %7.2f %4i  %4i %6.4f %6.4f %10.6f  %10.6f" %(minr,maxr,iminr,imaxr,fl,fr,(float(imaxr)-float(iminr))/npix_star,map[0,:].sum()/npix_star))
      if angle !=0.:
         map=ndimage.interpolation.rotate(map,angle,reshape=False,cval=0.,order=1,prefilter=False)
   map=map.reshape((npix_star,OSF,npix_star,OSF))
   map=map.mean(axis=3).mean(axis=1)
      
   return map


##initialise the star
Rs=510                                                     # radius of the star in pixels
npix_star=1025                                             # number of pixels in the stellar map
OSF=5                                                      # oversampling factor for star to avoid jagged edges
OSF_R=2                                                    # oversampling factor for rings to avoid jagged edges
u1=0.2752                                                  # linear limbdarkening coefficient
u2=0.3790                                                  # quadratic limbdarkening coefficient
xc=512.5                                                   # x-coordinate of disk centre
yc=512.5                                                   # y-coordinate of disk centre
A=0.8                                                      # line depth
veq=130.                                                   # V sin i (km/s)
l_fwhm=20.                                                 # Intrinsic line FWHM (km/s)
lw=l_fwhm/2.35                                             # Intinsice line width in sigma (km/s)
vgrid=np.arange(-160,160.1,0.1)                            # velocity grid

profile=make_lineprofile(npix_star,Rs,xc,vgrid,A,veq,lw)   # make line profile for each vertical slice of the star
star=make_star(npix_star,OSF,xc,yc,Rs,u1,u2)               # make a limb-darkened stellar disk
sflux=star.sum(axis=1)                                     # sum up the stellar disk across the x-axis
improf=np.sum(sflux[:,np.newaxis]*profile,axis=0)                # calculate the spectrum for an unocculted star
normalisation=improf[0]
improf=improf/normalisation

R_ring=np.asarray([0.01,0.025,0.05,0.10,0.25,0.5,0.75,1.00])
lambda_star=np.arange(-45.,90.1,45.)
opacity=np.asarray([0.25,0.5,0.75,1.0])
pos=np.arange(-1,1.01,0.01)

hdu = fits.PrimaryHDU(pos)
hdu.writeto("new_sim/positions.fits",overwrite=True)
hdu = fits.PrimaryHDU(lambda_star)
hdu.writeto("new_sim/lambda_star.fits",overwrite=True)
hdu = fits.PrimaryHDU(vgrid)
hdu.writeto("new_sim/vgrid.fits",overwrite=True)
hdu = fits.PrimaryHDU(opacity)
hdu.writeto("new_sim/opacity.fits",overwrite=True)
hdu = fits.PrimaryHDU(improf)
hdu.writeto("new_sim/reference_profile.fits",overwrite=True)

map0=np.zeros((npix_star,npix_star))
for dR in R_ring:
##initialise ring parameters
    print ("Starting on ring HWHM: %4.2f R_star"% dR)
    dRRs=dR*Rs
    npix=4096
    xc_r=npix/2
    
    line_profile=np.zeros( (pos.shape[0],lambda_star.shape[0],opacity.shape[0],vgrid.shape[0]) )
    
    for i,lam in enumerate(lambda_star):
        print i,lam
        for k,pr in enumerate(pos):
            tmap=overlay_ring(npix_star,np.round(pr*Rs+xc),dRRs,lam,map0,OSF)
            for j,et in enumerate(opacity):
                map=star*(1.-tmap*et)
                sflux=map.sum(axis=0)
                line_profile[k,i,j,:]=np.sum(sflux[:,np.newaxis]*profile,axis=0)/normalisation
         
    hdu = fits.PrimaryHDU(line_profile)
    hdu.writeto(str("new_sim/simulation_RR_%4.2f.fits" % (2*dR)),overwrite=True)
