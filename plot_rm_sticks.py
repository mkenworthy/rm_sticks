# filenames are  the width of the ring in stellar radii
# position, angle_of_star, opacity, wavelength
# position - center of the stick relative to the center of the star in
# units of star radius
# angle_of_star - angle between rotation axis of star and ring long axis
# in indices - look it up in another array contianing the degrees
# opacity of the ring - 1 is black, 0 is transparent, 0.25, 0.5, 0.75,
# 1.0
# velocity - another fits file contains the vgrid velocities.

# 201, 4, 4, 3201

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

stick_file = 'new_sim/simulation_RR_0.10.fits'

lineprof = fits.getdata(stick_file, header=False)

print(lineprof.shape)

pos = fits.getdata("new_sim/positions.fits")
lambda_star = fits.getdata("new_sim/lambda_star.fits")
vgrid = fits.getdata("new_sim/vgrid.fits")
opacity = fits.getdata("new_sim/opacity.fits")
improf = fits.getdata("new_sim/reference_profile.fits")

print(pos.shape) # 201
print(lambda_star.shape) # 4
print('angles of star versus stick: {}'.format(lambda_star))
print(vgrid.shape) # 3201
print(opacity.shape) # 4
print('opacity: {}'.format(opacity))
print(improf.shape) # 3201

# okay, take a loop over pos
for npos in np.arange(0, 201, 20):
    plt.plot(vgrid, improf,color='k', linewidth=5)
    lincol = '{:.2f}'.format(npos/201.)
    plt.plot(vgrid, lineprof[npos, 2, 3, :], color=lincol)
#plt.show()

# each vertical panel has the star and stick, then the absolute line
# profile, then the differential line profile. x axis is the same in all
# of them.


import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8,12))

lambdas = np.array((1,2,3)) # the index of the lambda angles that we want to plot

vel_range = 200

# lineprof.shape = 201, 4, 4, 3201

# subtract off improf off all of them... 3201
lineprofsub = lineprof - improf
# outside of the line profile (at the highest and lowest velocity)
# that's the flux of the star.
# subtract that off to remove the flux decrement
# we want the value at (:,:,:,0) and subtract that all off
lineprofsubsub = lineprofsub - lineprofsub[:,:,:,0:1]

gs_lambdas = gridspec.GridSpec(1, lambdas.size)

# make the coordinates for a stick of width w, then offset by d, then
# rotate by alpha

#the function from https://gis.stackexchange.com/questions/23587/how-do-i-rotate-the-polygon-about-an-anchor-point-using-python-script
def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts-cnt,np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]))+cnt

# make the coordinates of a stick for plotting
def make_stick(off=0.5, w=0.2, ang=45.):
    pts = np.array([[-w/2,-5],[-w/2,5],[w/2,5],[w/2,-5]])
    ang_rad = ang *np.pi / 180.
    return(Rotate2D(pts+off,np.array([0,0]), ang_rad))

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

for i, (lam) in enumerate(lambdas):
    # make three vertical panels for each lambda angle
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1,
            subplot_spec=gs_lambdas[i], wspace=0.0, hspace=0.0)
    ax_star = plt.Subplot(fig, inner_grid[0])
    ax_line = plt.Subplot(fig, inner_grid[1])
    ax_diff = plt.Subplot(fig, inner_grid[2])

    ax_star.set_title('ang = {}'.format(lambda_star[lam]))

    # make the stellar disk
    circle=plt.Circle((0,0),1.0, color='r')

    patches = []
    patches.append(circle)

    # fix the axes to be square and remove the Axis Labels
    ax_star.set_xlim([-1.1,1.1])
    ax_star.set_ylim([-1.1,1.1])
    ax_star.set_aspect('equal')
    ax_star.axes.get_xaxis().set_visible(False)
    ax_star.axes.get_yaxis().set_visible(False)

    # plot the direct line profiles
    for npos in np.arange(0, 201, 20):
        ax_line.plot(vgrid, improf,color='k', linewidth=5)
        lincol = '{:.2f}'.format(npos/201.)
        ax_line.plot(vgrid, lineprof[npos, lam, 3, :], color=lincol)

        # plot the differential line profiles
        lincol = '{:.2f}'.format(npos/201.)
        ax_diff.plot(vgrid, lineprofsubsub[npos, lam, 3, :], color=lincol)


        # generate the relevant stick
        width = 0.10
        offset = pos[npos]
        la = lambda_star[lam]
        stick_coord = make_stick(offset, width, la)
        stick_patch = Polygon(stick_coord, closed=True, color='blue')
        patches.append(stick_patch)

    p = PatchCollection(patches, alpha=0.4)
    ax_star.add_collection(p)

#    ax_star.add_artist(circle)
    ax_line.set_xlim([-vel_range, vel_range])
    ax_line.set_ylim([0.75, 1.05])

    ax_diff.set_xlim([-vel_range, vel_range])
    ax_diff.set_ylim([-0.001, 0.05])

    # add the figures to the plot
    fig.add_subplot(ax_star)
    fig.add_subplot(ax_line)
    fig.add_subplot(ax_diff)

plt.show()



