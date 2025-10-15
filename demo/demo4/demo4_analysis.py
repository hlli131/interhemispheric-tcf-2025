# =================================================
# demo4: physical_mechanisms_explanation
# =================================================

import cmaps
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['font.size'] = 18


# Load data
sat_JJASON = xr.open_dataset('sat_JJASON.nc').t2m
u850_JJASON = xr.open_dataset('u850_JJASON.nc').u850
v850_JJASON = xr.open_dataset('v850_JJASON.nc').v850
sat_DJFMAM = xr.open_dataset('sat_DJFMAM.nc').t2m
u850_DJFMAM = xr.open_dataset('u850_DJFMAM.nc').u850
v850_DJFMAM = xr.open_dataset('v850_DJFMAM.nc').v850
vws_JJASON = xr.open_dataset('vws_JJASON.nc').vws
u200_JJASON = xr.open_dataset('u200_JJASON.nc').u200
v200_JJASON = xr.open_dataset('v200_JJASON.nc').v200
vws_DJFMAM = xr.open_dataset('vws_DJFMAM.nc').vws
u200_DJFMAM = xr.open_dataset('u200_DJFMAM.nc').u200
v200_DJFMAM = xr.open_dataset('v200_DJFMAM.nc').v200
mean_w_JJASON = xr.open_dataset('mean_w_JJASON.nc').w
lat_JJASON = np.load('lat_JJASON.npy')
p_JJASON = np.load('p_JJASON.npy')
v_JJASON = np.load('v_JJASON.npy')
w_JJASON = np.load('w_JJASON.npy')
v_sig_JJASON = np.load('v_sig_JJASON.npy')
w_sig_JJASON = np.load('w_sig_JJASON.npy')
mean_w_DJFMAM = xr.open_dataset('mean_w_DJFMAM.nc').w
lat_DJFMAM = np.load('lat_DJFMAM.npy')
p_DJFMAM = np.load('p_DJFMAM.npy')
v_DJFMAM = np.load('v_DJFMAM.npy')
w_DJFMAM = np.load('w_DJFMAM.npy')
v_sig_DJFMAM = np.load('v_sig_DJFMAM.npy')
w_sig_DJFMAM = np.load('w_sig_DJFMAM.npy')


# Define MDRs (Major Development Regions) for tropical cyclones
MDRs = [
    {'name': 'WNP', 'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
    {'name': 'ENP', 'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
    {'name': 'NA', 'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
    {'name': 'SI', 'lon_min': 55, 'lon_max': 105, 'lat_min': -15, 'lat_max': -5},
    {'name': 'SP', 'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
    {'name': 'NI', 'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20},  
]


# Create figure
fig = plt.figure(figsize=(20, 13), dpi=500)

# Common settings for map plots
def setup_map_plot(ax):
    """Set up common map elements"""
    ax.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
    ax.add_feature(cfeature.LAND, fc='w')
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-50, 51, 25))
    ax.set_xticks(np.arange(-180, 181, 20), minor=True)
    ax.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
    ax.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
    ax.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
    ax.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
    ax.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
    ax.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
    ax.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5) 
        spine.set_color('k')
    return ax

def add_quiver(ax, u_data, v_data, scale, key_value, key_label):
    """Add wind quiver plot with key"""
    skip = 3
    lon2d, lat2d = np.meshgrid(u_data.lon, u_data.lat)
    u_skip = u_data.values[::skip, ::skip]
    v_skip = v_data.values[::skip, ::skip]
    lon_skip = lon2d[::skip, ::skip]
    lat_skip = lat2d[::skip, ::skip]
    q = ax.quiver(lon_skip, lat_skip, u_skip, v_skip, scale=scale, scale_units='inches',
                  width=0.0015, headwidth=5, headlength=7, pivot='middle', transform=ccrs.PlateCarree())
    qk = ax.quiverkey(q, 0.9, 1.05, key_value, key_label, labelpos='E', coordinates='axes', fontproperties={'size': 15})
    return q

def setup_colorbar(cbar):
    """Set up colorbar with consistent styling"""
    cbar.ax.tick_params(axis='x', which='both', length=0, width=0)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.5)

# Plot 1: SAT & 850-hPa wind (JJASON)
ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree(central_longitude=180))
setup_map_plot(ax1)
cf1 = ax1.contourf(sat_JJASON.lon, sat_JJASON.lat, sat_JJASON, levels=np.linspace(-1.5, 1.5, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.BlueWhiteOrangeRed, extend='both')
cbar1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.15, aspect=50)
cbar1.ax.set_xlabel(r'$\mathrm{\Delta SAT\ (\degree C)}$', fontsize=18, labelpad=5)
cbar1.ax.set_xticks(np.arange(-1.5, 1.8, 0.3))
setup_colorbar(cbar1)
add_quiver(ax1, u850_JJASON, v850_JJASON, 3, 1, r'$\mathrm{1\ m\ s^{-1}}$')
ax1.set_title(r'$\mathbf{a}$', fontsize=22, loc='left')
ax1.set_title('SAT & 850-hPa wind (JJASON)', fontsize=22)

# Plot 2: SAT & 850-hPa wind (DJFMAM)
ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
setup_map_plot(ax2)
cf2 = ax2.contourf(sat_DJFMAM.lon, sat_DJFMAM.lat, sat_DJFMAM, levels=np.linspace(-1.5, 1.5, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.BlueWhiteOrangeRed, extend='both')
cbar2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.15, aspect=50)
cbar2.ax.set_xlabel(r'$\mathrm{\Delta SAT\ (\degree C)}$', fontsize=18, labelpad=5)
cbar2.ax.set_xticks(np.arange(-1.5, 1.8, 0.3))
setup_colorbar(cbar2)
add_quiver(ax2, u850_DJFMAM, v850_DJFMAM, 3, 1, r'$\mathrm{1\ m\ s^{-1}}$')
ax2.set_title(r'$\mathbf{d}$', fontsize=22, loc='left')
ax2.set_title('SAT & 850-hPa wind (DJFMAM)', fontsize=22)

# Plot 3: VWS & 200-hPa wind (JJASON)
ax3 = fig.add_subplot(3, 2, 3, projection=ccrs.PlateCarree(central_longitude=180))
setup_map_plot(ax3)
cf3 = ax3.contourf(vws_JJASON.lon, vws_JJASON.lat, vws_JJASON, levels=np.linspace(-2, 2, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.MPL_RdBu_r, extend='both')
cbar3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', pad=0.15, aspect=50)
cbar3.ax.set_xlabel(r'$\mathrm{\Delta VWS\ (m\ s^{-1})}$', fontsize=18, labelpad=5)
cbar3.ax.set_xticks(np.arange(-2, 2.1, 0.4))
setup_colorbar(cbar3)
add_quiver(ax3, u200_JJASON, v200_JJASON, 6, 2, r'$\mathrm{2\ m\ s^{-1}}$')
ax3.set_title(r'$\mathbf{b}$', fontsize=22, loc='left')
ax3.set_title('VWS & 200-hPa wind (JJASON)', fontsize=22)

# Plot 4: VWS & 200-hPa wind (DJFMAM)
ax4 = fig.add_subplot(3, 2, 4, projection=ccrs.PlateCarree(central_longitude=180))
setup_map_plot(ax4)
cf4 = ax4.contourf(vws_DJFMAM.lon, vws_DJFMAM.lat, vws_DJFMAM, levels=np.linspace(-2, 2, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.MPL_RdBu_r, extend='both')
cbar4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', pad=0.15, aspect=50)
cbar4.ax.set_xlabel(r'$\mathrm{\Delta VWS\ (m\ s^{-1})}$', fontsize=18, labelpad=5)
cbar4.ax.set_xticks(np.arange(-2, 2.1, 0.4))
setup_colorbar(cbar4)
add_quiver(ax4, u200_DJFMAM, v200_DJFMAM, 6, 2, r'$\mathrm{2\ m\ s^{-1}}$')
ax4.set_title(r'$\mathbf{e}$', fontsize=22, loc='left')
ax4.set_title('VWS & 200-hPa wind (DJFMAM)', fontsize=22)

# Common settings for cross-section plots
def setup_cross_section(ax):
    """Set up common cross-section plot elements"""
    ax.set_xlim([-30, 30])
    ax.set_xticklabels(['30°S', '20°S', '10°S', '0°', '10°N', '20°N', '30°N'])
    ax.set_ylim([1000, 105])
    ax.set_yscale('log')
    ax.set_yticks([1000, 700, 500, 300, 200, 105])
    ax.set_yticklabels([1000, 700, 500, 300, 200, 100])
    ax.minorticks_off()
    ax.set_ylabel('Pressure (hPa)', fontsize=20)
    ax.set_xticks(np.arange(-30, 31, 5), minor=True)
    ax.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
    ax.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
    ax.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
    ax.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5) 
        spine.set_color('k')
    return ax

# Create custom colormap with white center
def create_white_cmap():
    """Create colormap with white center for vertical velocity"""
    colors = list(cmaps.NCV_blue_red(np.linspace(0, 1, 20)))
    colors[9:11] = [(1, 1, 1, 1), (1, 1, 1, 1)]
    return ListedColormap(colors)

# Plot 5: Hadley circulation (JJASON)
ax5 = fig.add_subplot(3, 2, 5)
setup_cross_section(ax5)
cmap_white = create_white_cmap()
cf5 = ax5.contourf(mean_w_JJASON.lat, mean_w_JJASON.pressure, mean_w_JJASON*864,
                   levels=np.linspace(-40, 40, 21), cmap=cmap_white, extend='both')
cbar5 = fig.colorbar(cf5, ax=ax5, orientation='horizontal', pad=0.15, aspect=50)                
cbar5.ax.set_xlabel(r'$\mathrm{Vertical\ pressure\ velocity\ (hPa\ d^{-1})}$', fontsize=18, labelpad=5)
cbar5.ax.set_xticks(np.arange(-40, 42, 8))
cbar5.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar5.ax.spines.values():
    spine.set_linewidth(1.5)
q5 = ax5.quiver(lat_JJASON, p_JJASON, v_JJASON, w_JJASON, scale=0.8, scale_units='inches', color='k',
                width=0.002, headwidth=5, headlength=7, pivot='middle')
q5_sig = ax5.quiver(lat_JJASON, p_JJASON, v_sig_JJASON, w_sig_JJASON, scale=0.8, scale_units='inches', color='g',
                    width=0.002, headwidth=5, headlength=7, pivot='middle')
qk5 = ax5.quiverkey(q5, 0.95, 1.05, 0.5, r'$0.3$', labelpos='E', coordinates='axes', fontproperties={'size': 15})
ax5.set_title(r'$\mathbf{c}$', fontsize=22, loc='left')
ax5.set_title('Hadley circulation (JJASON)', fontsize=22)
rect = patches.Rectangle((5, 106), 15, 885, ls='-', lw=2.5, ec='purple', fc='none', zorder=10)
ax5.add_patch(rect)

# Plot 6: Hadley circulation (DJFMAM)
ax6 = fig.add_subplot(3, 2, 6)
setup_cross_section(ax6)
cmap_white = create_white_cmap()
cf6 = ax6.contourf(mean_w_DJFMAM.lat, mean_w_DJFMAM.pressure, mean_w_DJFMAM*864, 
                   levels=np.linspace(-40, 40, 21), cmap=cmap_white, extend='both')
cbar6 = fig.colorbar(cf6, ax=ax6, orientation='horizontal', pad=0.15, aspect=50)                
cbar6.ax.set_xlabel(r'$\mathrm{Vertical\ pressure\ velocity\ (hPa\ d^{-1})}$', fontsize=18, labelpad=5)
cbar6.ax.set_xticks(np.arange(-40, 42, 8))
cbar6.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar6.ax.spines.values():
    spine.set_linewidth(1.5)
q6 = ax6.quiver(lat_DJFMAM, p_DJFMAM, v_DJFMAM, w_DJFMAM, scale=0.8, scale_units='inches', color='k',
                width=0.002, headwidth=5, headlength=7, pivot='middle')
q6_sig = ax6.quiver(lat_DJFMAM, p_DJFMAM, v_sig_DJFMAM, w_sig_DJFMAM, scale=0.8, scale_units='inches', color='g',
                    width=0.002, headwidth=5, headlength=7, pivot='middle')
qk6 = ax6.quiverkey(q6, 0.95, 1.05, 0.5, r'$0.3$', labelpos='E', coordinates='axes', fontproperties={'size': 15})    
ax6.set_title(r'$\mathbf{f}$', fontsize=22, loc='left')
ax6.set_title('Hadley circulation (DJFMAM)', fontsize=22)
rect = patches.Rectangle((-15, 106), 10, 885, ls='-', lw=2.5, ec='purple', fc='none', zorder=10)
ax6.add_patch(rect)

# Add MDR rectangles to maps
for ax in (ax1, ax3):
    for MDR in MDRs:
        if MDR['lat_min'] >= 0:              
            rect = patches.Rectangle(
                (MDR['lon_min'], MDR['lat_min']),
                 MDR['lon_max'] - MDR['lon_min'],
                 MDR['lat_max'] - MDR['lat_min'],
                 ls='-', lw=2.5, ec='r', fc='none', transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)
for ax in (ax2, ax4):
    for MDR in MDRs:
        if MDR['lat_max'] <= 0:               
            rect = patches.Rectangle(
                (MDR['lon_min'], MDR['lat_min']),
                 MDR['lon_max'] - MDR['lon_min'],
                 MDR['lat_max'] - MDR['lat_min'],
                 ls='-', lw=2.5, ec='r', fc='none', transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)

plt.tight_layout()
# plt.savefig('FigD4.png', bbox_inches='tight', dpi=300)
plt.show()