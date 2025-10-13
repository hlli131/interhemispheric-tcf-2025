import numpy as np
import xarray as xr

def calculate_VWS(u200, u850, v200, v850):
    """
    Calculate vertical wind shear magnitude between 850 and 200 hPa
    
    Parameters:
    -----------
    u200, u850 : xarray.DataArray
        Zonal wind components at 200 hPa and 850 hPa (m/s)
    v200, v850 : xarray.DataArray
        Meridional wind components at 200 hPa and 850 hPa (m/s)
        
    Returns:
    --------
    VWS : xarray.DataArray
        Vertical wind shear magnitude (m/s)
    """
    # Calculate wind differences between 200 hPa and 850 hPa
    delta_u = u200 - u850
    delta_v = v200 - v850
    
    # Calculate wind shear magnitude
    VWS = np.sqrt(delta_u ** 2 + delta_v ** 2)
    VWS.name = 'VWS'
    VWS.attrs['long_name'] = 'vertical wind shear between 850 and 200 hPa'
    VWS.attrs['units'] = 'm/s'
    VWS = VWS.astype(np.float32)
    
    return VWS


def calculate_dudy500(u500):
    """
    Calculate meridional gradient of zonal wind at 500 hPa
    
    Parameters:
    -----------
    u500 : xarray.DataArray
        Zonal wind component at 500 (m/s)
        
    Returns:
    --------
    dudy500 : xarray.DataArray
        Meridional gradient of zonal wind at 500 hPa (1/s)
    """
    # Calculate meridional gradient of zonal wind at 500 hPa
    dudy500 = u500.differentiate(coord='lat', edge_order=1)
    dudy500 = dudy500.where(dudy500.lat >= 0, -dudy500)      # reverse the SH
    dudy500 /= 111000                                        # convert [m/s/degree] to [1/s]
    dudy500.name = 'dudy500'
    dudy500.attrs['long_name'] = 'meridional gradient of zonal wind at 500 hPa'
    dudy500.attrs['units'] = '1/s'
    dudy500 = dudy500.astype(np.float32)
    
    return dudy500


def calculate_av850(rv850):
    """
    Calculate absolute vorticity at 850 hPa
    
    Parameters:
    -----------
    rv850 : xarray.DataArray
        Relative vorticity at 850 hPa
        
    Returns:
    --------
    av850 : xarray.DataArray
        Absolute vorticity at 850 hPa (1/s)
    """
    # Earth's angular velocity (rad/s)
    omega = 7.2921e-5 
    
    # Calculate Coriolis parameter
    f = 2 * omega * np.sin(np.deg2rad(rv850.lat)).rename('f')
    
    # Expand dimensions to match relative vorticity data
    f = f.expand_dims({'time': rv850.time.size, 'lon': rv850.lon.size}, axis=[0, 2])
    f = f.assign_coords(time=rv850.time, lon=rv850.lon)
    f.name = 'f'
    f.attrs['long_name'] = 'Coriolis parameter'
    f.attrs['units'] = '1/s'
    f = f.astype(np.float32)
    
    # Calculate absolute vorticity (f + relative vorticity)
    av850 = f + rv850
    av850.name = 'av850'
    av850.attrs['long_name'] = 'absolute vorticity at 850 hPa'
    av850.attrs['units'] = '1/s'
    av850 = av850.astype(np.float32)
    
    return av850


def calculate_DGPI(VWS, dudy500, w500, av850):
    """
    Calculate the Dynamic Genesis Potential Index (DGPI)
    
    Parameters:
    -----------
    data_dir : str
        Directory containing input data files
        
    Returns:
    --------
    engpi : xarray.DataArray
        The DGPI values
    """
    # Coefficient weights for DGPI calculation
    coef = {
    'VWS'    :-1.7,   # Vertical Wind Shear (negative impact)
    'dudy500': 2.3,   # Meridional gradient of zonal wind at 500 hPa
    'w500'   : 3.4,   # Vertical pressure velocity at 500 hPa
    'av850'  : 2.4,   # Absolute vorticity at 850 hPa
}
    
    # Calculate individual terms with normalization
    VWS_term = (2.0 + 0.1 * VWS) ** (coef['VWS'])
    dudy500_term = (5.5 - 1e5 * dudy500) ** (coef['dudy500'])
    w500_term = (5.0 - 20 * w500) ** (coef['w500'])
    av850_term = (5.5 + np.abs(1e5 * av850)) ** (coef['av850'])
    
    # Calculate DGPI by multiplying all terms
    DGPI = VWS_term * dudy500_term * w500_term * av850_term * np.exp(-11.8) - 1.0
    
    return dgpi


# Load input data
u200 = xr.open_dataset('u200.nc')
u850 = xr.open_dataset('u850.nc')
v200 = xr.open_dataset('v200.nc')
v850 = xr.open_dataset('v850.nc')
VWS = calculate_VWS(u200, u850, v200, v850)
u500 = xr.open_dataset('u500.nc')
dudy500 = calculate_dudy500(u500)
w500 = xr.open_dataset('w500.nc')
rv850 = xr.open_dataset('rv850.nc')
av850 = calculate_av850(rv850)


# Calculate DGPI using the defined function
dgpi_result = calculate_DGPI(VWS, dudy500, w500, av850)