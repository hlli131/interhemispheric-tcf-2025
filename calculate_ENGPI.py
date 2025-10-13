import numpy as np
import xarray as xr

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


def calculate_ENGPI(av850, rh600, MPI, VWS):
    """
    Calculate Environmental Genesis Potential Index (ENGPI)
    
    Parameters:
    -----------
    data_dir : str
        Directory containing input data files
        
    Returns:
    --------
    engpi : xarray.DataArray
        Environmental Genesis Potential Index
    """
    # Coefficient weights for ENGPI calculation
    coef = {
        'av850': 1.5,   # Absolute vorticity at 850 hPa
        'rh600': 3.0,   # Relative humidity at 600 hPa  
        'MPI': 3.0,     # Maximum Potential Intensity
        'VWS': -2.0,    # Vertical Wind Shear (negative impact)
    }
    
    # Calculate individual terms with normalization
    av850_term = (np.abs(1e5 * av850)) ** (coef['av850'])
    rh600_term = (rh600 / 50) ** (coef['rh600'])
    MPI_term = (MPI / 70) ** (coef['MPI'])
    VWS_term = (1.0 + 0.1 * VWS) ** (coef['VWS'])
    
    # Calculate ENGPI by multiplying all terms
    engpi = av850_term * rh600_term * MPI_term * VWS_term
    
    return engpi


# Load input data
rv850 = xr.open_dataset('rv850.nc')
av850 = calculate_av850(rv850)
rh600 = xr.open_dataset('rh600.nc')
MPI = xr.open_dataset('MPI.nc')
u200 = xr.open_dataset('u200.nc')
u850 = xr.open_dataset('u850.nc')
v200 = xr.open_dataset('v200.nc')
v850 = xr.open_dataset('v850.nc')
VWS = calculate_VWS(u200, u850, v200, v850)


# Calculate ENGPI using the defined function
engpi_result = calculate_ENGPI(rv850, rh600, MPI, VWS)