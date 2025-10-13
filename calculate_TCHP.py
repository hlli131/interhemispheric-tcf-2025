import numpy as np
import xarray as xr

# Read ocean temperature (OT) and 26°C isotherm depth (D26) data
OT = xr.open_dataset('OT.nc')
D26 = xr.open_dataset('D26.nc')

def calc_TCHP(ot, d26):
    """
    Calculate Tropical Cyclone Heat Potential (TCHP) from ocean temperature data.
    
    TCHP represents the heat energy available in the ocean upper layers that can 
    potentially fuel tropical cyclone intensification. It integrates the heat content 
    from the surface down to the depth of the 26°C isotherm.
    
    Parameters:
    -----------
    ot : xarray.DataArray
        Ocean temperature data with 'depth' dimension [°C]
    d26 : xarray.DataArray  
        Depth of 26°C isotherm [m]
    
    Returns:
    --------
    tchp : xarray.DataArray
        Tropical Cyclone Heat Potential [kJ/cm²]
    """
    
    # Physical constants
    rho = 1026   # Seawater density [kg/m³]
    Cp = 4178    # Specific heat capacity of seawater [J/kg/°C]
    
    # Calculate temperature excess above 26°C (set negative values to 0)
    dt = (ot - 26).where(ot > 26, 0)
    
    # Calculate vertical layer thicknesses using depth gradient
    dz = xr.DataArray(np.gradient(ot.depth), dims=['depth'], coords={'depth': ot.depth})
    
    # Create mask for layers above the 26°C isotherm depth
    depth_mask = ot.depth <= d26.expand_dims(dim={'depth': ot.depth}, axis=1)
    
    # Calculate TCHP by integrating heat content through depth
    # 1e-7 converts from [J/m²] to [kJ/cm²]
    tchp = (dt * dz).where(depth_mask, 0).sum(dim='depth') * rho * Cp * 1e-7
    
    return tchp