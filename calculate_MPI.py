import tcpyPI
import xarray as xr

# Read required datasets
SST = xr.open_dataset('SST.nc')
MSLP = xr.open_dataset('MSLP.nc')
T = xr.open_dataset('T.nc')
SH = xr.open_dataset('SH.nc')

# Merge all datasets into a single dataset
CalcMPI = xr.merge([SST, MSLP, T, SH])

def calc_MPI(ds):
    """
    Calculate Maximum Potential Intensity (MPI) using tcpyPI package
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing SST, MSLP, air temperature profile, and specific humidity profile
        
    Returns:
    --------
    xarray.DataArray
        MPI values
    """
    
    # Apply the tcpyPI.pi function to calculate MPI
    res = xr.apply_ufunc(
        tcpyPI.pi,    # Function to calculate MPI
        ds.sst,       # Sea surface temperature
        ds.mslp,      # Mean sea level pressure  
        ds.pressure,  # Pressure levels
        ds.t,         # Air temperature profile
        ds.q,         # Specific humidity profile
        kwargs={
            'CKCD': 0.9,       # Exchange coefficients ratio
            'ascent_flag': 0,  # Ascent flag
            'diss_flag': 1,    # Dissipation flag
            'ptop': 50,        # Top pressure level (hPa)
            'miss_handle': 1   # Missing value handling
        },
        input_core_dims=[
            [],             # sst
            [],             # mslp
            ['pressure'],   # pressure: pressure dimension
            ['pressure'],   # t: pressure dimension
            ['pressure'],   # q: pressure dimension
        ],
        output_core_dims=[
            [],  # MPI
            [],  # Pmin
            [],  # IFL
            [],  # TO
            [],  # OTL
        ],
        vectorize=True,               # Vectorize the operation
        keep_attrs='drop_conflicts',  # Handle attribute conflicts
    )

    # Unpack results: MPI, minimum pressure, inflow layer, outflow temperature, outflow temperature level
    MPI, Pmin, IFL, TO, OTL = res
    
    # Create result dataset with all calculated variables
    result = xr.Dataset(
        data_vars={
            'mpi': MPI,   # Maximum Potential Intensity
            'pmin': Pmin, # Minimum central pressure
            'ifl': IFL,   # Inflow layer properties
            'to': TO,     # Outflow temperature
            'olt': OTL,   # Outflow temperature level
        },
    )
    
    # Return the MPI values
    return result.mpi

# Calculate MPI using the defined function
mpi_result = calc_MPI(CalcMPI)