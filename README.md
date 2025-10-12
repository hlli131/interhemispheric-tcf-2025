# Interhemispheric tropical cyclone frequency trends
![Status](https://img.shields.io/badge/status-Under_Review-yellow)
![Version](https://img.shields.io/badge/version-2025.10.12-red)
![Language](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)

> For peer review only. The final version will be updated upon acceptance.


## 📖 Brief introduction
This repository includes the following directories:
- *`observed_interhemispheric_contrast`*
- *`key_factors_identification`*
- *`detection_and_attribution_analysis`*
- *`physical_mechanisms_explanation`*
- *`demo`*

| Directory name | Description |
| ---------- | ---------- |
| *observed_interhemispheric_contrast* | Analyze and plot the contrast in TCF trends between hemispheres (**Fig. 1**) |
| *key_factors_identification* | Identify key factors and quantify their contributions and interactions applying IML (**Fig. 2**) |
| *detection_and_attribution_analysis* | Detect and attribute TCF to human fingerprints using OFM and CMIP6 simulations (**Fig. 3**) |
| *physical_mechanisms_explanation* | Explain the physical mechanisms through coupled thermodynamic and dynamic pathways (**Fig. 4**) |
| *demo* | Test the above codes on a small dataset and output the corresponding results (**Figs. D1–D4**)|


## 🖥️ Configuration (desktop)
- **Platform**: Windows Subsystem for Linux (WSL)  
- **Dependencies**:
  ```
  # Python==3.11
  numpy==1.26.4
  scipy==1.14.0
  pandas==2.2.3
  xarray==2025.4.0
  netCDF4==1.7.2
  matplotlib==3.10.0
  cartopy==0.24.1
  cmaps==2.0.1
  scikit-learn==1.6.1
  xgboost==3.0.1
  lightgbm==4.6.0
  statsmodels==0.14.4
  pymannkendall==1.4.3
  metpy==1.7.0
  xesmf==0.8.7 (not recommended on Windows)
  ```
- **Hardware**:
  ```
  Memory: 16 GB
  CPU: Intel(R) Core(TM) i5-14500 (14 cores, 20 threads)
  GPU: NVIDIA GeForce GTX 750 Ti (dedicated) & Intel(R) UHD Graphics 770 (integrated)
  ```

## ⚙️ Installation
All required packages can be installed via `conda` (from [**Conda-forge**](https://conda-forge.org)) or `pip` (from [**PyPI**](https://pypi.org)) using the following commands:
```
# Using conda (recommended) ✅
conda install <package_name> -c conda-forge

# Using pip ✅
pip install <package_name>
```
**⏱️ Expected installation time**: Typically completes **within 5 minutes**, depending on the network speed and system configuration.


## 🚀 Demo
The *`demo`* directory provides lightweight analysis using small sample datasets, which allows users to quickly verify whether all components are functioning properly before conducting full analysis. Each demo is **completely standalone** and includes the **analysis codes**, **sample data** and **corresponding output**:
```
demo/
├── demo1/
│   ├── demo1_analysis.ipynb
│   ├── demo1_data/
│   └── FigD1.png
├── demo2/
│   ├── demo2_analysis.ipynb
│   ├── demo2_data/
│   └── FigD2.png
├── demo3/
│   ├── demo3_analysis.ipynb
│   ├── demo3_data/
│   └── FigD3.png
└── demo4/
    ├── demo4_analysis.ipynb
    ├── demo4_data/
    └── FigD4.png
```
**⚠️ Please note**: These outputs (e.g., Figs. D1–D4) are generated from small sample datasets for validation purposes only and are expected to differ from the main results (Figs. 1–4).


## 📊 Testing


## 📦 Data availability
Original datasets for full analysis are publicly available from the following sources:

### TC observations
- **International Best Track Archive for Climate Stewardship (IBTrACS)**  
  Source: https://www.ncei.noaa.gov/products/international-best-track-archive  

### Atmospheric reanalysis
- **ECMWF Fifth-Generation Reanalysis (ERA5)**  
  Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means  

### Oceanic datasets
- **ECMWF Ocean Reanalysis System 5 (ORAS5)**  
  Source: https://cds.climate.copernicus.eu/datasets/reanalysis-oras5  
- **Hadley Centre Sea Ice and Sea Surface Temperature (HadISST)**  
  Source: https://www.metoffice.gov.uk/hadobs/hadisst  
- **Extended Reconstructed Sea Surface Temperature version 6 (ERSSTv6)**  
  Source: https://www.ncei.noaa.gov/products/extended-reconstructed-sst  

### Multimodel simulations
- **Coupled Model Intercomparison Project Phase 6 (CMIP6)**  
  Source: https://pcmdi.llnl.gov/CMIP6  


## 📄 Licence
> This repository is open source under the **MIT License**.
