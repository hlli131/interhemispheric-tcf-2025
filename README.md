# Interhemispheric tropical cyclone frequency trends
![Status](https://img.shields.io/badge/status-Under_Review-yellow)
![Version](https://img.shields.io/badge/version-2025.10.12-red)
![Language](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)

> For peer review only. The final version will be updated upon acceptance.


## 📖 Brief introduction
This repository includes the following folders:
- *observed_interhemispheric_contrast*
- *key_factors_identification*
- *detection_and_attribution_analysis*
- *physical_mechanisms_explanation*
- *demo*

| Folder name | Description |
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


## 🛠️ Environmental  settings

## 🔧 Dependencies


## 🧪 Test


## 📊 Evaluation


## 📦

### Using Python

```
from your_package import main_module

# 
instance = main_module.YourClass()

# 
result = instance.process_data("your input")
print(result)
```

## 📘 Instructions


## 📄 Licence
> This repository is open source under the **MIT License**.
