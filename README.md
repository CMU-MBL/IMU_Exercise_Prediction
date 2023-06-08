# Seven Things to Know about Exercise Monitoring with Inertial Sensing Wearables

## Overview

This repository presents the implementation of our collaborative project between UPenn/Orthopaedics, UDel/Physical Therapy, and CMU/Engineering on exercise prediction. The manuscript reporting outcomes of this project is currently under review at the *IEEE Journal of Biomedical and Health Informatics (JBHI)*. 

The preprint version is now available on [TechRxiv](https://doi.org/10.36227/techrxiv.23296487.v1).

You may preserve the following directory tree to run the code on your local machine without further modifications.

```
$ Directory tree
.
├── data\
│   └── (containing sample IMU data)
├── model\
│   ├── Type1.py
│   ├── Type2.py
│   └── Type3.py
├── notebook_version\
│   └── (containing notebook versions of the implementation)
├── pretrained_model\
│   ├── ... 
│   ├── ...
│   └── ... 
├── utils
│   ├── ... 
│   ├── ...
│   └── ... 
├── tuning.py
├── eval.py
├── constants.py
└── demo.py
```

You may use data samples in the `data` to run the code, or download the full dataset on [SimTK](https://simtk.org/projects/imu-exercise).

## Implementation 

The implementation was tested with `Python 3.8.10` and the following packages:

- `numpy 1.22.4`
- `scipy 1.7.3`
- `pandas 1.5.3`
- `scikit-learn 1.2.0`
- `torch 1.13.1+cu116` 
- `tqdm 4.64.1`

In addition, `matplotlib 3.6.3` and `seaborn 0.12.2` were used for plots.

## Quick Start

TBD.

## Citation

If your find the code helpful for your work, please consider citing [our paper](https://doi.org/10.36227/techrxiv.23296487.v1)

```
Phan, Vu; Song, Ke; Silva, Rodrigo Scattone; Silbernagel, Karin G.; Baxter, Josh R.; Halilaj, Eni (2023). Seven Things to Know about Exercise Monitoring with Inertial Sensing Wearables. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.23296487.v1
```





