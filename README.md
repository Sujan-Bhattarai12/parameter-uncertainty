# Measure of parameter uncertainty using High Performance computing and Machine Learning
**The project output is in production and is not public as of now. The Dashboard should be available soon**
![Cluster](cluster.png)

## Overview
This project leverages the power of supercomputing to handle and analyze extensive climate data produced by climate modeling at the National Center for Atmospheric Research (NCAR). The model used for this data is part of the Intergovernmental Panel on Climate Change (IPCC) climate prediction efforts. The shear volume size of the data is 4TB, which will be handled with cluster computing, resource provided by UCAR and UCSB.

## Objectives
The main goal of this project to use the principle of parameter uncertainty in quantifying the effects of each variables in a output.There are 32 parameters and they all exert different infuence in response variables and that differs for each of the response variable. This project aims to quantify how much influence parameters exert using FAST method, also leveraging the machine learning. The purpose of machine learning is to to train the model in small sets of dataset so that entire 4TB can be avoided. This saves resource and computational cost.

## Methodology
1. **Data Handling**: Utilized supercomputing resources to process and manage the large volume of NETCDF climate data. The datasets will be read on tiles from 40 clusters.
2. **Gaussian Regression**: The dataset will be used to train a GPR based machine learning model. When the model has sufficient high accuracy, which is greater than 70%, then FAST test can be integrated.
3. **Sensitivity Analysis**: Apply the FAST and measure the sensitivity of the model outputs to each of 32 variables. Lets say if you are interested in leaf nitrogen, then you will get a barplot plotted with 32 variables.
