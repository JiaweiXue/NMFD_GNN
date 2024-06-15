# NMFD-GNN 
* Network Macroscopic Fundamental Diagram (NMFD) - Graph Neural Network (GNN).
* A physics-informed machine learning model for traffic state imputation (TSI).

## Introduction
* TSI refers to the estimation of missing values of traffic variables, such as flow rate and traffic density, using available data.
* This study proposes NMFD-GNN, a physics-informed machine learning model that fuses the NMFD with the GNN to perform network-wide TSI. 
* Our proposed NMFD-GNN model and its variants, NMFD-GNN-HINGE and NMFD-GNN-UPPER, are evaluated on road networks located in Zurich and London from the UTD19 dataset (https://utd19.ethz.ch/). 
  
## Directory structure
* **utils**: preparing the features and labels for the TSI task.
* **model**: building NMFD-GNN, NMFD-GNN-HINGE, and NMFD-GNN-UPPER models.
* **main**: training and testing the model.
* **result**: presenting implementation results.
* **figure**: describing problems, methods, and data.

## Requirements
* Python 2.7.5 or higher.
* Torch 2.0.0 or higher. 

## Paper
**Network Macroscopic Fundamental Diagram-informed Graph Learning for Traffic State Imputation.**
Poster presentation at ISTTT25; Publication on Transportation Research Part B: Methodological.
Jiawei Xue, Eunhan Ka, Yiheng Feng, Satish V. Ukkusuri\*, June 2024.

## Building the PIML to perform the TSI task
<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/task.png" width="300">
</p>

## NMFD-GNN = the physics module (the λ-trapezoidal MFD) + the machine learning module (the graph convolutional network)
* The λ-trapezoidal MFD was proposed by the following study:
* Ambühl, et al. (2020). A functional form with a physical meaning for the macroscopic fundamental diagram. Transportation Research Part B: Methodological.
<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/method.png" width="500">
</p>

## Study areas in Zurich and London

<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/study_area.png" width="400">
</p>

## MFDs
<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/mfd.png" width="550">
</p>

## License
MIT license
