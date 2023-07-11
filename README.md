# NMFD-GNN 
* Network Macroscopic Fundamental Diagram - Graph Neural Network.
* A physics-informed machine learning (PIML) model for traffic state imputation (TSI).

## Introduction
* TSI refers to the estimation of missing values of traffic variables, such as flow rate and traffic density, using available data.
* For the TSI task, pure data-driven methods lack traffic engineering-related interpretations. The parameters of the learned neural network model can be messy with positive and negative values without physical meaning, making the model a black box to transportation researchers.
* This study proposes NMFD-GNN, a PIML model that fuses the NMFD with the GNN to perform network-wide TSI. 
* Our proposed NMFD-GNN model and its variant, NMFD-GNN-HINGE, are evaluated on road networks located in Zurich and London from the UTD19 dataset (https://utd19.ethz.ch/). 
  
## Directory structure
* **utils**: the process of preparing the features and labels for the TSI task.
* **model**: our proposed NMFD-GNN and NMFD-GNN-HINGE models.
* **result**: the screenshot of our implementation results.
* **figure**: description of problems, methods, and data.

## Requirements
* Python 2.7.5 or higher.
* Torch 2.0.0 or higher. 

## Manuscript
**Network Macroscopic Fundamental Diagram-Informed Graph Learning for Traffic State Imputation.**
Jiawei Xue, Eunhan Ka, Yiheng Feng, Satish V. Ukkusuri\*, July 2023.

## Building the PIML to perform the TSI task
<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/task.png" width="300">
</p>

## NMFD-GNN = the physics module (the λ-trapezoidal MFD) + the machine learning module (the graph convolutional network)
* The λ-trapezoidal MFD was proposed by the following study:
* Ambühl, et al. (2020). A functional form with a physical meaning for the macroscopic fundamental diagram. Transportation Research Part B.
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
