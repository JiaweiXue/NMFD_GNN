# NMFD-GNN (Network Macroscopic Fundamental Diagram - Graph Neural Network)
A physics-informed machine learning model for traffic state imputation.

## Introduction
* Traffic state imputation (TSI) refers to the estimation of missing values of traffic variables, such as flow rate and traffic density, using available data.
* Despite these advantages, purely data-driven models have limitations. For example, they lack traffic engineering-related interpretations. The parameters of the learned neural network model can be messy with positive and negative values without physical meaning, making the model a black box to transportation researchers.
* This study proposes NMFD-GNN, a physics-informed machine learning (PIML) model that fuses the NMFD with the GNN to perform network-wide TSI. 
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

## Building the PIML to perform the TSI task.
<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/task.png" width="500">
</p>

## The PIML is composed of the physics module based on the λ-trapezoidal MFD, and the machine learning module based on the graph convolutional network.
<p align="center">
  <img src="https://github.com/JiaweiXue/NMFD_GNN/blob/main/figure/method.png" width="500">
</p>

## License
MIT license
