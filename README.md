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
Jiawei Xue, Eunhan Ka, Yiheng Feng, Satish V. Ukkusuri\*, June 2024. 

Poster presentation at ISTTT25; Publication on Transportation Research Part B: Methodological.



## Building NMFD-GNN to perform the TSI task
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

The following papers form a solid foundation for this study. We sincerely thank their contributions to the community.

| Index | Authors | Title | Publication |  
| :-----| :-----| :-----| :-----|
| 1 | Loder, A., L. Ambühl, M. Menendez, and K. W. Axhausen| Understanding traffic capacity of urban networks | Scientific Reports, 2019 |
| 2 | Johari, M., M. Keyvan-Ekbatani, L. Leclercq, D. Ngoduy, and H. S. Mahmassani| Macroscopic network-level traffic models: Bridging fifty years of development toward the next era | TR-Part C, 2021 |
| 3 | Ambühl, L, A. Loder, M. C. Bliemer, M. Menendez, and K. W. Axhausen| A functional form with a physical meaning for the macroscopic fundamental diagram | TR-Part B, 2020 |
| 4 | Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C. and Sun, M| Graph neural networks: A review of methods and applications | AI Open, 2020 |
| 5 | Liang. Y., Z. Zhao, and L. Sun| Memory-augmented dynamic graph convolution networks for traffic data imputation with diverse missing patterns | TR-Part C, 2022 |

## License
MIT license
