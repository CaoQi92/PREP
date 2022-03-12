# PREP: Pre-training with Temporal Elapse Inference for Popularity Prediction

This repository is an Pytorch implementation of our proposed pre-training framework for popularity prediction in the following paper:

[Qi Cao, Huawei Shen, Yuanhao Liu, Jinhua Gao, Xueqi Cheng. 2022. PREP: Pre-training with Temporal Elapse Inference for Popularity Prediction. In Companion Proceedings of the Web Conference 2022 (WWW ’22 Companion), April 25–29, 2022, Virtual Event.]()

For more details, you can refer this paper.

##  Introduction

Predicting the popularity of online content is a fundamental problem in various applications. One practical challenge takes roots in the varying length of observation time or prediction horizon, i.e., a good model for popularity prediction is desired to handle various prediction settings. 

&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/CaoQi92/PREP/blob/main/Figure/Various_Popularity_Prediction_Settings.png" width="600" align="center">

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Figure 1: Various Popularity Prediction Settings

However, most existing methods adopt a separate training paradigm for each prediction setting and the obtained model for one setting is difficult to be generalized to others, causing a great waste of computational resources and a large demand for downstream labels. To solve the above issues, we propose a novel pre-training framework for popularity prediction, namely PREP, aiming to pre-train a general representation model from the readily available unlabeled diffusion data, which can be effectively transferred into various prediction settings. 

&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/CaoQi92/PREP/blob/main/Figure/Seperate_Pretraining.png" width="600" align="center">

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Figure 2: Comparison between Separate Training an Pre-training Framework

We design a novel pretext task for pre-training, i.e., temporal elapse inference for two randomly sampled time slices of popularity dynamics, impelling the representation model to effectively learn intrinsic knowledge about popularity dynamics. 

&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/CaoQi92/PREP/blob/main/Figure/Temporal_Elapse_Inference.png" width="600" align="center">

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Figure 3: Illustration of temporal elapse inference. 


Experimental results conducted on two real datasets demonstrate the generalization and efficiency of the pre-training framework for different popularity prediction task settings.

## Requirement
Python 2.7

Pytorch

## Usage
#### 1. Pre-training the deep representation model through unlabeled diffusion data
```
sh run_pretrain.sh
```
For detailed description of pre-training parameters, you can run
```
python -u PREP_TCN_pretrain.py --help
```
#### 2. Fine-tuning the pre-trained modoel through few downstream labels
```
sh run_downstream.sh
```
For detailed description of fine-tuning parameters, you can run
```
python -u PREP_TCN_downstream.py --help
```

## Cite
Please cite our paper if you use this code in your work:
```
@inproceedings{cao2020coupledgnn,
  title={PREP: Pre-training with Temporal Elapse Inference for Popularity Prediction},
  author={Cao, Qi and Shen, Huawei and Liu, Yuanhao and Gao, Jinhua and Cheng, Xueqi},
  booktitle={Companion Proceedings of the Web Conference 2022},
  series={WWW ’22 Companion},
  year={2022}
}
```
