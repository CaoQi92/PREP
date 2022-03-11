# PREP: Pre-training with Temporal Elapse Inference for Popularity Prediction

This repository is an Pytorch implementation of our proposed pre-training framework for popularity prediction in the following paper:

[Qi Cao, Huawei Shen, Yuanhao Liu, Jinhua Gao, Xueqi Cheng. 2022. PREP: Pre-training with Temporal Elapse Inference for Popularity Prediction. In Companion Proceedings of the Web Conference 2022 (WWW ’22 Companion), April 25–29, 2022, Virtual Event.]()

For more details, you can refer this paper.

##  Introduction

Predicting the popularity of online content is a fundamental problem in various applications. One practical challenge takes roots in the varying length of observation time or prediction horizon, i.e., a good model for popularity prediction is desired to handle various prediction settings. 




However, most existing methods adopt a separate training paradigm for each prediction setting and the obtained model for one setting is difficult to be generalized to others, causing a great waste of computational resources and a large demand for downstream labels. To solve the above issues, we propose a novel pre-training framework for popularity prediction, namely PREP, aiming to pre-train a general representation model from the readily available unlabeled diffusion data, which can be effectively transferred into various prediction settings. We design a novel pretext task for pre-training, i.e., temporal elapse inference for two randomly sampled time slices of popularity dynamics, impelling the representation model to effectively learn intrinsic knowledge about popularity dynamics. Experimental results conducted on two real datasets demonstrate the generalization and efficiency of the pre-training framework for different popularity prediction task settings.

## Requirement

## Usage

## Cite
