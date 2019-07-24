# EMANet

This repository is for Expectation-Maximization Attention Networks for Semantic Segmentation (to appear in ICCV 2019, Oral presentation),

by [Xia Li](https://xialipku.github.io/), [Zhisheng Zhong](https://zzs1994.github.io/), [Jianlong Wu](https://jlwu1992.github.io/), [Yibo Yang](https://github.com/iboing), [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm) and [Hong Liu](https://scholar.google.com/citations?user=4CQKG8oAAAAJ&hl=en).

### citation
If you find EMANet useful in your research, please consider citing:

	@inproceedings{li19,
	    author={Xia Li and Zhisheng Zhong and Jianlong Wu and Zhisheng Zhong and Zhouchen Lin and Hong Liu},
	    title={Expectation-Maximization Attention Networks for Semantic Segmentation},
	    booktitle={International Conference on Computer Vision},   
	    year={2019},   
	}

### table of contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Ablation studies](#ablation-studies)
- [Comparisons with SOTAs](#comparision-with-sotas)

## Introduction

## Usage

## Ablation Studies

## Comparisons with SOTAs

|Method      	| Backbone      | mIoU(\%)	|
|:-------------:|:-------------:|:---------:|
|Wide ResNet 	| WideResNet-38 | 84.9		|
|PSPNet 		| ResNet-101    | 85.4		|
|DeeplabV3		| ResNet-101 	| 85.7		|
|PSANet			| ResNet-101	| 85.7		|
|EncNet 		| ResNet-101    | 85.9		|
|DFN			| ResNet-101    | 86.2		|
|Exfuse			| ResNet-101    | 86.2		|
|IDW-CNN    	| ResNet-101    | 86.3		|
|SDN 			| DenseNet-161  | 86.6		|
|DIS        	| ResNet-101    | 86.8		|
|**EMANet** 	| ResNet-101	| **87.7**	|
|GCN     		| ResNet-152    | 83.6		|
|RefineNet		| ResNet-152    | 84.2		|
|DeeplabV3+		| Xception-65   | 87.8		|
|Exfuse 		| ResNeXt-131   | 87.9		|
|MSCI       	| ResNet-152    | 88.0		|
