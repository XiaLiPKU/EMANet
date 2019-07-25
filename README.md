# EMANet

This repository is for Expectation-Maximization Attention Networks for Semantic Segmentation (to appear in ICCV 2019, Oral presentation),

by [Xia Li](https://xialipku.github.io/), [Zhisheng Zhong](https://zzs1994.github.io/), [Jianlong Wu](https://jlwu1992.github.io/), [Yibo Yang](https://scholar.google.com.hk/citations?user=DxXXnCcAAAAJ&hl=en), [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm) and [Hong Liu](https://scholar.google.com/citations?user=4CQKG8oAAAAJ&hl=en).

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

Tab 1. Detailed comparisons with Deeplabs. All results are achieved with the backbone ResNet-101 and output stride 8.  The FLOPs and memory are computed with the input size 513×513. SS: Single scale input during test. MS: Multi-scale input. Flip: Adding left-right flipped input. EMANet (256) and EMANet (512) represent EMANet withthe number of input channels for EMA as 256 and 512, respectively. 

|Method     |SS   |MS+Flip|FLOPs |Memory|Params|
|:---------:|:---:|:-----:|:----:|:----:|:----:|
|ResNet-101 |-    |-      |190.6G|2.603G|42.6M |
|DeeplabV3  |78.51|79.77  |+63.4G|+66.0M|+15.5M|
|DeeplabV3+ |79.35|80.57  |+84.1G|+99.3M|+16.3M|
|PSANet     |78.51|79.77  |+56.3G|+59.4M|+18.5M|
|EMANet(256)|79.73|80.94  |**+21.1G**|**+12.3M**|**+4.87M**|
|**EMANet(512)**|**80.05**|**81.32**  |+43.1G|+22.1M|+10.0M|

To be note, the majority overheads of EMANets come from the 3x3 convs before and after the EMA Module. As for the EMA Module itself, its computation is only 1/3 of a 3x3 conv's, and its parameter number is even smaller than a 1x1 conv.

## Comparisons with SOTAs

Tab 2. Comparisons on the PASCAL VOC test dataset. OS means the output stride for training. For test, output stride is set as 8.

|Method      	| Backbone      | mIoU(\%)	|
|:-------------:|:-------------:|:-------------:|
|GCN     	| ResNet-152    | 83.6		|
|RefineNet	| ResNet-152    | 84.2		|
|Wide ResNet 	| WideResNet-38 | 84.9		|
|PSPNet 	| ResNet-101    | 85.4		|
|DeeplabV3	| ResNet-101 	| 85.7		|
|PSANet		| ResNet-101	| 85.7		|
|EncNet 	| ResNet-101    | 85.9		|
|DFN		| ResNet-101    | 86.2		|
|Exfuse		| ResNet-101    | 86.2		|
|IDW-CNN    	| ResNet-101    | 86.3		|
|SDN 		| DenseNet-161  | 86.6		|
|DIS        	| ResNet-101    | 86.8		|
|**EMANet101-OS16** 	| ResNet-101	| **87.3**	|
|**EMANet101-OS8** 	| ResNet-101	| **87.7**	|
|DeeplabV3+	| Xception-65   | 87.8		|
|Exfuse 	| ResNeXt-131   | 87.9		|
|MSCI       	| ResNet-152    | 88.0		|
|**EMANet152-OS16** 	| ResNet-152	| **88.0**	|
|**EMANet152-OS8** 	| ResNet-152	| running	|
