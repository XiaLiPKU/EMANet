# EMANet

## Background

This repository is for [Expectation-Maximization Attention Networks for Semantic Segmentation](https://xialipku.github.io/publication/expectation-maximization-attention-networks-for-semantic-segmentation/) (to appear in ICCV 2019, Oral presentation),

by [Xia Li](https://xialipku.github.io/), [Zhisheng Zhong](https://zzs1994.github.io/), [Jianlong Wu](https://jlwu1992.github.io/), [Yibo Yang](https://scholar.google.com.hk/citations?user=DxXXnCcAAAAJ&hl=en), [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm) and [Hong Liu](https://scholar.google.com/citations?user=4CQKG8oAAAAJ&hl=en) from Peking University.

**The source code is now available!**

### citation
If you find EMANet useful in your research, please consider citing:

	@inproceedings{li19,
	    author={Xia Li and Zhisheng Zhong and Jianlong Wu and Yibo Yang and Zhouchen Lin and Hong Liu},
	    title={Expectation-Maximization Attention Networks for Semantic Segmentation},
	    booktitle={International Conference on Computer Vision},   
	    year={2019},   
	}

### table of contents
- [Introduction](#introduction)
- [Design](#design)
- [Usage](#usage)
- [Ablation studies](#ablation-studies)
- [Comparisons with SOTAs](#comparision-with-sotas)

## Introduction
Self-attention mechanism has been widely used for various tasks. It is designed to compute the representation of each position by a weighted sum of the features at all positions. Thus, it can capture long-range relations for computer vision tasks. However, it is computationally consuming. Since the attention maps are computed w.r.t all other positions. In this paper, we formulate the attention mechanism into an expectation-maximization manner and iteratively estimate a much more compact set of bases upon which the attention maps are computed. By a weighted summation upon these bases, the resulting representation is low-rank and deprecates noisy information from the input. The proposed Expectation-Maximization Attention (EMA) module is robust to the variance of input and is also friendly in memory and computation. Moreover, we set up the bases maintenance and normalization methods to stabilize its training procedure. We conduct extensive experiments on popular semantic segmentation benchmarks including PASCAL VOC, PASCAL Context, and COCO Stuff, on which we set new records.
![EMA Unit](https://xialipku.github.io/publication/expectation-maximization-attention-networks-for-semantic-segmentation/featured_hu9697392da5f15752eba3cc9c3d5fcfba_326134_720x0_resize_lanczos_2.png)

## Design

As so many peers have starred at this repo, I feel the great pressure, and try to release the code with high quality.
That's why I didn't release it until today (Aug, 22, 2018). It's known that the design of the code structure is not an easy thing.
Different designs are suitable for different usage. Here, I aim at making research on Semantic Segmentation, especially on PASCAL VOC, more easier.
So, I delete necessary encapsulation as much as possible, and leave over less than 10 python files. To be honest, the global variables in settings are
not a good design for large project. But for research, it offers great flexibility. So, hope you can understand that

For research, I recommand seperatting each experiment with a folder. Each folder contains the whole project, and should be named as the experiment settings, such as 'EMANet101.moving_avg.l2norm.3stages'. Through this, you can keep tracks of all the experiments, and find their differences just by the 'diff' command.

## Usage

1. Install the libraries listed in the 'requirements.txt'
2. Downloads [images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [labels](https://drive.google.com/file/d/1OqX6s07rFqtu-JZCjdjnJDv1QfDz9uG7/view?usp=sharing) of PASCAL VOC and SBD, decompress them together.
3. Downloads the pretrained [ResNet50](https://hangzh.s3.amazonaws.com/encoding/models/resnet50-ebb6acbb.zip) and [ResNet101](https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.zip), unzip them, and put into the 'models' folder.
4. Change the 'DATA_ROOT' in settings.py to where you place the dataset.
5. Run `sh clean.sh` to clear the models and logs from the last experiment.
6. Run `python train.py` for training and `sh tensorboard.sh` for visualization on your browser.
7. Or you can download the [pretraind model](https://drive.google.com/file/d/1rOfV1dpcvW2lxfGJ9RtntBXF-z64kjsC/view?usp=sharing), put into the 'models' folder, and skip step 6.
8. Run `python eval.py` for validation

## Ablation Studies

The following results are referred from the paper. For this repo, it's not strange to get even higer performance. If so, I'd like you share it in the issue.
By now, this repo only provides the SS inference. I may release the code for MS and Flip latter.

Tab 1. Detailed comparisons with Deeplabs. All results are achieved with the backbone ResNet-101 and output stride 8. The FLOPs and memory are computed with the input size 513×513. SS: Single scale input during test. MS: Multi-scale input. Flip: Adding left-right flipped input. EMANet (256) and EMANet (512) represent EMANet withthe number of input channels for EMA as 256 and 512, respectively. 

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

Note that, for validation on the 'val' set, you just have to train 30k on the 'trainaug' set.
But for test on the evaluation server, you should first pretrain on COCO, and then 30k on 'trainaug', and another 30k on the 'trainval' set.

Tab 2. Comparisons on the PASCAL VOC test dataset.

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
|**EMANet101** 	| ResNet-101	| **87.7**	|
|DeeplabV3+	| Xception-65   | 87.8		|
|Exfuse 	| ResNeXt-131   | 87.9		|
|MSCI       	| ResNet-152    | 88.0		|
|**EMANet152** 	| ResNet-152	| **88.2**	|

## Code Borrowed From
[RESCAN](https://github.com/XiaLiPKU/RESCAN)

[Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

