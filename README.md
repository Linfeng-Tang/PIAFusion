<h1 align="center"><a href="https://www.sciencedirect.com/science/article/abs/pii/S156625352200032X">PIAFusion: A Progressive Infrared and Visible Image Fusion Network Based on Illumination Aware</a></h1>

<p align="center"><a href="https://github.com/Linfeng-Tang">Linfeng Tang</a>&emsp; Jiteng Yuan&emsp; Hao Zhang&emsp; Xingyu Jiang&emsp; <a href="https://sites.google.com/site/jiayima2013">Jiayi Ma</a></p>
<p align="center"><strong>Wuhan University</strong></p>
<p align="center"><strong>Information Fusion</strong> &middot; 2022</p>
<p align="center"><a href="https://esi.help.clarivate.com/Content/overview.htm"><img src="https://img.shields.io/badge/%F0%9F%94%A5_ESI_Hot-Top_0.1%25-E85D3F?style=flat-square" alt="ESI Hot Paper (top 0.1%)"></a> <a href="https://esi.help.clarivate.com/Content/overview.htm"><img src="https://img.shields.io/badge/%F0%9F%8F%86_ESI_Highly_Cited-Top_1%25-D4A017?style=flat-square" alt="ESI Highly Cited Paper (top 1%)"></a><br><sub><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=PyRqpAsAAAAJ&citation_for_view=PyRqpAsAAAAJ:YsMSGLbcyi4C">Google Scholar &middot; <strong>1,265 citations</strong></a> &middot; updated July 18, 2026</sub></p>

The PyTorch implementation of our project, accomplished by @[linklist2](https://github.com/linklist2), can be fetched from [https://github.com/linklist2/PIAFusion_pytorch](https://github.com/linklist2/PIAFusion_pytorch).

A new benchmark dataset for infrared and visible fusion are released in this paper, which is termed **[MSRS](https://github.com/Linfeng-Tang/MSRS)**.

## Architecture
![The overall framework of the progressive infrared and visible image fusion algorithm based on illumination-aware.](https://github.com/Linfeng-Tang/PIAFusion/blob/main/Figure/PIAFusion.png)

## Example

![An example of illumination imbalance.](https://github.com/Linfeng-Tang/PIAFusion/blob/main/Figure/Illumination_aware.png)
An example of illumination imbalance. From left to right: infrared image, visible image, the fused results of DenseFuse, FusionGAN, and our proposed PIAFusion.
The visible image contains abundant information, such as texture details in the daytime (top row). But salient targets and textures are all included in the infrared image at nighttime (bottom row). Existing methods ignore the illumination imbalance issues, causing detail loss and thermal target degradation. Our algorithm can adaptively integrate meaningful information according to illumination conditions.
## Recommended Environment

 - [ ] tensorflow-gpu 1.14.0 
 - [ ] scipy 1.2.0   
 - [ ] numpy 1.19.2
 - [ ] opencv 3.4.2 

## To Training

 ### Training the Illumination-Aware Sub-Network
Run: "python main.py --epoch=100 --is_train=True model_type=Illum --DataSet=MSRS"
The dataset for training the illumination-aware sub-network can be download from [data_illum.h5](https://pan.baidu.com/s/1D7XVGFyPgn9lH6JxYXt65Q?pwd=PIAF).

### Training the Illmination-Aware Fusion Network
Run: "python main.py --epoch=30 --is_train=True model_type=PIAFusion --DataSet=MSRS"
The dataset for training the illumination-aware fusion network can be download from [data_MSRS.h5](https://pan.baidu.com/s/1D7XVGFyPgn9lH6JxYXt65Q?pwd=PIAF).

## To Testing
### The MSRS Dataset
Run: "python main.py --is_train=False model_type=PIAFusion --DataSet=MSRS"

### The RoadScene Dataset
Run: "python main.py --is_train=False model_type=PIAFusion --DataSet=RoadScene"

### The TNO Dataset
Run: "python main.py --is_train=False model_type=PIAFusion --DataSet=TNO"

## Acknowledgement
Our Multi-Spectral Road Scenarios (**[MSRS](https://github.com/Linfeng-Tang/MSRS)**) dataset is constructed on the basis of the **[MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)** dataset[1].

[1] Ha, Q., Watanabe, K., Karasawa, T., Ushiku, Y., Harada, T., 2017. Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes, in: Proceedings of the IEEE International Conference on Intelligent Robots and Systems, pp.5108–5115.

## If this work is helpful to you, please cite it as：
```
@article{Tang2022PIAFusion,
  title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
  author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
  journal={Information Fusion},
  volume = {83-84},
  pages = {79-92},
  year = {2022},
  issn = {1566-2535},
  publisher={Elsevier}
}
```
