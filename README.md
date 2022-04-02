# PIAFusion
This is official tensorflow implementation of “[PIAFusion: A progressive infrared and visible image fusion network based on illumination aware](https://www.sciencedirect.com/science/article/pii/S156625352200032X)”

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
The dataset for training the illumination-aware sub-network can be download from [data_illum.h5](https://pan.baidu.com/s/12Xu1Ep7kKwe03HFBQYWkhw?pwd=PIAF ).

### Training the Illmination-Aware Fusion Network
Run: "python main.py --epoch=30 --is_train=True model_type=PIAFusion --DataSet=MSRS"
The dataset for training the illumination-aware fusion network can be download from [data_MSRS.h5](https://pan.baidu.com/s/1cO_wn2DOpiKLjHPaM1xZYQ?pwd=PIAF).

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
  year={2022},
  publisher={Elsevier}
}
```
