# PIAFusion
The code of “PIAFusion: A Progressive Infrared and Visible Image Fusion Network Based on Illumination Aware”

## Recommended Environment

 - [ ] tensorflow-gpu 1.14.0 
 - [ ] scipy 1.2.0   
 - [ ] numpy 1.19.2
 - [ ] opencv 3.4.2 

## To Training

 ### Training the Illumination-Aware Sub-Network
Run: "python main.py --epoch=100 --is_train=True model_type=Illum --DataSet=MSRS"
### Training the Illmination-Aware Fusion Network
Run: "python main.py --epoch=30 --is_train=True model_type=PIAFusion --DataSet=MSRS"
The training data can be load from [here](1111).

## To Testing
### The MSRS Dataset
Run: "python main.py --is_train=False model_type=PIAFusion --DataSet=MSRS"

### The RoadScene Dataset
Run: "python main.py --is_train=False model_type=PIAFusion --DataSet=RoadScene"

### The TNO Dataset
Run: "python main.py --is_train=False model_type=PIAFusion --DataSet=TNO"

#### Acknowledgement
Our Multi-Spectral Road Scenarios (**MSRS**) dataset is constructed on the basis of the **[MFNet](https://github.com/haqishen/MFNet-pytorch)** dataset[1].

[1] Ha, Q., Watanabe, K., Karasawa, T., Ushiku, Y., Harada, T., 2017. Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes, in: Proceedings of the IEEE
International Conference on Intelligent Robots and Systems, pp.
5108–5115.

## If this work is helpful to you, please cite it as：
