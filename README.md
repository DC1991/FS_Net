# Category-Level 6D Pose Estimation

This code is for our CVPR2021 oral paper: FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism. (I will upload the code in next few weeks, thanks for you patience.)

## Experiment setup

OS: Ubuntu 16.04

GPU: 1080 Ti

Programme language: Python 3.6
 
If you find our [paper](http://arxiv.org/abs/2103.07054) or code is useful, please cite our paper:

@InProceedings{Chen_2021_CVPR,  
author = {Chen, Wei and Jia, Xi and Chang, Hyung Jin and Duan, Jinming and Linlin, Shen and Leonardis, Ales},  
title = {FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism},  
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2021}  
}  


## Contributions
Our framework is built on our previous work [G2L-Net](https://github.com/DC1991/G2L_Net), with the following Contributions:
 1. New latent feature learning  
        >>> [3D graph convolution](https://github.com/j1a0m0e4sNTU/3dgcn/issues) based observed points reconstruction

 2. New rotation representation  
        >>> Decomposable vector-based rotation representation

 3. New 3D data augmentation  
        >>> Box-cage based, online 3D data augmentation



## Pre requirements

You can find the main requirements in 'requirement.txt'.

### Trained model and sample data
>>Please download the data.zip [here](https://drive.google.com/file/d/15efs1IIjbRnWIlh-9sXMfbqyL4S08bEG/view?usp=sharing
>), and the unzip the 'trained_model.zip' under 'yolov3_fsnet/' folder and
 'test_scene_1
.zip' under 'yolov3_fsnet/data/' folder.   

>>The trained model for YOLOv3 will be downloaded automatically. We use the 'yolov5l.pt' provided by this [git](https
>://github.com/ultralytics/yolov3). According to our test, this trained model works well for category 'laptop', you
> may need to re-train the 2D detect model for other categories.

## Demo

python yolov3_fsnet/detect_fsnet.py  
please note: The code is created and debugged in Pycharm, therefore you may need to change the import head in other
 python IDE. 
## Training
### Generate training point clouds
To generate your own dataset, first use the data preprocess code provided in this [git](https://github.com/mentian/object-deformnet/blob/master/preprocess/pose_data.py), and then use the code
 provided in 'gen_pts.py'.

### Training FS_Net
#### YOLOv3 Training
For 2D detection training part, please refer to this [git](https://github.com/ultralytics/yolov3)
#### FS_Net Training
The sample training data and training code will be uploaded soon.