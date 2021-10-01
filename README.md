# Aim: To Augment 3D Benchmark Dataset and Train 3D Object Detection Model for Crowd Pedestrian detection for better accuracy

## GIT Repository (this repo):  
 - https://github.com/LidarSpecialists/LIDAR_Crowd_Detection_Team19_Project

## System Demo - (Stage 1) - LIDAR Crowd Pedestrian Detection Demo  
 - https://www.youtube.com/watch?v=vpy4CWgZYRM

## System Demo - (Stage 2) - LIDAR Crowd Data Augmentation Demo   
 - https://www.youtube.com/watch?v=ucfATfkJFZk

## Presentation Link 
- https://github.com/LidarSpecialists/LIDAR_Crowd_Detection_Team19_Project/blob/main/reports/Lidar_Crowd_Detection_Team19_ProjectReport_Phase2_v0(1).docx

## Checkpoint of Dense Detection Model
- https://drive.google.com/drive/folders/1ayAreMIk_EU4jIGYCqFv5fUa30klFlZL?usp=sharing

## Augmented Dataset Google DRive location* 
- https://drive.google.com/drive/folders/1-1zCORipbNB8GMjkjxuwxvJOFl0rlbcj?usp=sharing
- Size: 32 GB


# Sponsor Company Intro

SenseTime (https://www.sensetime.com/en) is a leading global company focused on developing responsible AI technologies that advance the world’s economies, society and humanity for a better tomorrow.  They have made a number of technological breakthroughs, one of which is the first ever computer system in the world to achieve higher detection accuracy than the human eye. 

# Business Problem Background

SenseTime has sponsored us with this project to create a benchmark dataset for LIDAR crowd detection by introducing a novel data augmentation method for LiDAR-only learning problems that can greatly increase the convergence speed and performance. This will be done using computer vision and other deep learning tools.

# Augmented 3D Benchmark Dataset
Aim is to augment the dataset from KITTI / Waymo Open Dataset for Dense scenarios. A Pedestrian to be placed densely on a terrain or street without collision with other points in the 3D space. The optimal target will be to have 7000+ training point cloud frames per attribute value.

This augmented dataset will be used for Capstone projects and future R&D projects in SenseTime. A good benchmark dataset is a cornerstone of model training. With a good benchmark dataset, researchers can significantly reduce the time spent on data collection and labelling required before training a model. For example, MNIST is one of the most popular deep learning datasets for handwritten digits recognition and has been widely used by data scientists to train and test new architectures or frameworks. 

# Standard Datasets
We used KITTI & Smantic KITTI datasets but could not leverage Waymo Open dataset because it is entirely in a different format than the KITTI and hard to convert to KITTI format as well.

# 3D Detection Baseline Model
Point RCNN - https://github.com/sshaoshuai/PointRCNN

# Other Important References 
## VFNet
 *https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf*
## KITTI
 *http://www.cvlibs.net/datasets/kitti*

# Stage1 Data augmentation
Please download the official KITTI 3D object detection dataset (http://www.cvlibs.net/datasets/kitti/) & semantic kitti dataset and organize the downloaded files as follows:
```
Capstone
├── data
│   ├── KITTI
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
│   │   ├── sequence
│   │   │   ├──0
│   │   │   ├──1
```
use the following commend to generate the pedestrian database or use the generated database available in the link above. Directory and parameter could be set in the script

python kitti_crop_20210615.py

use the following commend to generate drop pedestrian densely on to street and generated augmented data. Range of number of pedestrian and crowdness could be set in the script and functions to specify the augmented data. The data will be auto generated into out_dir

python pedestrian_dispense_20200909.py 


# Setup Steps

## Dataset preparation

Please download the official KITTI 3D object detection dataset (http://www.cvlibs.net/datasets/kitti/) and organize the downloaded files as follows:
```
PointRCNN
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```

Here the images are only used for visualization and the road planes are optional for data augmentation in the training.

## Pretrained model

You could download the pretrained model(Car) of PointRCNN from here(~15MB), which is trained on the train split (3712 samples) and evaluated on the val split (3769 samples) and test split (7518 samples). The performance on validation set is as follows:
```
Car AP@0.70, 0.70, 0.70:
bbox AP:96.91, 89.53, 88.74
bev  AP:90.21, 87.89, 85.51
3d   AP:89.19, 78.85, 77.91
aos  AP:96.90, 89.41, 88.54
```


## Inference
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rcnn 
```

* To evaluate all the checkpoints of a specific training config file, add the `--eval_all` argument, and run the command as follows:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --eval_mode rcnn --eval_all
```

* To generate the results on the *test* split, please modify the `TEST.SPLIT=TEST` and add the `--test` argument. 

Here you could specify a bigger `--batch_size` for faster inference based on your GPU memory. Note that the `--eval_mode` argument should be consistent with the `--train_mode` used in the training process. If you are using `--eval_mode=rcnn_offline`, then you should use `--rcnn_eval_roi_dir` and `--rcnn_eval_feature_dir` to specify the saved features and proposals of the validation set. Please refer to the training section for more details. 

## Training
Currently, the two stages of PointRCNN are trained separately. Firstly, to use the ground truth sampling data augmentation for training, we should generate the ground truth database as follows:
```
python generate_gt_database.py --class_name 'Car' --split train
```

### Training of RPN stage
* To train the first proposal generation stage of PointRCNN with a single GPU, run the following command:
```
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200
```

* To use **mutiple GPUs for training**, simply add the `--mgpus` argument as follows:
```
CUDA_VISIBLE_DEVICES=0,1 python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200 --mgpus
```

After training, the checkpoints and training logs will be saved to the corresponding directory according to the name of your configuration file. Such as for the `default.yaml`, you could find the checkpoints and logs in the following directory:
```
PointRCNN/output/rpn/default/
```
which will be used for the training of RCNN stage. 

### Training of RCNN stage
Suppose you have a well-trained RPN model saved at `output/rpn/default/ckpt/checkpoint_epoch_200.pth`, 
then there are two strategies to train the second stage of PointRCNN. 

(a) Train RCNN network with fixed RPN network to use online GT augmentation: Use `--rpn_ckpt` to specify the path of a well-trained RPN model and run the command as follows:
```
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth
```
(b) Train RCNN network with offline GT augmentation: 
1. Generate the augmented offline scenes by running the following command:
```
python generate_aug_scene.py --class_name Car --split train --aug_times 4
```
2. Save the RPN features and proposals by adding `--save_rpn_feature`:

* To save features and proposals for the training, we set `TEST.RPN_POST_NMS_TOP_N=300` and `TEST.RPN_NMS_THRESH=0.85` as follows:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --eval_mode rpn --ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth --save_rpn_feature --set TEST.SPLIT train_aug TEST.RPN_POST_NMS_TOP_N 300 TEST.RPN_NMS_THRESH 0.85
```

* To save features and proposals for the evaluation, we keep `TEST.RPN_POST_NMS_TOP_N=100` and `TEST.RPN_NMS_THRESH=0.8` as default:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --eval_mode rpn --ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth --save_rpn_feature
```
3. Now we could train our RCNN network. Note that you should modify `TRAIN.SPLIT=train_aug` to use the augmented scenes for the training, and use `--rcnn_training_roi_dir` and `--rcnn_training_feature_dir` to specify the saved features and proposals in the above step:
```
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn_offline --epochs 30  --ckpt_save_interval 1 --rcnn_training_roi_dir ../output/rpn/default/eval/epoch_200/train_aug/detections/data --rcnn_training_feature_dir ../output/rpn/default/eval/epoch_200/train_aug/features
```
For the offline GT sampling augmentation, the default setting to train the RCNN network is `RCNN.ROI_SAMPLE_JIT=True`, which means that we sample the RoIs and calculate their GTs in the GPU. I also provide the CPU version proposal sampling, which is implemented in the dataloader, and you could enable this feature by setting `RCNN.ROI_SAMPLE_JIT=False`. Typically the CPU version is faster but costs more CPU resources since they use mutiple workers.  

**For mutiple GPUs** simply add the `--mgpus` argument as above. Also can increase the `--batch_size` by using multiple GPUs for training.

