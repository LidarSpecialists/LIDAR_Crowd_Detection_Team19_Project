# Aim: To Augment 3D Benchmark Dataset and Train 3D Object Detection Model for Crowd Pedestrian detection for better accuracy

- GIT Repository (this repo):  https://github.com/LidarSpecialists/LIDAR_Crowd_Detection_Team19_Project

- System Demo - (Stage 1) - LIDAR Crowd Pedestrian Detection Demo  https://www.youtube.com/watch?v=vpy4CWgZYRM 

- System Demo - (Stage 2) - LIDAR Crowd Data Augmentation Demo   https://www.youtube.com/watch?v=ucfATfkJFZk 

- Presentation Link 


# Sponsor Company Intro

SenseTime (https://www.sensetime.com/en) is a leading global company focused on developing responsible AI technologies that advance the world’s economies, society and humanity for a better tomorrow.  They have made a number of technological breakthroughs, one of which is the first ever computer system in the world to achieve higher detection accuracy than the human eye. 

# Business Problem Background

SenseTime has sponsored us with this project to create a benchmark dataset for LIDAR crowd detection by introducing a novel data augmentation method for LiDAR-only learning problems that can greatly increase the convergence speed and performance. This will be done using computer vision and other deep learning tools.

# Augmented 3D Benchmark Dataset
Aim is to augment the dataset from KITTI[14] / Waymo Open Dataset[30] for Dense scenarios. A Pedestrian to be placed densely on a terrain or street without collision with other points in the 3D space. The optimal target will be to have 7000+ training point cloud frames per attribute value.

This augmented dataset will be used for Capstone projects and future R&D projects in SenseTime. A good benchmark dataset is a cornerstone of model training. With a good benchmark dataset, researchers can significantly reduce the time spent on data collection and labelling required before training a model. For example, MNIST is one of the most popular deep learning datasets for handwritten digits recognition and has been widely used by data scientists to train and test new architectures or frameworks. 

*Augmented Dataset Google DRive location* 
- https://drive.google.com/drive/folders/1-1zCORipbNB8GMjkjxuwxvJOFl0rlbcj?usp=sharing
- Size: 32 GB

# Standard Datasets
We used KITTI & Smantic KITTI datasets but could not leverage Waymo Open dataset because it is entirely in a different format than the KITTI and hard to convert to KITTI format as well.

# 3D Detection Baseline Model
Point RCNN - https://github.com/sshaoshuai/PointRCNN

# Other Important References 
VFNet - https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf
KITTI - http://www.cvlibs.net/datasets/kitti/


