
# STAGE-2 - 3D Crowd and Detection from Point Cloud

## 3D object detector
To directly generated accurate 3D box proposals from raw point cloud in a bottom-up manner, which are then refined in the canonical coordinate by the proposed bin-based 3D box regression loss. 
it is evaluated on the KITTI dataset and achieves a good performance on the KITTI 3D object detection

### Supported features and ToDo list
- [x] Multiple GPUs for training
- [x] GPU version rotated NMS
- [x] PyTorch 1.0
- [x] TensorboardX

# Hwardware recommendations
* 8 core CPU
* 32 GB ram and higher
* RTX 3060 Nvidia
* CUDA 10 drivers
* Ubuntu OS

## Installation Steps (Applicable for ubuntu 18.04 and 20)
### Software Requirements
All the codes are tested in the following environment:
* Linux 
* Python 3.6+
* PyTorch 1.0

### Install instructions 

a. Clone the PointRCNN repository.
```shell
git clone https://github.com/LidarSpecialists/LIDAR_Crowd_Detection_Team19_Project
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` .

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:

```shell
sh build_and_install.sh
```

## Dataset download distribution
Download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
You will have to register online for this.
```
PointRCNN
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```

## Pretrained model download Links
Pretrained model(Car) of PointRCNN from [here(~15MB)](https://drive.google.com/file/d/1aapMXBkSn5c5hNTDdRNI74Ptxfny7PuC/view?usp=sharing), which is trained on the *train* split (3712 samples) and evaluated on the *val* split (3769 samples) and *test* split (7518 samples). The performance on validation set is as follows:

### Quick execution demo
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt PointRCNN.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False
```

## Other modes of execution
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

All the codes supported **mutiple GPUs**, simply add the `--mgpus` argument as above. And you could also increase the `--batch_size` by using multiple GPUs for training.
