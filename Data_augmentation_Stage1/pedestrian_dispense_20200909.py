# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 23:34:02 2021
successfully dispense people on the ground for frame 0000
successfully  amend to save the bounding box of argumented data note that the calibration file need to be change name after move to 
/home/capstone/kitti_object_vis/data/object/training/calib
drop with the consideration of alpha

edit on 14/04/2021 to let R0*T_v2c*(velo xyz)
@author: ian
"""
import numpy as np
import pandas as pd
import random
import ian_function
import math
from os.path import join, basename, splitext
from glob import glob
from shutil import copy2
import os

def create_dense(maxp,minp,semantic_pc_pathname,semantic_label_pathname,semantic_calib_pathname,out_pc_pathname,out_semanticlabel_pathname,out_objectlabel_pathname,out_calib_pathname):
    ###### parameter setting #######
    ###### parameter setting #######
    maxpedestrian=maxp
    minpedestrian=minp
    pedestrian_load_dir='/home/capstone/data/cropped_pedestrian4_pc'
    pedestrian_label_load_dir='/home/capstone/data/cropped_pedestrian4_label'
    #load semantic label(label) and semantic kittipoint cloud
#    filename='/home/capstone/data/dataset/sequences/00/labels/000001.label'
    filename=semantic_label_pathname
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))
#    point_cloud_load_pathname='/home/capstone/data/dataset/sequences/00/velodyne/000001.bin'
    point_cloud_load_pathname=semantic_pc_pathname
    point_cloud = np.fromfile(point_cloud_load_pathname, dtype=np.float32).reshape(-1, 4)
    #load original calibration file of the drop scene and get t_v2c
#    calib_pathname='/home/capstone/data/dataset/sequences/00/calib.txt'
    calib_pathname=semantic_calib_pathname
    ###### parameter setting #######
    ###### parameter setting #######

#    copy calib to output
    copy2(semantic_calib_pathname, out_calib_pathname)
    
    
    
    pc_terrain=ian_function.get_placement_df(point_cloud,label,40)
    with open(calib_pathname, 'r') as f:
        calibs = f.readlines()
    
    for calib in calibs:
        calib = calib.split()
        if len(calib) != 0 and calib[0] == 'Tr:':
            T_v2c = np.array([float(x) for x in calib[1:]]).reshape(3, 4)
            T_v2c = np.vstack([T_v2c, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)])
        if len(calib) != 0 and calib[0] == 'P0:':
            P0 = np.array([float(x) for x in calib[1:]]).reshape(3, 4)
            P0 = np.vstack([P0, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)])
    save_labels=[] # the label file for detection, full of pedestrian        
    ##for loop below logic for pedestrain drop--Done
    
    pedestrian_pathnames = glob(join(pedestrian_load_dir, '*.bin'))
    random.shuffle(pedestrian_pathnames)
    
    no_pedestrian=random.randint(minpedestrian,maxpedestrian)
    i=0
    for pedestrian_filename in pedestrian_pathnames:
        i+=1
        if len(pc_terrain)==0:
            break
        if i==no_pedestrian:
            break
        ##load pedestrian bounding box file and decide which zone should the pedestrian locate based on its original xyz
        
    #    pedestrian_labelfile=join(pedestrian_label_load_dir,pedestrian_filename[43:53], '*.txt'.format(i))
        pedestrian_labelfile=pedestrian_label_load_dir+'/'+pedestrian_filename[43:53]+'.txt'
    
        with open(pedestrian_labelfile, 'r') as f:
            pd_labels= f.readlines()
        for pd_label in pd_labels:
            cls, tru, occ, alp, x1, y1, x2, y2, h, w, l, x, y, z, ry = pd_label.split()
        pedestrian_zone=ian_function.get_pedestrian_zone(x,y)
#        print('pedestrian zone='+pedestrian_zone)
        if pc_terrain[pc_terrain['zone']==pedestrian_zone].empty:
            continue
        pc_terrain_select=pc_terrain[pc_terrain['zone']==pedestrian_zone].sample()
    
    # load pedestrain point cloud
    ##   pedestrian_filename='/home/capstone/data/cropped_pedestrian2/000011-001.bin'
        pc_pedestrian = np.fromfile(pedestrian_filename, dtype=np.float32).reshape(-1, 4)
        # randomly choose xyz of terrian pick point to pede
        x=pc_terrain_select.iloc[0].at['x']
        y=pc_terrain_select.iloc[0].at['y']
        z=pc_terrain_select.iloc[0].at['z']
        pc_pedestrian[:,0]+=x
        pc_pedestrian[:,1]+=y
        pc_pedestrian[:,2]+=z
        # add pedestrain dataset to pc and label30 to label file
        point_cloud=np.vstack([point_cloud,pc_pedestrian])
        ##pc_pedestrain=np.column_stack((pc_pedestrain,np.ones(( pc_pedestrain.shape[0],1))*30))
        # add label
        label_pedestrian=np.full(pc_pedestrian.shape[0],30)
        label=np.concatenate((label, label_pedestrian))
        # remove pedestrain poin cloud range to avoid overlap
        mask = (
            (pc_terrain['x'] >=x-1) *
            (pc_terrain['x'] <=x+1) *
            (pc_terrain['y'] >=y-1) *
            (pc_terrain['y'] <=y+1) 
        )
        pc_terrain = pc_terrain[~mask]
    
    #    mask = (
    #        (pc_terrain[:, 0] >=x-1) *
    #        (pc_terrain[:, 0] <=x+1) *
    #        (pc_terrain[:, 1] >=y-1) *
    #        (pc_terrain[:, 1] <=y+1) 
    #    )
    #    pc_terrain = pc_terrain[~mask,:]
    
    
        ## Add bounding box
    
        #xyz is the central point(velo) for new pedestrian; t_v*xyz to get new pedestrian xyz(camera)
    #    t_p=np.dot(P0,T_v2c)    
        t_p=T_v2c   
        t_c = np.dot(t_p , np.array([x, y, z, 1.0]).reshape(4, 1))
        x_c, y_c, z_c, _ = t_c.flatten().tolist()
        
        save_labels.append([cls, tru, occ, alp, 0, 0, 0, 0, h, w, l, round(x_c,2), round(y_c,2), round(z_c,2), ry])
       # exit if no place to drop patron    
        if len(pc_terrain)==0:
            break
        if i==no_pedestrian:
            break
    ##for loop above logic for pedestrain
    #save argumented point cloud and semantic label
    point_cloud.astype(np.float32).tofile(out_pc_pathname )  
    label.astype(np.uint32).tofile(out_semanticlabel_pathname) 
    
    
    #save object label
    object_labels_pathname =out_objectlabel_pathname     
    #with open(object_labels_pathname, 'w') as filehandle:
    #    for item in save_labels:
    ##        filehandle.write(str(item))
    #        filehandle.write(','.join(save_labels))
    #    filehandle.writelines("%s\n " % str(item) for item in save_labels)
    #    
    import csv
    with open(object_labels_pathname, 'w') as f:
        csv_writer = csv.writer(f,delimiter=" ")
        csv_writer.writerows(save_labels)


def main_process():        
    maxp=50
    minp=20
    no_scene=2
    semantic_dir='/home/capstone/data/dataset/sequences/00'
    out_dir='/media/ian/1c18cc3e-0ef6-41a7-8918-23e94866183c/home/augment_data/00'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    semantic_pc_pathnames = glob(join(semantic_dir+'/velodyne', '*.bin'))
    i=0
    for semantic_pc_pathname in semantic_pc_pathnames:
        i+=1
        semantic_pc_pathname      =semantic_pc_pathname
        semantic_label_pathname   =semantic_dir+'/labels/'+semantic_pc_pathname[50:56]+'.label'
        semantic_calib_pathname   =semantic_dir+'/calib.txt'
        out_pc_pathname           =out_dir+'/velodyne/'+semantic_pc_pathname[50:56]+'.bin'
        out_semanticlabel_pathname=out_dir+'/semantic_labels/'+semantic_pc_pathname[50:56]+'.label'
        out_objectlabel_pathname  =out_dir+'/object_labels/'+semantic_pc_pathname[50:56]+'.txt'
        out_calib_pathname        =out_dir+'/calib/'+semantic_pc_pathname[50:56]+'.txt'
        
        if not os.path.exists(out_dir+'/velodyne'):
            os.makedirs(out_dir+'/velodyne')
        if not os.path.exists(out_dir+'/semantic_labels'):
            os.makedirs(out_dir+'/semantic_labels')
        if not os.path.exists(out_dir+'/object_labels'):
            os.makedirs(out_dir+'/object_labels')
        if not os.path.exists(out_dir+'/calib'):
            os.makedirs(out_dir+'/calib')
        create_dense(maxp,minp,semantic_pc_pathname,semantic_label_pathname,semantic_calib_pathname,out_pc_pathname,out_semanticlabel_pathname,out_objectlabel_pathname,out_calib_pathname)
        
        
        if i>=no_scene:
            break
    
if __name__=='__main__':
    
    maxp=50
    minp=20
    no_scene=2
    semantic_dir='/home/capstone/data/dataset/sequences/00'
    out_dir='/media/ian/1c18cc3e-0ef6-41a7-8918-23e94866183c/home/augment_data/00'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    semantic_pc_pathnames = glob(join(semantic_dir+'/velodyne', '*.bin'))
    i=0
    for semantic_pc_pathname in semantic_pc_pathnames:
        i+=1
        semantic_pc_pathname      =semantic_pc_pathname
        semantic_label_pathname   =semantic_dir+'/labels/'+semantic_pc_pathname[50:56]+'.label'
        semantic_calib_pathname   =semantic_dir+'/calib.txt'
        out_pc_pathname           =out_dir+'/velodyne/'+semantic_pc_pathname[50:56]+'.bin'
        out_semanticlabel_pathname=out_dir+'/semantic_labels/'+semantic_pc_pathname[50:56]+'.label'
        out_objectlabel_pathname  =out_dir+'/object_labels/'+semantic_pc_pathname[50:56]+'.txt'
        out_calib_pathname        =out_dir+'/calib/'+semantic_pc_pathname[50:56]+'.txt'
        
        if not os.path.exists(out_dir+'/velodyne'):
            os.makedirs(out_dir+'/velodyne')
        if not os.path.exists(out_dir+'/semantic_labels'):
            os.makedirs(out_dir+'/semantic_labels')
        if not os.path.exists(out_dir+'/object_labels'):
            os.makedirs(out_dir+'/object_labels')
        if not os.path.exists(out_dir+'/calib'):
            os.makedirs(out_dir+'/calib')
        create_dense(maxp,minp,semantic_pc_pathname,semantic_label_pathname,semantic_calib_pathname,out_pc_pathname,out_semanticlabel_pathname,out_objectlabel_pathname,out_calib_pathname)
        print('complete scene='+str(i))
        
        if i>=no_scene:
            break