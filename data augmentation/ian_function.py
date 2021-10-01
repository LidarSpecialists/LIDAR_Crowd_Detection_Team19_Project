# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:25:25 2021

andauthor: ian
"""

import numpy as np
import pandas as pd
import math

def get_placement_df(point_cloud,label,drop_type):
    #point_cloud file in np array
    #label in np array
    #drop_type:which place want to drop pedestrian,refering to semantic kitti lable. for example drop_type=40
    d=np.sqrt(point_cloud[:,0]**2+point_cloud[:,1]**2)
    pc_semantic_label=np.column_stack((point_cloud,label,d))
    mask = pc_semantic_label[:, 4] == drop_type
    pc_terrain=pc_semantic_label[mask, :]
    
    df_pc=pd.DataFrame(pc_terrain, columns=['x','y','z','i','l','d']) 
    # Need to add lable of zone to the terrain potential drop position
    
    con_zone=[(df_pc['x']>=0) & (df_pc['y']>=0) & (df_pc['d']<=20),
            (df_pc['x']>=0) & (df_pc['y']>=0) & (df_pc['d']>20),
            (df_pc['x']>=0) & (df_pc['y']<0) & (df_pc['d']<=20),
            (df_pc['x']>=0) & (df_pc['y']<0) & (df_pc['d']>20),
            (df_pc['x']<0) & (df_pc['y']<0) & (df_pc['d']<=20),
            (df_pc['x']<0) & (df_pc['y']<0) & (df_pc['d']>20),
            (df_pc['x']<0) & (df_pc['y']>=0) & (df_pc['d']<=20),
            (df_pc['x']<0) & (df_pc['y']>=0) & (df_pc['d']>20)
            ]
    df_pc['zone']=np.select(con_zone,['A1','A2','B1','B2','C1','C2','D1','D2'],default=None)
    return df_pc


#determine the zone of this pedestrian originally
#    D2     A2
#      D1 A1
#      C1 B1
#    C2     B2  
def get_pedestrian_zone(x,y):
# x,y in velodyne coordinate after transforemation, it was transformed and stored in label file after processed with kitti_crop_20210615
    x=float(x)
    y=float(y)    
    d=math.sqrt(x**2+y**2)
    if (x>=0) & (y>=0) & (d<=20):
        pedestrian_zone = 'A1'
    elif(x>=0) & (y>=0) & (d>20):
        pedestrian_zone = 'A2'
    elif(x>0) & (y<0) & (d<=20):
        pedestrian_zone = 'B1'
    elif(x>0) & (y<0) & (d>20):
        pedestrian_zone ='B2'
    elif(x<0) & (y<0) & (d<=20):
        pedestrian_zone = 'C1'
    elif(x<0) & (y<0) & (d>20):
        pedestrian_zone = 'C2'      
    elif(x<0) & (y>0) & (d<=20):
        pedestrian_zone = 'D1'
    elif(x<0) & (y>0) & (d>20):
        pedestrian_zone = 'D2'  
    else:
        pedestrian_zone = math.nan
    return pedestrian_zone
    