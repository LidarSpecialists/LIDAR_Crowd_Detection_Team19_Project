#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:55:06 2021

@author: ian
"""
import os

a = open("/media/ian/1c18cc3e-0ef6-41a7-8918-23e94866183c/home/PointRCNN/data/KITTI/object/testing/testing.txt", "w")
for path, subdirs, files in os.walk(r'/media/ian/1c18cc3e-0ef6-41a7-8918-23e94866183c/home/PointRCNN/data/KITTI/object/testing/velodyne'):
   for filename in sorted(files):
      a.write(filename[0:6] + os.linesep)
      
