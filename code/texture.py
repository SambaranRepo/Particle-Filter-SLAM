'''
: Texture mapping for the Particle Filter SLAM
'''

import numpy as np
from glob import glob 
import os,sys
import pr2_utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, figure
from tqdm import tqdm
import pickle
import cv2
import yaml

sb = 475.143600050775 / 1000

robot_T_Lidar = np.asarray([[0.00130201, 0.796097, 0.605167, 0.8349], \
    [0.999999, -0.000419027, -0.00160026, -0.0126869],\
    [-0.00102038, 0.605169, -0.796097, 1.76416], \
    [0, 0, 0, 1]])

def meta_constructor(loader, node):
    value = loader.construct_mapping(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:opencv-matrix', meta_constructor)
left_camera = open("code/param/left_camera.yaml")
left_camera = yaml.load(left_camera, Loader = yaml.FullLoader)
right_camera = open("code/param/right_camera.yaml")
right_camera = yaml.load(right_camera, Loader = yaml.FullLoader)
# print(f"left camera yaml  \n : {left_camera}")
# print(f"right camera yaml \n: {right_camera}")

Ks = np.asarray(left_camera['camera_matrix']['data']).reshape(3,-1)
print(type(Ks))

K = np.asarray(left_camera['projection_matrix']['data']).reshape(3,-1)
print(type(K))

f = K[0][0] / Ks[0][0]
# print(f)

robot_T_stereo = np.asarray([[-0.00680499, -0.0153215, 0.99985, 1.64239],[ -0.999977, 0.000334627, -0.00680066, 0.247401],\
    [-0.000230383, -0.999883, -0.0153234,  1.58411],[0, 0, 0 ,1]])

with open('map_parameters.pkl', 'rb') as f: 
    X = pickle.load(f)
    MAP = X[0]
    
trajectory = MAP['traj']
map_image = MAP['image']

stereo_left_t = []
stereo_right_t = []
folder = "code/stereo_images/"

for file in os.listdir(folder +  'stereo_left'): 
    if file.endswith('.png'): 
        filename = file.split('.')[0]
        stereo_left_t.append(int(filename))

for file in os.listdir(folder +  'stereo_right'): 
    if file.endswith('.png'): 
        filename = file.split('.')[0]
        stereo_right_t.append(int(filename))
stereo_left_t.sort()
stereo_right_t.sort()
stereo_left_t = np.asarray(stereo_left_t).reshape(-1,1)
stereo_right_t = np.asarray_chkfinite(stereo_right_t).reshape(-1,1)
stereo_t = np.hstack((stereo_left_t, stereo_right_t))
stereo_t_diff = stereo_right_t - stereo_left_t
index = np.where(stereo_t_diff != 0)[0]
stereo_left_t = np.delete(stereo_left_t, index, axis = 0) 
stereo_right_t = np.delete(stereo_right_t, index, axis = 0) 

for i in tqdm(stereo_left_t): 
    d_img = pr2_utils.compute_stereo(int(i))
d_img = np.asarray(d_img)
print(d_img.shape)
print(d_img[0,250,350])
rgb_img = []
folder = 'code/stereo_images/stereo_left/'
for i in range(len(stereo_left_t)): 
    image = f'{int(stereo_left_t[i])}.png'
    rgb_img.append(cv2.imread(folder + image))
rgb_img = np.asarray(rgb_img)
print(rgb_img.shape)

def stereo_to_world(d_img, rgb_img): 
    for i in range(2): 
        # print(d_img[i])
        # print(type(f))
        z = (0.975 * sb)/d_img[i]
        # print(z.shape)
        uv = np.indices((560, 1280)).reshape(2,-1)
        print(uv.shape)


stereo_to_world(d_img, rgb_img)

#Implement texture mapping