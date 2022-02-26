'''
: Texture mapping for the Particle Filter SLAM
'''

from tkinter import image_names
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
import time

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

f = float(K[0][0] / Ks[0][0])
print(type(f))

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
stereo_right_t = np.asarray(stereo_right_t).reshape(-1,1)
stereo_t = np.hstack((stereo_left_t, stereo_right_t))
stereo_t_diff = stereo_right_t - stereo_left_t
index = np.where(stereo_t_diff != 0)[0]
stereo_left_t = np.delete(stereo_left_t, index, axis = 0) 
stereo_right_t = np.delete(stereo_right_t, index, axis = 0) 

# for i in tqdm(range(len(stereo_left_t))): 
# for i in tqdm(range(200)): 
#     d_img = pr2_utils.compute_stereo(int(stereo_left_t[i]))
# d_img = np.asarray(d_img)
# # print(d_img.shape)
# # # print(d_img[0,250,350])

# rgb_img = []
# folder = 'code/stereo_images/stereo_left/'
# for i in range(200): 
#     image = f'{int(stereo_left_t[i])}.png'
#     img = (cv2.imread(folder + image,0))
#     img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     rgb_img.append(img)
# rgb_img = np.asarray(rgb_img)
# # print(rgb_img.shape)

# with open("texture_mapping_variable.pkl", 'wb') as f:
#     pickle.dump([d_img, rgb_img], f)

with open('texture_mapping_variable.pkl', 'rb') as f:
    X = pickle.load(f)
d_img = X[0]
rgb_img = X[1] 

lidar_time, lidar_data = pr2_utils.read_data_from_csv('code/sensor_data/lidar.csv')
encoder_time, encoder_data = pr2_utils.read_data_from_csv('code/sensor_data/encoder.csv')
fog_time, fog_data = pr2_utils.read_data_from_csv('code/sensor_data/fog.csv')
lidar_time, fog_time, encoder_time = lidar_time * (10**(-9)), fog_time * (10**(-9)), encoder_time*(10**(-9))


texture_map = np.zeros((MAP['sizex'],MAP['sizey'],3)).astype(np.int16)

def stereo_to_world(d_img, rgb_img): 
    for i in tqdm(range(200)): 
        print(stereo_left_t[i])
        z = (K[0][0] * sb)/d_img[i]
        d_img[i][d_img[i] > 4] = 0
        y = d_img[i].reshape(1,-1)
        depth = z.reshape(1,-1)
        depth[np.where(depth == float('inf'))] = 0
        # print(depth[0,20000:25000])
        uv = np.indices((1280, 560))
        uv = uv.reshape((2,-1))
        grid_uv = np.vstack((uv[1], uv[0]))
        grid = np.vstack((grid_uv, np.ones((1, 560*1280))))
        xyz_temp = y* (np.linalg.inv(K[0:3 , 0:3]) @ grid)
        xyz_temp = np.vstack((xyz_temp, np.ones((1, xyz_temp.shape[-1]))))
        xyz_w = robot_T_stereo @ xyz_temp

        #Uptill now we got the points of the camera frame to the robot frame. Now need to get the coordinates in world frame. 
        encoder_t = np.abs(int(stereo_left_t[i])*10**(-9) - encoder_time).argmin()
        index = encoder_t
        robot_pose = MAP['traj'][index]
        # print(f"encoder time near to stereo time : {encoder_t}")
        # print(f"current pose at this time : {robot_pose}")
        sx,sy,theta = robot_pose[0], robot_pose[1], robot_pose[2]
        # print(xyz_w[:, 20000:25000])
        world_T_robot = np.asarray([[np.cos(theta),-np.sin(theta), 0, sx], [np.sin(theta) , np.cos(theta) , 0 , sy], [0, 0, 1, 0], [0,0,0,1]])
        xyz_world = world_T_robot @ xyz_w
        # print(f"xyz_world : {xyz_world}")
        ground = np.logical_and((xyz_world[2] > 0),(xyz_world[2] < 0.5))
        xyz_world = xyz_world[:,ground]
        # print(f"xyz world : {xyz_world}")
        # time.sleep(10)

        optical_frame = (np.linalg.inv(world_T_robot @ robot_T_stereo) @ xyz_world)
        pixels = np.divide(K @ optical_frame, optical_frame[2])[:2,:]
        pixels = np.around(pixels).astype(np.int16)
        # print(pixels)
        # feasible_u = np.logical_and((pixels[0]>=0),(pixels[0]<=560))
        # feasible_v = np.logical_and((pixels[1]>=0),(pixels[1]<=1280))
        feasible_u = np.logical_and((pixels[0]>=0),(pixels[0]<=1280))
        feasible_v = np.logical_and((pixels[1]>=0),(pixels[1]<=560))
        feasible = np.logical_and(feasible_u, feasible_v)
        # print(f"feasuble u : {feasible_u}")
        # print(f"feasible v : {feasible_v}")
        # time.sleep(10)
        rgb_feasible = pixels[:,feasible]
        # print(f"rgb_feasible : {rgb_feasible}")
        xyz_ground = xyz_world[:,rgb_feasible]
        # print(rgb_img[i][feasible[0], feasible[1], :])
        x_ground = np.ceil((xyz_ground[0, :] - MAP['xmin'])/MAP['res']).astype(np.int16) - 1
        y_ground = np.ceil((xyz_ground[1,:] - MAP['ymin'])/ MAP['res']).astype(np.int16) - 1
        texture_map[x_ground, y_ground, :] = rgb_img[i][rgb_feasible[1], rgb_feasible[0], :]
            
    plt.imshow(texture_map)
    plt.show(block = True)
        
    return texture_map

texture_map = stereo_to_world(d_img, rgb_img)

