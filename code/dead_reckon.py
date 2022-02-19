'''
Dead reckoning 
'''

import numpy as np
from glob import glob 
import os,sys
import pr2_utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, figure
from tqdm import tqdm
import pickle
robot_T_FOG = np.asarray([[1,0,0,-0.335],[0,1,0,-0.035],[0, 0, 1, 0.78],[0, 0, 0, 1]])
dl,dr = 0.623479, 0.6228


def predict_step(mu, v, tau, omega, N = 1): 
    '''
    Implement the bayes filter predict step using a differential drive model
    v ---> From encoder
    w ---> From FOG
    x_(t + 1) = x_t + [vcos(theta); vsin(theta); omega] * tau + gaussian noise
    Bayes filter applied as a particle filter
    N particles, resampling using stratified resampling
    '''
    theta = mu[2]
    x = mu[0] + np.cos(theta) * v * tau
    y = mu[1] + np.sin(theta) * v * tau
    theta = mu[2] + omega * tau
    # print(f"Shape of x,y, theta : {x.shape, y.shape, theta.shape}")
    return x,y,theta

def slam(): 
    lidar_time, lidar_data = pr2_utils.read_data_from_csv('code/sensor_data/lidar.csv')
    encoder_time, encoder_data = pr2_utils.read_data_from_csv('code/sensor_data/encoder.csv')
    fog_time, fog_data = pr2_utils.read_data_from_csv('code/sensor_data/fog.csv')
    lidar_time, fog_time, encoder_time = lidar_time * (10**(-9)), fog_time * (10**(-9)), encoder_time*(10**(-9))
    w = 0 #initial angular velocity
    trajectory = [[], []]
    print(trajectory[0], trajectory[1])
    mu = np.zeros(3) #Initial hypothesis
    for count in tqdm(range(1,len(encoder_time))):
        
        t_n_encoder = encoder_time[count]
        t_p_encoder = encoder_time[count-1]
        t_n_fog = np.abs(t_n_encoder - fog_time).argmin()
        t_n_lidar = np.abs(t_n_encoder - lidar_time).argmin()
        zl_encoder = encoder_data[count][0] - encoder_data[count - 1][0] 
        zr_encoder = encoder_data[count][1] - encoder_data[count - 1][1] 
        tau = t_n_encoder - t_p_encoder
        # print(f"tau : {tau}")
        vl = np.pi * dl * zl_encoder / (4096 * tau)
        vr = np.pi * dr * zr_encoder / (4096 * tau)
        v = (vl + vr) / 2
        if t_n_fog < len(fog_time) - 15:
            w = (robot_T_FOG[:3,:3] @np.asarray([0,0, np.sum(fog_data[t_n_fog:t_n_fog + 10]) / (fog_time[t_n_fog + 10] - fog_time[t_n_fog])]))[-1]

        mu[0],mu[1],mu[2] = predict_step(mu, v, tau,w)
        # print(f"Updated mu : {mu}")
        trajectory[0].append((mu[0]))
        trajectory[1].append((mu[1]))
    # print(trajectory[0])
    show_trajectory(trajectory)

def show_trajectory(trajectory):
    plt.scatter(trajectory[1][1:], trajectory[0][1:], c = 'b') 
    plt.savefig('dead_reckon.jpg', format = 'jpg')
    plt.show(block = True)

if __name__ == '__main__':
    slam()

