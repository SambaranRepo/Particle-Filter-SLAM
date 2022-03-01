'''
This program implements a Particle Filter Simultaneous Localization and Mapping (SLAM) for a robot moving 2D space and orientation in SO(2) space
'''

from distutils.command.build import build
import numpy as np
from glob import glob 
import os,sys
import pr2_utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, figure
from tqdm import tqdm
import pickle
import cv2
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('N', help='Provide the number of particles that you want to use for the particle filter', type = int)
parser.add_argument('noise', help='Enter the noise in the angular velocity', type=float)
args = parser.parse_args()
N = args.N
noise = args.noise

_, encoder_data = pr2_utils.read_data_from_csv('code/sensor_data/encoder.csv')
l = len(encoder_data)

dl,dr = 0.623479, 0.6228

robot_T_Lidar = np.asarray([[0.00130201, 0.796097, 0.605167, 0.8349], \
    [0.999999, -0.000419027, -0.00160026, -0.0126869],\
    [-0.00102038, 0.605169, -0.796097, 1.76416], \
    [0, 0, 0, 1]])

robot_T_FOG = np.asarray([[1,0,0,-0.335],[0,1,0,-0.035],[0, 0, 1, 0.78],[0, 0, 0, 1]])


class Slam():

    def __init__(self, mode, k = 2):
        '''
        : The map is initialized as a grid cell arrangement. 
        : One copy of the map maintain the log odds of each cell to create the occupancy map. 
        : Another copy creates the free space map based on log odds of each cell.
        : Total number of particles considered is 100. The threshold for resampling is kept at 20. 
        '''
        if mode == 1:
            self.k = k
            self.MAP = {}
            self.MAP['res']   = 0.75 #meters
            self.MAP['xmin']  = -500 
            self.MAP['ymin']  = -1500
            self.MAP['xmax']  =  1500
            self.MAP['ymax']  =  500 
            self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
            self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
            self.MAP['map'] = -np.ones((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8)
            self.MAP['log_odds'] = np.zeros((self.MAP['sizex'], self.MAP['sizey']))
            self.MAP['free'] = np.zeros((self.MAP['sizex'], self.MAP['sizey']), dtype = np.int8)
            self.MAP['pose'] = np.zeros((l,2))
            self.MAP['traj'] = np.zeros((l, 3))
            self.MAP['image'] = np.zeros((self.MAP['sizex'], self.MAP['sizey'],3))
            self.omega_noise = noise
            self.x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res'])
            self.y_im = np.arange(self.MAP['ymin'], self.MAP['ymax'] + self.MAP['res'], self.MAP['res'])
            self.x_max_range = 1.25
            self.y_max_range = 1.25
            self.x_range = np.arange(-self.x_max_range, self.x_max_range + self.MAP['res'], self.MAP['res'])
            self.y_range = np.arange(-self.y_max_range, self.y_max_range + self.MAP['res'], self.MAP['res'])
            self.x_mid = int(self.x_max_range / self.MAP['res'])
            self.y_mid = int(self.y_max_range / self.MAP['res'])
            self.N = N
            self.N_threshold = int(0.5 * N)
            self.mu = np.zeros((self.N, 3))
            self.alpha = np.full(self.N, 1 / self.N) 
            self.correlation = np.zeros(self.N,)
        elif mode == 2:
            self.N = N
            self.N_threshold = int(0.5 * N)
            with open("map_parameters.pkl", 'rb') as f: 
                X = pickle.load(f)
                self.MAP, self.mu, self.alpha = X[0], X[1], X[2]
        

    def first_scan(self, sx, sy, theta, ranges):
        '''
        Plot the first lidar scan
        '''
        MAP = {}
        MAP['res']   = 0.5 #meters
        MAP['xmin']  = -65
        MAP['ymin']  = -65
        MAP['xmax']  =  65
        MAP['ymax']  =  65 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = -np.ones((MAP['sizex'],MAP['sizey']),dtype=np.int8)
        MAP['log_odds'] = np.zeros((MAP['sizex'], MAP['sizey']))
        MAP['free'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype = np.int8)
        angles = np.linspace(-5,185,286) * np.pi/180
        valid_index = np.logical_and((ranges < 60),(ranges>=0.1))
        ranges = ranges[valid_index]
        angles = angles[valid_index]
        x_lidar = np.cos(angles) * ranges
        y_lidar = np.sin(angles) * ranges
        z_lidar = np.asarray([0]*len(x_lidar))
        s_lidar = np.stack((x_lidar, y_lidar, z_lidar, [1]*len(x_lidar)))
        w_s_obj = robot_T_Lidar @ s_lidar
        w_x_obj = w_s_obj[0]
        w_y_obj = w_s_obj[1]
        w_z_obj = w_s_obj[2]
        valid_z = np.logical_and((w_z_obj > 0.3), (w_z_obj < 4))

        #Convert from meters to cells
        x_cell = np.ceil((w_x_obj - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 
        y_cell = np.ceil((w_y_obj - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        sx_cell = np.ceil((sx - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 
        sy_cell = np.ceil((sy - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
          
        valid_x = np.logical_and((x_cell < MAP['sizex']), (x_cell >= 0))
        valid_y = np.logical_and((y_cell < MAP['sizey']), (y_cell >= 0))
        valid_xy = np.logical_and(valid_x, valid_y)
        valid_xyz = np.logical_and(valid_xy, valid_z)
        x_cell = x_cell[valid_xyz]
        y_cell = y_cell[valid_xyz]
        w_x_obj = w_x_obj[valid_xyz]
        w_y_obj = w_y_obj[valid_xyz]
        output = []
        for i in range(len(x_cell)): 
            output.append(pr2_utils.bresenham2D(sx_cell,sy_cell,x_cell[i],y_cell[i]).astype(np.int))
    
        for i in range(len(output)):
            index1 = (output[i][0][:-1])
            index2 = (output[i][1][:-1])
            MAP['log_odds'][index1, index2] -= np.log(4)
        MAP['log_odds'][x_cell, y_cell] += np.log(4)
        MAP['log_odds'][MAP['log_odds'] < -2] = -2
        MAP['log_odds'][MAP['log_odds'] > 2] = 2
        MAP['map'] = (MAP['log_odds'] > 0).astype(np.int8)
        MAP['free'] = (MAP['log_odds'] < 0).astype(np.int8)
        fig = plt.figure(figsize=(8,6))
        plt.imshow(MAP['free'].T, cmap = 'hot')
        plt.savefig("first_scan.eps", format = 'eps', bbox_inches = 'tight')
        # plt.show(block = True)
        plt.close()


    def build_map(self, sx, sy, theta, ranges): 
        '''
        : Use lidar scan and map log odds to build a map. 
        : Ray tracing of lidar scans is done by the Bresenham Rasterisation Algorithm. 
        : Log odds of cells that the lidar scans pass through are decreased by log4. 
        : Log odds corresponding to cells where the lidar scans end are increased by log4. 
        : Log odds values are capped bw -10 and 10. 
        : If a cell has log odds > 0, that cell is classfied as occuppied, otherwise it is a free cell. 

        '''
        w_T_robot = np.asarray([[np.cos(theta),-np.sin(theta), 0, sx], [np.sin(theta) , np.cos(theta) , 0 , sy], [0, 0, 1, 0], [0,0,0,1]])
        
        angles = np.linspace(-5,185,286) * np.pi/180
        valid_index = np.logical_and((ranges < 60),(ranges>=0.1))
        ranges = ranges[valid_index]
        angles = angles[valid_index]
        x_lidar = np.cos(angles) * ranges
        y_lidar = np.sin(angles) * ranges
        z_lidar = np.asarray([0]*len(x_lidar))
        s_lidar = np.stack((x_lidar, y_lidar, z_lidar, [1]*len(x_lidar)))
        w_s_obj = w_T_robot @ robot_T_Lidar @ s_lidar
        w_x_obj = w_s_obj[0]
        w_y_obj = w_s_obj[1]
        w_z_obj = w_s_obj[2]
        valid_z = np.logical_and((w_z_obj > 0.3), (w_z_obj < 4))

        #Convert from meters to cells
        x_cell = np.ceil((w_x_obj - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1 
        y_cell = np.ceil((w_y_obj - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        sx_cell = np.ceil((sx - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1 
        sy_cell = np.ceil((sy - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
          
        valid_x = np.logical_and((x_cell < self.MAP['sizex']), (x_cell >= 0))
        valid_y = np.logical_and((y_cell < self.MAP['sizey']), (y_cell >= 0))
        valid_xy = np.logical_and(valid_x, valid_y)
        valid_xyz = np.logical_and(valid_xy, valid_z)
        x_cell = x_cell[valid_xyz]
        y_cell = y_cell[valid_xyz]
        w_x_obj = w_x_obj[valid_xyz]
        w_y_obj = w_y_obj[valid_xyz]
        output = []
        for i in range(len(x_cell)): 
            output.append(pr2_utils.bresenham2D(sx_cell,sy_cell,x_cell[i],y_cell[i]).astype(np.int))
    
        for i in range(len(output)):
            index1 = (output[i][0][:-1])
            index2 = (output[i][1][:-1])
            self.MAP['log_odds'][index1, index2] -= np.log(4)
        self.MAP['log_odds'][x_cell, y_cell] += np.log(4)
        self.MAP['log_odds'][self.MAP['log_odds'] < -2] = -2
        self.MAP['log_odds'][self.MAP['log_odds'] > 2] = 2
        self.MAP['map'] = (self.MAP['log_odds'] > 0).astype(np.int8)
        self.MAP['free'] = (self.MAP['log_odds'] < 0).astype(np.int8)
        

    def predict_step(self,v, tau,omega): 
        '''
        Implement the bayes filter predict step using a differential drive model
        v ---> From encoder
        w ---> From FOG
        x_(t + 1) = x_t + [vcos(theta); vsin(theta); omega] * tau + gaussian noise
        Bayes filter applied as a particle filter
        N particles, resampling using stratified resampling
        '''
        np.random.seed(4)
        theta = self.mu[:,2]
        v = v + np.random.normal(0,0.5, self.N)
        omega = omega + np.random.normal(0, self.omega_noise, self.N)
        self.mu[:,0] += np.cos(theta) * v * tau  
        self.mu[:,1] += np.sin(theta) * v * tau 
        self.mu[:,2] += omega * tau 
        
    def update_step(self, lidar): 
        '''
        : Transform the lidar scan to world frame using the pose of each of the robot particle. 
        : Define a similarity criteria between the lidar scan obtained corresponding to the given particle and the current MAP. 
        : A high similarity means that the current particle is an accurate candidate of the actual robot pose. 
        : Each particle weight is multiplied by the obtained correlation between the transformed lidar scan and the map and they are normalized. 
        : This way we get the updated weights of each particle. 
        '''
        x,y,theta = self.mu[:,0], self.mu[:,1], self.mu[:,2]
        angles = np.linspace(-5, 185, 286) * np.pi/180
        valid_lidar = np.logical_and((lidar >= 0.1), (lidar < 60))
        ranges = lidar[valid_lidar]
        angles = angles[valid_lidar]
        dx = np.cos(angles) * ranges
        dy = np.sin(angles) * ranges
        dz = [0] * len(dx)
        ones = [1]*len(dx)
        s_lidar = np.asarray([dx, dy, dz,ones])
        s_lidar = robot_T_Lidar @ s_lidar
        s_lidar = s_lidar[:3]
        p = np.asarray([x,y,([0]*len(x))]).T[:,:,None]
        R = np.zeros((self.N, 3, 3))
        for i in range(self.N): 
            R[i] = np.asarray([[np.cos(theta[i]), -np.sin(theta[i]), 0 ],[np.sin(theta[i]), np.cos(theta[i]), 0],[0,0,1]])
        w_s_lidar = R @ s_lidar + p

        for i in range(self.N): 
            w_x_obj, w_y_obj, w_z_obj = w_s_lidar[i][0],w_s_lidar[i][1],w_s_lidar[i][2]
            valid_z = np.logical_and((w_z_obj < 2), (w_z_obj > 0.3))
            w_x_cell = np.ceil((w_x_obj - self.MAP['xmin'])/self.MAP['res']).astype(np.int16) - 1
            w_y_cell = np.ceil((w_y_obj - self.MAP['ymin'])/self.MAP['res']).astype(np.int16) - 1
            valid_x = np.logical_and((w_x_cell >= 0), (w_x_cell < self.MAP['sizex']))
            valid_y = np.logical_and((w_y_cell >= 0), (w_y_cell < self.MAP['sizey']))
            valid = np.logical_and(valid_x, valid_y)
            valid = np.logical_and(valid, valid_z)
            x_obj = w_x_obj[valid]
            y_obj = w_y_obj[valid]
            Y = np.stack((x_obj, y_obj))
            correlation_matrix = pr2_utils.mapCorrelation(self.MAP['map'],self.x_im, self.y_im, Y, self.x_range, self.y_range)
            index = np.unravel_index(np.argmax(correlation_matrix, axis = None), correlation_matrix.shape)
            self.correlation[i] = correlation_matrix[index]
            # self.mu[i,0] += (index[0] - self.x_mid)*self.MAP['res']
            # self.mu[i,1] += (index[1] - self.y_mid)*self.MAP['res']
        
    def resample(self): 
        '''
        Stratified Resampling algorithm of a particle filter
        '''
        mu = self.mu
        alpha = self.alpha
        mu_new = []
        alpha_new = np.full(self.N, 1/self.N)
        j, c = 0,alpha[0]
        for k in range(self.N): 
            u = np.random.uniform(0, 1/self.N)
            beta = u + k/ self.N
            while beta > c: 
                j,c = j+1, c + alpha[j-1]
            mu_new.append(mu[j-1])
        self.mu = np.asarray(mu_new)
        self.mu  = self.mu.reshape(self.N,3)
        self.alpha = alpha_new


    def slam(self):
        '''
        : Combine all functions defined above to implement a particle filter SLAM. 
        : Our motion model is described by the Differential Drive Model. 
        : Control inputs velocity, angular velocity are obtained from the encoder and the FOG sensor data provided in csv files. 
        : Observations are present in the form of lidar scans. 
        '''
        
        lidar_time, lidar_data = pr2_utils.read_data_from_csv('code/sensor_data/lidar.csv')
        encoder_time, encoder_data = pr2_utils.read_data_from_csv('code/sensor_data/encoder.csv')
        fog_time, fog_data = pr2_utils.read_data_from_csv('code/sensor_data/fog.csv')
        lidar_time, fog_time, encoder_time = lidar_time * (10**(-9)), fog_time * (10**(-9)), encoder_time*(10**(-9))
        k = self.k
        for count in tqdm(range(0,len(encoder_time), k)):
            if count == 0: 
                self.first_scan(0,0,0,lidar_data[count])
            else:
                t_n_encoder = encoder_time[count]
                t_p_encoder = encoder_time[count - k]
                t_n_fog = np.abs(t_n_encoder - fog_time).argmin()
                t_p_fog = np.abs(t_p_encoder - fog_time).argmin()
                zl_encoder = encoder_data[count][0] - encoder_data[count-k][0] 
                zr_encoder = encoder_data[count][1] - encoder_data[count-k][1] 
                tau = t_n_encoder - t_p_encoder
                vl = np.pi * dl * zl_encoder / (4096 * tau)
                vr = np.pi * dr * zr_encoder / (4096 * tau)
                v = (vl + vr) / 2
                
                w = np.sum(fog_data[t_p_fog:t_n_fog , 2]) / (fog_time[t_n_fog] - fog_time[t_p_fog])
                
                #Predict step
                self.predict_step(v, tau,  w)

                best_particle = np.argmax(self.alpha)
                
                best_mu = self.mu[best_particle, :]
                x_cell = np.ceil((best_mu[0] - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
                y_cell = np.ceil((best_mu[1] - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
                self.MAP['pose'][count-k:count,0] = x_cell
                self.MAP['pose'][count-k:count,1] = y_cell
                self.MAP['traj'][count-k:count,0]= best_mu[0]
                self.MAP['traj'][count-k:count,1] = best_mu[1]
                self.MAP['traj'][count-k:count,2] = best_mu[2]
                #Update step
                if count % 5 == 0:
                    t_n_lidar = min(count, len(lidar_time) - 1)
                    self.update_step(lidar_data[t_n_lidar])
                    self.alpha *= np.exp(self.correlation- np.max(self.correlation))
                    self.alpha /= np.sum(self.alpha)

                    best_particle = np.argmax(self.alpha)

                    best_mu = self.mu[best_particle, :]
                    
                    #Build a map based on lidar scan of best particle
                    self.build_map(best_mu[0], best_mu[1], best_mu[2], lidar_data[t_n_lidar])

                    #Resampling
                    N_eff = 1/(np.sum(self.alpha**2))
                    if N_eff < self.N_threshold: 
                        self.resample()
                    
                    
                    x_cell = np.ceil((best_mu[0] - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
                    y_cell = np.ceil((best_mu[1] - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
                    
                    self.MAP['pose'][count-k:count,0] = x_cell
                    self.MAP['pose'][count-k:count,1] = y_cell
                    self.MAP['traj'][count-k:count,0]= best_mu[0]
                    self.MAP['traj'][count-k:count,1] = best_mu[1]
                    self.MAP['traj'][count-k:count,2] = best_mu[2]
                    if count % 25000 == 0 : 
                        self.show_MAP(count)
                
        
        with open("map_parameters.pkl", 'wb') as f:
            pickle.dump([self.MAP,self.mu, self.alpha] , f)
            

    def show_MAP(self,count): 
        '''
        : Plot the occupancy map, free map and the robot trajectory. 
        '''
        fig1 = plt.figure(figsize=(8,6))
        plt.imshow(self.MAP['map'].T, cmap = "hot")
        plt.gca().invert_yaxis()
        plt.title("Occupancy map")
        plt.savefig(f"grid_map_{count}.png", format = 'png', bbox_inches = 'tight')
        arrow_properties = dict(
            facecolor="red", width=2.5,
            headwidth=8)
        plt.close()
        fig2 = plt.figure(figsize=(8,6))
        plt.scatter(self.MAP['pose'][:,0],self.MAP['pose'][:,1],marker='d', c = 'g',s = 0.001)
        plt.annotate('Start',c = 'white', fontsize = 'medium', xy = (self.MAP['pose'][1,0], self.MAP['pose'][1,1]), xytext=(self.MAP['pose'][1,0] - 200, self.MAP['pose'][1,1] + 150), arrowprops = arrow_properties)
        plt.annotate('Finish',c = 'white', fontsize = 'medium', xy = (self.MAP['pose'][-10,0], self.MAP['pose'][-10,1]), xytext=(self.MAP['pose'][-10,0] + 200, self.MAP['pose'][-10,1] + 300), arrowprops = arrow_properties)
        plt.imshow(~self.MAP['free'].T, cmap = "Greys")
        plt.gca().invert_yaxis()
        plt.title("Free space map")
        plt.savefig(f"map_{count}.png", format = 'png', bbox_inches = 'tight')
        plt.close()

if __name__ == '__main__':
    slam = Slam(mode =1, k = 3)
    slam.slam()
    slam.show_MAP(l) 
