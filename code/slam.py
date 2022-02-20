from distutils.command.build import build
import numpy as np
from glob import glob 
import os,sys
import pr2_utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, figure
from tqdm import tqdm
import pickle
import sympy
from sympy import symbols, pprint
from sympy import sin, cos
from sympy import Matrix
from sympy.utilities.lambdify import lambdify

_, encoder_data = pr2_utils.read_data_from_csv('code/sensor_data/encoder.csv')
l = len(encoder_data)

dl,dr = 0.623479, 0.6228

(x,y,theta, x_lidar, y_lidar) = symbols(""" x y theta x_lidar y_lidar""", real = True)
w_T_robot = Matrix([[cos(theta) , -sin(theta), 0 , x], [sin(theta), cos(theta), 0 , y], [0,0,1,0], [0,0,0,1]])
robot_T_lid = Matrix([[0.00130201, 0.796097, 0.605167, 0.8349], \
    [0.999999, -0.000419027, -0.00160026, -0.0126869],\
    [-0.00102038, 0.605169, -0.796097, 1.76416], \
    [0, 0, 0, 1]])
w_T_lid = w_T_robot * robot_T_lid
w_T_lid = sympy.simplify(w_T_lid)
lidar_s_obj = Matrix([[x_lidar], [y_lidar], [0],[1]])
w_s_object = w_T_lid * lidar_s_obj
w_s_object = sympy.simplify(w_s_object)
lidar_points = lambdify((x_lidar, y_lidar, x, y , theta),w_s_object )

robot_T_Lidar = np.asarray([[0.00130201, 0.796097, 0.605167, 0.8349], \
    [0.999999, -0.000419027, -0.00160026, -0.0126869],\
    [-0.00102038, 0.605169, -0.796097, 1.76416], \
    [0, 0, 0, 1]])

robot_T_FOG = np.asarray([[1,0,0,-0.335],[0,1,0,-0.035],[0, 0, 1, 0.78],[0, 0, 0, 1]])


class Slam():

    def __init__(self, mode):
        if mode == 1:
            self.MAP = {}
            self.MAP['res']   = 0.4 #meters
            self.MAP['xmin']  = -1500 #meters
            self.MAP['ymin']  = -1500
            self.MAP['xmax']  =  1500
            self.MAP['ymax']  =  300 
            self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
            self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
            self.MAP['map'] = -np.ones((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8)
            self.MAP['log_odds'] = np.zeros((self.MAP['sizex'], self.MAP['sizey']))
            self.MAP['free'] = np.zeros((self.MAP['sizex'], self.MAP['sizey']), dtype = np.int8)
            self.MAP['pose'] = np.zeros((l,2))
            self.x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res'])
            self.y_im = np.arange(self.MAP['ymin'], self.MAP['ymax'] + self.MAP['res'], self.MAP['res'])
            self.x_max_range = 0.8
            self.y_max_range = 0.8
            self.x_range = np.arange(-self.x_max_range, self.x_max_range + self.MAP['res'], self.MAP['res'])
            self.y_range = np.arange(-self.y_max_range, self.y_max_range + self.MAP['res'], self.MAP['res'])
            self.x_mid = int(self.x_max_range / self.MAP['res'])
            self.y_mid = int(self.y_max_range / self.MAP['res'])
            self.N = 300
            self.N_threshold = 100
            self.mu = np.zeros((self.N, 3))
            self.alpha = np.full(self.N, 1 / self.N) 
            self.correlation = np.zeros(self.N,)
        elif mode == 2:
            self.N = 300
            self.N_threshold = 60
            with open("map_parameters.pkl", 'rb') as f: 
                X = pickle.load(f)
                self.MAP, self.mu, self.alpha = X[0], X[1], X[2]
        


    def build_map(self, sx, sy, theta, ranges): 
        '''
        Use lidar scan and map log odds to build a map
        '''
        w_T_robot = np.asarray([[np.cos(theta),-np.sin(theta), 0, sx], [np.sin(theta) , np.cos(theta) , 0 , sy], [0, 0, 1, 0], [0,0,0,1]])

        #Initialised map cells all to -1 to indicate free cells

        # print(f"This is what we get from the lidar : {ranges}")

        #Build map based on first lidar scan

        #Get coordinates of points in lidar frame
        
        angles = np.linspace(-5,185,286) * np.pi/180
        valid_index = np.logical_and((ranges < 70),(ranges> 0.1))
        ranges = ranges[valid_index]
        angles = angles[valid_index]
        x_lidar = np.cos(angles) * ranges
        y_lidar = np.sin(angles) * ranges
        z_lidar = np.asarray([0]*len(x_lidar))
        s_lidar = np.stack((x_lidar, y_lidar, z_lidar, [1]*len(x_lidar)))
        #Coordinates of lidar scan in vehicle frame
        w_s_obj = w_T_robot @ robot_T_Lidar @ s_lidar
        # print(f"Vehicle frame lidar coordinates : {s_vehicle}")
        w_x_obj = w_s_obj[0]
        w_y_obj = w_s_obj[1]
        w_z_obj = w_s_obj[2]
        valid_z = w_z_obj >0.4

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
        output = []
        for i in range(len(x_cell)): 
            output.append(pr2_utils.bresenham2D(sx_cell,sy_cell,x_cell[i], y_cell[i]).astype(np.int))
       

        #Bresenham gives the cells that are free according to lidar scan
        #Number of rows = Number of valid lidar scans
        #each row contains 2 arrays ---> First is free cell along X axis, second is free cell along Y axis

        #Decrease the log odds of free cells by log4, increase the log odds of occupied cells by log4
        for i in range(len(output)):
            index1 = (output[i][0][:-1])
            index2 = (output[i][1][:-1])
            # print(f"index1 : {index1}")

            self.MAP['log_odds'][index1, index2] -= np.log(4)
            self.MAP['log_odds'][output[i][0][-1], output[i][1][-1]] += np.log(4)
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
        theta = self.mu[:,2]
        self.mu[:,0] += np.cos(theta) * v * tau + np.random.normal(0,0.2, self.N)
        self.mu[:,1] += np.sin(theta) * v * tau + np.random.normal(0,0.2, self.N)
        self.mu[:,2] += omega * tau + np.random.normal(0,0.05, self.N)
        # print(f"Shape of x,y, theta : {x.shape, y.shape, theta.shape}")

    def update_step(self, lidar): 
        '''
        Update step for the particle filter
        Update alpha using map correlation
        '''
        angles = np.linspace(-5,185, 286) * np.pi / 180
        valid_lidar = np.logical_and((lidar < 70), (lidar >=0.1))
        ranges = lidar[valid_lidar]
        angles = angles[valid_lidar]
        x_lidar = np.cos(angles) * ranges
        y_lidar = np.sin(angles) * ranges
        z_lidar = np.asarray([0]*len(x_lidar))
        s_lidar = np.stack((x_lidar,  y_lidar, z_lidar, [1]*len(x_lidar)))
        robot_s_lidar = robot_T_Lidar @ s_lidar
        for i in range(len(self.alpha)): 
            x,y,theta = self.mu[i,0], self.mu[i,1], self.mu[i,2]
            w_T_robot = np.asarray([[np.cos(theta), -np.sin(theta), 0 , x],[np.sin(theta), np.cos(theta), 0, y],[0,0,1,0],[0,0,0,1]])
            world_s_lidar = w_T_robot @ robot_s_lidar
            w_x_lidar = world_s_lidar[0]
            w_y_lidar = world_s_lidar[1]
            Y  = np.stack((w_x_lidar, w_y_lidar))
            correlation_matrix = pr2_utils.mapCorrelation(self.MAP['map'], self.x_im, self.y_im, Y, self.x_range, self.y_range)
            index = np.unravel_index(np.argmax(correlation_matrix, axis = None), correlation_matrix.shape)
            self.correlation[i] = correlation_matrix[index]
            self.mu[i,0] += (index[0] - self.x_mid)*self.MAP['res']
            self.mu[i,1] += (index[1] - self.y_mid)*self.MAP['res'] 
        self.alpha *= np.exp(self.correlation)
        self.alpha /= np.sum(self.alpha)

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
        '''
        lidar_time, lidar_data = pr2_utils.read_data_from_csv('code/sensor_data/lidar.csv')
        encoder_time, encoder_data = pr2_utils.read_data_from_csv('code/sensor_data/encoder.csv')
        fog_time, fog_data = pr2_utils.read_data_from_csv('code/sensor_data/fog.csv')
        lidar_time, fog_time, encoder_time = lidar_time * (10**(-9)), fog_time * (10**(-9)), encoder_time*(10**(-9))
        w = 0 #initial angular velocity
        for count in tqdm(range(len(lidar_time))):
        # for count in tqdm(range(20001)):
            if count == 0: 
                self.build_map(0,0,0,lidar_data[count])
                self.show_MAP(count)
            else:
                t_n_encoder = encoder_time[count]
                t_p_encoder = encoder_time[count-1]
                t_n_fog = np.abs(t_n_encoder - fog_time).argmin()
                t_p_fog = np.abs(t_p_encoder - fog_time).argmin()
                t_n_lidar = np.abs(t_n_encoder - lidar_time).argmin()
                zl_encoder = encoder_data[count][0] - encoder_data[count - 1][0] 
                zr_encoder = encoder_data[count][1] - encoder_data[count - 1][1] 
                tau = t_n_encoder - t_p_encoder
                # print(f"tau : {tau}")
                vl = np.pi * dl * zl_encoder / (4096 * tau)
                vr = np.pi * dr * zr_encoder / (4096 * tau)
                v = (vl + vr) / 2
                
                w = (robot_T_FOG[:3,:3] @np.asarray([0,0, np.sum(fog_data[t_n_fog:t_p_fog]) / (fog_time[t_n_fog] - fog_time[t_p_fog])]))[-1]
            
                # print(f"Linear velocity : {v}, Angular velocity : {w}")
                #Predict step
                self.predict_step(v, tau,  w)

                #Update step
                if count % 5 == 0:
                    self.update_step(lidar_data[t_n_lidar])
                    best_particle = np.argmax(self.correlation)
                    print(f"maximum correlation is {np.max(self.correlation)}")

                    best_mu = self.mu[best_particle, :]
                    print(f"Best particles is : {best_mu}")
                    #Build a map based on lidar scan of best particle
                    self.build_map(best_mu[0], best_mu[1], best_mu[2], lidar_data[t_n_lidar])

                    #Resampling
                    N_eff = 1/(np.sum(self.alpha**2) + 1e-4)
                    if N_eff < self.N_threshold: 
                        # print(f"Resampling particles")
                        self.resample()
                    
                    x_cell = np.ceil((best_mu[0] - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
                    y_cell = np.ceil((best_mu[1] - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
                    
                    self.MAP['pose'][count][0] = x_cell
                    self.MAP['pose'][count][1] = y_cell - (3750 - 2250)
                # print(f"trajectory of robot: {self.MAP['pose']}")
                if count % 500 == 0 : 
                    with open("map_parameters.pkl", 'wb') as f:
                        pickle.dump([self.MAP,self.mu, self.alpha] , f)
                    self.show_MAP(count)
                
             
    
    
    def shift_map(self): 
        temp_map = -np.ones((self.MAP['sizex'], self.MAP['sizey'])).astype(np.int8)          
        temp_free_map = np.zeros((self.MAP['sizex'], self.MAP['sizey'])).astype(np.int8)
        index = list(np.where(self.MAP['map'] == 1))
        index[1] -= (3750 - 2250)
        temp_map[index] = 1

        index = list(np.where(self.MAP['free'] == 1))
        index[1] -= (3750 - 2250)
        temp_free_map[index] = 1
        return temp_map, temp_free_map




    def show_MAP(self,count): 
        '''
        '''
        temp_map, temp_free_map = self.shift_map()

        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(121)
        plt.imshow(temp_map.T, cmap = "hot")
        plt.title("Occupancy map")

        ax2 = fig.add_subplot(122)
        
        plt.scatter(self.MAP['pose'][:,0],self.MAP['pose'][:,1],marker='o', c = 'g',s = 2)
        plt.imshow(temp_free_map.T, cmap = "hot")
        plt.title("Free space map")
        plt.savefig(f"map_{count}.png", format = 'png')
        plt.close()






if __name__ == '__main__':
    slam = Slam(mode = 1)
    slam.slam()
    slam.show_MAP(l - 1)

#Implement resampling, select time based on motion 
