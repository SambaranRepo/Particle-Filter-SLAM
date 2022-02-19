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

_, lidar_data = pr2_utils.read_data_from_csv('code/sensor_data/lidar.csv')
l = len(lidar_data)

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
            self.MAP['res']   = 0.25 #meters
            self.MAP['xmin']  = -20 #meters
            self.MAP['ymin']  = -1500
            self.MAP['xmax']  =  1780
            self.MAP['ymax']  =  100 
            self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
            self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
            self.MAP['map'] = -np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8)
            self.MAP['log_odds'] = np.full((self.MAP['sizex'], self.MAP['sizey']), 0.5)
            self.MAP['free'] = np.zeros((self.MAP['sizex'], self.MAP['sizey']), dtype = np.int8)
            self.MAP['pose'] = np.zeros((l,2))
            self.N = 200
            self.N_threshold = 40
            self.mu = np.zeros((self.N, 3))
            self.alpha = np.full(self.N, 1 / self.N) 
            self.correlation = np.zeros((self.N, 1))
        elif mode == 2:
            self.N = 200
            self.N_threshold = 40
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
        valid_index = np.logical_and((ranges < 65),(ranges> 0.1))
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
        valid_z = w_z_obj > 0.5

        #Convert from meters to cells
        x_cell = np.ceil((w_x_obj - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1 
        y_cell = np.ceil((-w_y_obj + self.MAP['ymax']) / self.MAP['res'] ).astype(np.int16)-1
        sx_cell = np.ceil((sx - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1 
        sy_cell = np.ceil((-sy + self.MAP['ymax']) / self.MAP['res'] ).astype(np.int16)-1
        valid_x = np.logical_and((x_cell < self.MAP['sizex']), (x_cell >= 0))
        valid_y = np.logical_and((y_cell < self.MAP['sizey']), (y_cell >= 0))
        valid_xy = np.logical_and(valid_x, valid_y)
        valid_xyz = np.logical_and(valid_xy, valid_z)
        x_cell = x_cell[valid_xyz]
        y_cell = y_cell[valid_xyz]
        # sx_cell = sx_cell[np.logical_and((sx_cell < self.MAP['sizex']), (sx_cell >= 0))]
        # sy_cell = sy_cell[np.logical_and((sy_cell < self.MAP['sizey']), (sy_cell >= 0))]
        #Starting ray 
        output = []
        for i in range(len(x_cell)): 
            output.append(pr2_utils.bresenham2D(sx_cell,sy_cell,x_cell[i], y_cell[i]).astype(np.int))
        # print(f"Output 1 of bresenham : {len(output[100][0])}")
        # print(f"Output 2 of bresenham : {len(output[100][1])}")
        # print(f"length of bresenham output : {len(output)}")

        #Bresenham gives the cells that are free according to lidar scan
        #Number of rows = Number of valid lidar scans
        #each row contains 2 arrays ---> First is free cell along X axis, second is free cell along Y axis

        #Decrease the log odds of free cells by log4, increase the log odds of occupied cells by log4
        # print(f"Bresenham output : {output[:5]}")
        for i in range(len(output)):
            index1 = (output[i][0][:-1])
            index2 = (output[i][1][:-1])
            # print(f"index1 : {index1}")

            self.MAP['log_odds'][index2, index1] -= np.log(4)
            self.MAP['log_odds'][output[i][1][-1], output[i][0][-1]] += np.log(4)
        self.MAP['log_odds'][self.MAP['log_odds'] < -10] = -10
        self.MAP['log_odds'][self.MAP['log_odds'] > 10] = 10
        self.MAP['map'] = (self.MAP['log_odds'] > 0).astype(np.int8)
        self.MAP['free'] = (self.MAP['log_odds'] < 0).astype(np.int8)
        

    def predict_step(self, mu, v, tau, omega): 
        '''
        Implement the bayes filter predict step using a differential drive model
        v ---> From encoder
        w ---> From FOG
        x_(t + 1) = x_t + [vcos(theta); vsin(theta); omega] * tau + gaussian noise
        Bayes filter applied as a particle filter
        N particles, resampling using stratified resampling
        '''
        theta = self.mu[:,2]
        x = mu[:,0] + np.cos(theta) * v * tau + np.random.normal(0,0.04, self.N)
        y = mu[:,1] + np.sin(theta) * v * tau + np.random.normal(0,0.04, self.N)
        theta = mu[:,2] + omega * tau + np.random.normal(0,0.01, self.N)
        # print(f"Shape of x,y, theta : {x.shape, y.shape, theta.shape}")
        return x,y,theta

    def update_step(self, alpha, mu, lidar): 
        '''
        Update step for the particle filter
        Update alpha using map correlation
        '''
        x_max_range = 1.25
        y_max_range = 1.25
        x_range = np.arange(-x_max_range, x_max_range + self.MAP['res'], self.MAP['res'])
        y_range = np.arange(-y_max_range, y_max_range + self.MAP['res'], self.MAP['res'])
        x_mid = int(x_max_range / self.MAP['res'])
        y_mid = int(y_max_range/self.MAP['res'])
        x_p,y_p,theta_p = mu[:,0],mu[:,1],mu[:,2]
        angles = np.linspace(-5,185,286) * np.pi/180
        valid_index = np.logical_and((lidar < 65),(lidar> 0.1))
        # print(f"{np.sum(valid_index)}")
        ranges = lidar[valid_index]
        angles = angles[valid_index]
        x_lid = np.cos(angles) * ranges
        y_lid = np.sin(angles) * ranges
        # print(f"lidar x : {x_lid}")
        # print(f"Lidar y : {y_lid}")
        z_lid = np.asarray([0]*len(x_lid))
        s_lid = np.stack((x_lid, y_lid, z_lid, [1]*len(x_lid)))
        robot_s_obj = robot_T_Lidar @ s_lid

        x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map

        for i in range(len(alpha)):
           
            w_T_robot = np.asarray([[np.cos(theta_p[i]),-np.sin(theta_p[i]), 0, x_p[i]], [np.sin(theta_p[i]) , np.cos(theta_p[i]) , 0 , y_p[i]], [0, 0, 1, 0], [0,0,0,1]])
            
            w_s_obj = w_T_robot @ robot_s_obj
            w_x_obj = w_s_obj[0]
            w_y_obj = w_s_obj[1]
            print(f"w_x_obj : {len(w_x_obj)}")
            print(f"w_y_obj : {len(w_y_obj)}")
            # w_x_cell = np.ceil((w_x_obj - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
            # w_y_cell = np.ceil((w_y_obj - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
            # valid_x = np.logical_and((w_x_cell < self.MAP['sizex']),(w_x_cell >= 0))
            # valid_y = np.logical_and((w_y_cell < self.MAP['sizey']), (w_y_cell >=0))
            # valid = np.logical_and(valid_x , valid_y)
            # w_x_cell = w_x_cell[valid]
            # w_y_cell = w_y_cell[valid]
            # sx,sy = x_p[i], y_p[i]
            # sx_cell = np.ceil((sx - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1 
            # sy_cell = np.ceil((-sy + self.MAP['ymax']) / self.MAP['res'] ).astype(np.int16)-1 
            # self.temp_MAP = self.MAP['map'].copy()
            # output = []
            # for j in range(len(w_x_cell)): 
            #     output = (pr2_utils.bresenham2D(sx_cell,sy_cell,w_x_cell[j], w_y_cell[j]).astype(np.int))
            #     self.correlation[i] += np.sum(self.temp_MAP[output[1][:-1], output[0][:-1]] == 1)
            # # print(f"Correlation value at {i} : {self.correlation[i]}")
            Y = np.vstack((w_x_obj[0], w_y_obj[1]))
            correlation_matrix = pr2_utils.mapCorrelation(self.MAP['map'], x_im, y_im, Y, x_range, y_range)
            
            index = np.unravel_index(np.argmax(correlation_matrix, axis = None), correlation_matrix.shape)
            self.correlation[i] = correlation_matrix[index]
            print(f"correlation matrix : {correlation_matrix}")
        alpha = alpha * self.correlation
        alpha /= (np.sum(alpha) + 1e-4)
        # print(f"New alpha : {alpha}")
        # print(f"whats happening : {np.sum(alpha)}")
        return alpha, self.correlation

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
                j,c = j+1, c + alpha[j%500]
            mu_new.append(mu[j%500])
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
            else:
                # try:
                # t_n_lidar = lidar_time[count]
                # t_n_encoder = np.abs(encoder_time - t_n_lidar).argmin()
                # t_n_fog = np.abs(fog_time - t_n_lidar).argmin()
                # t_p_lidar = lidar_time[count-1]
                # t_p_encoder = np.abs(encoder_time - t_p_lidar).argmin()
                # t_p_fog = np.abs(fog_time - t_p_lidar).argmin()
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
                # w = robot_T_FOG[:3,:3] @ (fog_data[t_n_fog][2] - fog_data[t_p_fog][2])/(t_n_fog - t_p_fog)
                if t_n_fog < len(fog_time) - 15:
                    w = (robot_T_FOG[:3,:3] @np.asarray([0,0, np.sum(fog_data[t_n_fog : t_n_fog + 10][2])/(fog_time[t_n_fog + 10] - fog_time[t_n_fog])]))[-1]
                # print(f"Linear velocity : {v}, Angular velocity : {w}")
                #Predict step
                self.mu[:,0],self.mu[:,1],self.mu[:,2] = self.predict_step(self.mu, v, tau, w)

                #Update step
                if count % 15 == 0:
                    self.alpha,correlation = self.update_step(self.alpha, self.mu, lidar_data[t_n_lidar])
                    best_particle = np.argmax(correlation)
                    # print(f"Maximum correlation : {np.max(correlation)}")
                    best_mu = self.mu[best_particle, :]
                    # print(f"Best particles is : {best_mu}")
                    #Build a map based on lidar scan of best particle
                    self.build_map(best_mu[0], best_mu[1], best_mu[2], lidar_data[t_n_lidar])

                    #Resampling
                    N_eff = 1/(np.sum(self.alpha**2) + 1e-4)
                    if N_eff < self.N_threshold: 
                        # print(f"Resampling particles")
                        self.resample()
                    
                    x_cell = np.ceil((best_mu[0] - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
                    y_cell = np.ceil((-best_mu[1] + self.MAP['ymax']) / self.MAP['res'] ).astype(np.int16)-1
                    self.MAP['pose'][count][0] = x_cell
                    self.MAP['pose'][count][1] = y_cell 
                # print(f"trajectory of robot: {self.MAP['pose']}")
                if count % 25000 == 0 : 
                    with open("map_parameters.pkl", 'wb') as f:
                        pickle.dump([self.MAP,self.mu, self.alpha] , f)
                    self.show_MAP(count)
                
                # except Exception as e: 
                #     continue
    
    def show_MAP(self,count): 
        '''
        '''
        fig = plt.figure(figsize=(18,6))

        ax1 = fig.add_subplot(121)
        plt.imshow(self.MAP['map'],cmap="hot")
        plt.title("Occupancy map")

        ax2 = fig.add_subplot(122)
        plt.scatter(self.MAP['pose'][:,0],self.MAP['pose'][:,1],marker='o', c = 'g',s = 1)
        plt.imshow(self.MAP['free'],cmap="hot")
        plt.title("Free space map")
        plt.savefig(f"map_{count}.png", format = 'png')


        plt.show(block = True)






if __name__ == '__main__':
    slam = Slam(mode = 1)
    slam.slam()
    # slam.show_MAP()

#Implement resampling, select time based on motion 
