# This code serves to transfer camera calibration parameters from Jianwei's .mat file format (used for his 3d reconstruction code)
# to my .npz file format (used for epipolar line alignment.)

from scipy.io import loadmat
import numpy as np
import calibrateCameras as ccam

filename = "camera_parameters.mat"

mat_vars = loadmat(filename)

dist = [0,0,0,0,0]
K = [0,0,0,0,0]
R = [0,0,0,0,0]
t = [0,0,0,0,0]
F = [0,0,0,0,0]

# Distortion Coefficients not given by Jianwei. Zeros written to file
dist[0] = np.zeros((1,5))
dist[1] = np.zeros((1,5))
dist[2] = np.zeros((1,5))
dist[3] = np.zeros((1,5))
dist[4] = np.zeros((1,5))

# Camera Intrinsic Matrices. 
K[0] = mat_vars['K1']
K[1] = mat_vars['K2']
K[2] = mat_vars['K3']
K[3] = mat_vars['K4']
K[4] = mat_vars['K5']

# Camera Rotation Matrices.
R[0] = mat_vars['Rt1']
R[1] = mat_vars['Rt2']
R[2] = mat_vars['Rt3']
R[3] = mat_vars['Rt4']
R[4] = mat_vars['Rt5']

# Camera Translation Matrices
t[0] = mat_vars['tt1']
t[1] = mat_vars['tt2']
t[2] = mat_vars['tt3']
t[3] = mat_vars['tt4']
t[4] = mat_vars['tt5']

# Fundamental Matrices. Only four given as fundamental matrices are relative to main camera view. 
F[0] = np.eye(3)
F[1] = mat_vars['F_21']
F[2] = mat_vars['F_31']
F[3] = mat_vars['F_41']
F[4] = mat_vars['F_51']


save_var = (dist,K,R,t,F)

ccam.saveCalibration(save_var, 'camera_parameters.npz')