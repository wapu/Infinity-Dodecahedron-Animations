import numpy as np
from math import sin, cos, pi


# Transformation matrices
def x_rot(t):
    return np.matrix(
        [[1, 0, 0, 0],
         [0, cos(t), -sin(t), 0],
         [0, sin(t), cos(t), 0],
         [0, 0, 0, 1]])

def y_rot(t):
    return np.matrix(
        [[cos(t), 0, sin(t), 0],
         [0, 1, 0, 0],
         [-sin(t), 0, cos(t), 0],
         [0, 0, 0, 1]])

def z_rot(t):
    return np.matrix(
        [[cos(t), -sin(t), 0, 0],
         [sin(t), cos(t), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

def translation(x,y,z):
    return np.matrix(
        [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]])

# Camera matrix
def get_camera_matrix(theta_x=0, theta_y=0, theta_z=0, x=0, y=0, z=1):
    camera_to_world = z_rot(theta_z) * y_rot(theta_y) * x_rot(theta_x) * translation(x,y,z)
    return np.linalg.inv(camera_to_world)[:3]

# 3D to screen coordinates
def project(camera_matrix, points, screen_size):
    points = np.concatenate([points, np.ones((len(points),1))], axis=1).T
    points = np.array(camera_matrix * points)
    z = points[2,:]
    points = points / -z
    center = np.array(screen_size)/2
    points = points[:2,:].T * np.array([1, -1]) * center + center
    return points, -z
