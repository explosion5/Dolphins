"""
This file contains all the methods responsible for saving the generated data in the correct output format.

"""

import numpy as np
import os
import sys
import glob
import logging
from utils import degrees_to_radians
import carla
import math
from numpy.linalg import pinv, inv
from PIL import Image


def save_groundplanes(planes_fname, player, lidar_height):
    from math import cos, sin
    """ Saves the groundplane vector of the current frame.
        The format of the ground plane file is first three lines describing the file (number of parameters).
        The next line is the three parameters of the normal vector, and the last is the height of the normal vector,
        which is the same as the distance to the camera in meters.
    """
    rotation = player.get_transform().rotation
    pitch, roll = rotation.pitch, rotation.roll
    # Since measurements are in degrees, convert to radians
    pitch = degrees_to_radians(pitch)
    roll = degrees_to_radians(roll)
    # Rotate normal vector (y) wrt. pitch and yaw
    normal_vector = [cos(pitch)*sin(roll),
                     -cos(pitch)*cos(roll),
                     sin(pitch)
                     ]
    normal_vector = map(str, normal_vector)
    with open(planes_fname, 'w') as f:
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        f.write("{} {}\n".format(" ".join(normal_vector), lidar_height))
    logging.info("Wrote plane data to %s", planes_fname)


def save_ref_files(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        logging.info("Wrote reference files to %s", path)


def save_image_data(filename, image):
    logging.info("Wrote image data to %s", filename)
    image.save_to_disk(filename)


def save_lidar_data(filename, point_cloud, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.
        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        lidar_array = [[point[0], -point[1], point[2], 1.0]
                       for point in point_cloud]
        lidar_array = np.array(lidar_array).astype(np.float32)

        logging.debug("Lidar min/max of x: {} {}".format(
                      lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of y: {} {}".format(
                      lidar_array[:, 1].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of z: {} {}".format(
                      lidar_array[:, 2].min(), lidar_array[:, 0].max()))
        lidar_array.tofile(filename)

    point_cloud.save_to_disk(filename)




def save_kitti_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)

def save_loc_data(filename, loc,loc_rc):
    loc=np.array(loc)
    loc_rc=np.array(loc_rc)
    ravel_mode = 'C'
    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))
    with open(filename, 'w') as f:
        write_flat(f, "ego_vehicle" , loc)
        write_flat(f, "other_vehicle" , loc_rc)
    logging.info("Wrote loc data to %s", filename)
    
def save_loc_data_0(filename, loc):
    loc=np.array(loc)
    ravel_mode = 'C'
    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))
    with open(filename, 'w') as f:
        write_flat(f, "ego_vehicle" , loc)
    logging.info("Wrote loc data to %s", filename)
    
def save_loc_data_2(filename, loc,loc_r,loc_rc):
    loc=np.array(loc)
    loc_r=np.array(loc_r)
    loc_rc=np.array(loc_rc)
    ravel_mode = 'C'
    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))
    with open(filename, 'w') as f:
        write_flat(f, "ego_vehicle" , loc)
        write_flat(f, "fore_vehicle" , loc_r)
        write_flat(f, "other_vehicle" , loc_rc)
    logging.info("Wrote loc data to %s", filename)

def save_calibration_matrices(filename, intrinsic_mat, rt_2,rt_r,rt_rc):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    f=1000
    k=[[f,0,960,0],[0,f,540,0],[0,0,1,0]]
    it=np.array(k)

    R_r=np.dot(inv(rt_2),rt_r)
    L_r=R_r
    R_rc=np.dot(inv(rt_2),rt_rc)
    L_rc=R_rc
    kitti_to_carla=np.mat([[0,0,1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
    R_r=np.dot(R_r,kitti_to_carla)
    R_rc=np.dot(R_rc,kitti_to_carla)
    carla_to_kitti=np.mat([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
    R_r=np.dot(carla_to_kitti,R_r)
    R_rc=np.dot(carla_to_kitti,R_rc)

    
    ravel_mode = 'C'
    P0 = it
    P1 = np.dot(it,R_r)
    P2 = np.dot(it,R_rc)
    
    R_2_to_r=np.dot(inv(rt_r),rt_2)
    R_2_to_r=np.dot(R_2_to_r,kitti_to_carla)
    R_2_to_r=np.dot(carla_to_kitti,R_2_to_r)
    P_2_to_r = np.dot(it,R_2_to_r)
    
    R_2_to_rc=np.dot(inv(rt_rc),rt_2)
    R_2_to_rc=np.dot(R_2_to_rc,kitti_to_carla)
    R_2_to_rc=np.dot(carla_to_kitti,R_2_to_rc)
    P_2_to_rc = np.dot(it,R_2_to_rc)
    

    R0 = np.identity(3)
    vel_to_world = np.mat([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    vel_ego_to_cam = np.mat([[0,-1,0,0],[0,0,-1,0],[1,0,0,0]])
    vel_to_kitti = np.mat([[0,1,0,0],[0,0,-1,0],[1,0,0,0]])
    vel_r_to_cam = np.dot(L_r,vel_to_world)
    vel_r_to_cam = np.dot(vel_to_kitti,vel_r_to_cam)
    vel_rc_to_cam = np.dot(L_rc,vel_to_world)
    vel_rc_to_cam = np.dot(vel_to_kitti,vel_rc_to_cam)

    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
         # Avod expects all 4 P-matrices even though we only use the first
        write_flat(f, "P0" , P0)
        write_flat(f, "P1" , np.array(P1))
        write_flat(f, "P2" , np.array(P2))
        write_flat(f, "P3" , P0)
        write_flat(f, "Pc-r" , np.array(P_2_to_r))
        write_flat(f, "Pc-rc" , np.array(P_2_to_rc))
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", np.array(vel_ego_to_cam))
        write_flat(f, "Tr_velo_r_to_cam", np.array(vel_r_to_cam))
        write_flat(f, "Tr_velo_rc_to_cam", np.array(vel_rc_to_cam))
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)