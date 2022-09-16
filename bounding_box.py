import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



from datadescriptor import KittiDescriptor
from camera_utils import *
from constants import WINDOW_HEIGHT, WINDOW_WIDTH, MAX_RENDER_DEPTH_IN_METERS, MIN_VISIBLE_VERTICES_FOR_RENDER, VISIBLE_VERTEX_COLOR, OCCLUDED_VERTEX_COLOR, MIN_BBOX_AREA_IN_PX
from utils import degrees_to_radians
import logging
from image_converter import depth_to_array
import numpy as np


def transform_points(transform, points):
    """
    Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
    """
    # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.

    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # Point transformation
    points = np.mat(transform.get_matrix()) * points
    # Return all but last row
    return points[0:3].transpose()


def bbox_2d_from_agent(agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP):  # rotRP expects point to be in Kitti lidar format
    """ Creates bounding boxes for a given agent and camera/world calibration matrices.
        Returns the modified image that contains the screen rendering with drawn on vertices from the agent """
    bbox = vertices_from_extension(ext)
    # transform the vertices respect to the bounding box transform
    bbox = transform_points(bbox_transform, bbox)
    # the bounding box transform is respect to the agents transform
    # so let's transform the points relative to it's transform
    bbox = transform_points(agent_transform, bbox)
    # agents's transform is relative to the world, so now,
    # bbox contains the 3D bounding box vertices relative to the world
    # Additionally, you can logging.info these vertices to check that is working
    # Store each vertex 2d points for drawing bounding boxes later
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def calculate_occlusion_stats(image, vertices_pos2d, depth_map, draw_vertices=True):
    """ Draws each vertex in vertices_pos2d if it is in front of the camera 
        The color is based on whether the object is occluded or not.
        Returns the number of visible vertices and the number of vertices outside the camera.
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # if the point is in front of the camera but not too far away
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), vertex_depth, depth_map)
            if is_occluded:
                vertex_color = OCCLUDED_VERTEX_COLOR
            else:
                num_visible_vertices += 1
                vertex_color = VISIBLE_VERTEX_COLOR
            if draw_vertices:
                draw_rect(image, (y_2d, x_2d), 4, vertex_color)
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def create_kitti_datapoint_ego(agent, intrinsic_mat, extrinsic_mat, image, depth_image, player, rotRP, draw_3D_bbox=True):
    """ Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to
    be labeled """
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent)

    if obj_type is None:
        logging.warning(
            "Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP)
    depth_map = depth_to_array(depth_image)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=draw_3D_bbox)

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat)
    x, y, z = [float(x) for x in midpoint][0:3]
    
    flag=0
    from math import pi
    datapoint_c = KittiDescriptor()
    if x*x <= 10000 and y*y < 1600:
        flag=1

        datapoint_c.set_bbox([0,0,0,0])
        rotation_y = get_relative_rotation_y(agent, player) % pi # 取余数
        datapoint_c.set_3d_object_dimensions(ext)
        datapoint_c.set_type(obj_type)
        datapoint_c.set_3d_object_location(midpoint)
        datapoint_c.set_rotation_y(rotation_y)
        if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:
            bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
            datapoint_c.set_bbox(bbox_2d)
        
    # At least N vertices has to be visible in order to draw bbox
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:
        flag=2
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_bounding_box(image, vertices_pos2d)
        
        rotation_y = get_relative_rotation_y(agent, player) % pi # 取余数

        datapoint = KittiDescriptor()
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(obj_type)
        datapoint.set_3d_object_location(midpoint)
        datapoint.set_rotation_y(rotation_y)
        
    if num_visible_vertices < MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:     
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_invisible_bounding_box(image, vertices_pos2d)
    
    if flag==2:
        return image, datapoint_c,datapoint
    elif flag==1:
        return image,datapoint_c, None
    else:
        return image, None, None
        
        
def create_kitti_datapoint_ego_invi(agent, intrinsic_mat, extrinsic_mat, image, depth_image, player, rotRP, draw_3D_bbox=True):
    """ Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to
    be labeled """
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent)

    if obj_type is None:
        logging.warning(
            "Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP)
    depth_map = depth_to_array(depth_image)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=False)

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat)
    x, y, z = [float(x) for x in midpoint][0:3]
    
    flag=0
    from math import pi
    datapoint_c = KittiDescriptor()
        
    if num_visible_vertices < MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:     
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_invisible_bounding_box(image, vertices_pos2d)
    return image

def create_kitti_datapoint(agent, intrinsic_mat, extrinsic_mat, image, depth_image, player, rotRP, draw_3D_bbox=True):
    """ Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to
    be labeled """
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent)

    if obj_type is None:
        logging.warning(
            "Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP)
    depth_map = depth_to_array(depth_image)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=draw_3D_bbox)

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat)
    x, y, z = [float(x) for x in midpoint][0:3]
    
    flag=0
    from math import pi
    datapoint_c = KittiDescriptor()
    if x*x <= 10000 and y*y < 1600:
        flag=1

        datapoint_c.set_bbox([0,0,0,0])
        rotation_y = get_relative_rotation_y(agent, player) % pi # 取余数
        datapoint_c.set_3d_object_dimensions(ext)
        datapoint_c.set_type(obj_type)
        datapoint_c.set_3d_object_location(midpoint)
        datapoint_c.set_rotation_y(rotation_y)
        if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:
            bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
            datapoint_c.set_bbox(bbox_2d)
        
    # At least N vertices has to be visible in order to draw bbox
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:
        flag=2
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_bounding_box(image, vertices_pos2d)
        
        rotation_y = get_relative_rotation_y(agent, player) % pi # 取余数

        datapoint = KittiDescriptor()
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(obj_type)
        datapoint.set_3d_object_location(midpoint)
        datapoint.set_rotation_y(rotation_y)
   
    
    if flag==2:
        return image, datapoint_c,datapoint
    elif flag==1:
        return image,datapoint_c, None
    else:
        return image, None, None

def create_kitti_datapoint_ori(agent, intrinsic_mat, extrinsic_mat, image, depth_image, player, rotRP, draw_3D_bbox=True):
    """ Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to
    be labeled """
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent)

    if obj_type is None:
        logging.warning(
            "Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP)
    depth_map = depth_to_array(depth_image)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=draw_3D_bbox)

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat)
    #x, y, z = [float(x) for x in midpoint][0:3]
    
    from math import pi
        
    # At least N vertices has to be visible in order to draw bbox
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_bounding_box(image, vertices_pos2d)
        
        rotation_y = get_relative_rotation_y(agent, player) % pi # 取余数

        datapoint = KittiDescriptor()
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(obj_type)
        datapoint.set_3d_object_location(midpoint)
        datapoint.set_rotation_y(rotation_y)
        
        return image,datapoint
    else:
        return image, None
        
def create_kitti_datapoint_cloud(agent, intrinsic_mat, extrinsic_mat, image, depth_image, player, rotRP, draw_3D_bbox=True):
    """ Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to
    be labeled """
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent)

    if obj_type is None:
        logging.warning(
            "Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP)
    depth_map = depth_to_array(depth_image)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=draw_3D_bbox)

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat)
    # At least N vertices has to be visible in order to draw bbox
    x, y, z = [float(x) for x in midpoint][0:3]
    
    if x*x <= 10000 and y*y < 1600:
        datapoint = KittiDescriptor()
        datapoint.set_bbox([0,0,0,0])
        if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < 4:
            bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
            area = calc_bbox2d_area(bbox_2d)
            if area < MIN_BBOX_AREA_IN_PX:
                logging.info("Filtered out bbox with too low area {}".format(area))
                return None
            
            datapoint.set_bbox(bbox_2d)
        from math import pi
        # xiu gai
        rotation_y = get_relative_rotation_y(agent, player) % pi # 取余数

        
        
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(obj_type)
        datapoint.set_3d_object_location(midpoint)
        datapoint.set_rotation_y(rotation_y)
        return datapoint
    else:
        return None


def get_relative_rotation_y(agent, player):
    """ Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""
    # We only car about the rotation for the classes we do detection on
    if agent.get_transform():
        rot_agent = agent.get_transform().rotation.yaw
        rot_car = player.get_transform().rotation.yaw
        return degrees_to_radians(rot_agent - rot_car)


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """ Accepts a bbox which is a list of 3d world coordinates and returns a list 
        of the 2d pixel coordinates of each vertex. 
        This is represented as a tuple (y, x, d) where y and x are the 2d pixel coordinates
        while d is the depth. The depth can be used for filtering visible vertices.
    """
    vertices_pos2d = []

    for vertex in bbox:
        pos_vector = vertex_to_world_vector(vertex)
        # Camera coordinates
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 2d pixel coordinates

        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # The actual rendered depth (may be wall or other object instead of vertex)
        vertex_depth = pos2d[2]
        # x_2d, y_2d = WINDOW_WIDTH - pos2d[0],  WINDOW_HEIGHT - pos2d[1]
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))


    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """ Returns the coordinates of the vector in correct carla world format (X,Y,Z,1) """
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def vertices_from_extension(ext):
    """ Extraxts the 8 bounding box vertices relative to (0,0,0)
    https://github.com/carla-simulator/carla/commits/master/Docs/img/vehicle_bounding_box.png 
    8 bounding box vertices relative to (0,0,0)
    """
    return np.array([
        [ext.x,   ext.y,   ext.z],  # Top left front
        [- ext.x,   ext.y,   ext.z],  # Top left back
        [ext.x, - ext.y,   ext.z],  # Top right front
        [- ext.x, - ext.y,   ext.z],  # Top right back
        [ext.x,   ext.y, - ext.z],  # Bottom left front
        [- ext.x,   ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])


def transforms_from_agent(agent):
    """ Returns the KITTI object type and transforms, locations and extension of the given agent """
    if agent.type_id.find('walker') != -1:
        obj_type = 'Pedestrian'
        agent_transform = agent.get_transform()
        bbox_transform = carla.Transform(agent.bounding_box.location, agent.bounding_box.rotation)
        ext = agent.bounding_box.extent
        agent_transform.location.z-=ext.z
        bbox_transform.location.z+=ext.z
        location = agent.get_transform().location
        location.z-=ext.z
        #print("walker")
        #print(bbox_transform)
    elif agent.type_id.find('vehicle') != -1:
        obj_type = 'Car'
        agent_transform = agent.get_transform()
        bbox_transform = carla.Transform(agent.bounding_box.location, agent.bounding_box.rotation)
        ext = agent.bounding_box.extent
        location = agent.get_transform().location
        #print("car")
        #print(bbox_transform)
    else:
        return (None, None, None, None, None)
    return obj_type, agent_transform, bbox_transform, ext, location


def calc_bbox2d_area(bbox_2d):
    """ Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple 
    """
    xmin, ymin, xmax, ymax = bbox_2d
    return (ymax - ymin) * (xmax - xmin)
