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
import logging
import math
import pygame
import random
import queue
import numpy as np
from bounding_box import create_kitti_datapoint
from constants import *
import image_converter
from dataexport import *


""" OUTPUT FOLDER GENERATION """
PHASE = "Town00"
OUTPUT_FOLDER = os.path.join("_out", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes','velodyne_r','image_r','label_r','velodyne_rc','image_rc','label_rc','loc','label_C_2','label_C_r','label_C_rc']#,'velodyne_rc2','image_rc2','label_rc2']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')
LIDAR_R_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne_r/{0:06}.bin')
LABEL_R_PATH = os.path.join(OUTPUT_FOLDER, 'label_r/{0:06}.txt')
IMAGE_R_PATH = os.path.join(OUTPUT_FOLDER, 'image_r/{0:06}.png')
LIDAR_RC_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne_rc/{0:06}.bin')
LABEL_RC_PATH = os.path.join(OUTPUT_FOLDER, 'label_rc/{0:06}.txt')
IMAGE_RC_PATH = os.path.join(OUTPUT_FOLDER, 'image_rc/{0:06}.png')
LOC_PATH = os.path.join(OUTPUT_FOLDER, 'loc/{0:06}.txt')
LABEL_C_PATH = os.path.join(OUTPUT_FOLDER, 'label_C_2/{0:06}.txt')
LABEL_C_R_PATH = os.path.join(OUTPUT_FOLDER, 'label_C_r/{0:06}.txt')
LABEL_C_RC_PATH = os.path.join(OUTPUT_FOLDER, 'label_C_rc/{0:06}.txt')


class SynchronyModel(object):
    def __init__(self):
        self.world, self.init_setting, self.client, self.traffic_manager = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.non_player = []
        self.actor_list = []
        self.frame = None
        self.player = None
        self.road = None
        self.roadc = None
        self.captured_frame_no = self.current_captured_frame_num()
        self.sensors = []
        self._queues = []
        self.main_image = None
        self.depth_image = None
        self.point_cloud = None
        self.main_image_r = None
        self.depth_image_r = None
        self.point_cloud_r = None
        self.main_image_rc = None
        self.depth_image_rc = None
        self.point_cloud_rc = None
        self.loc_rot = None
        self.extrinsic = None
        self.extrinsic_r = None
        self.extrinsic_rc = None
        self.intrinsic, self.my_camera, self.rsu_camera, self.rsuc_camera = self._span_player()
        self._span_non_player()

    def __enter__(self):
        # set the sensor listener function
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def current_captured_frame_num(self):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(OUTPUT_FOLDER, 'label_2/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        answer = "A"
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def tick(self, timeout):
        # Drive the simulator to the next frame and get the data of the current frame
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                
                return data

    def __exit__(self, *args, **kwargs):
        # cover the world settings
        self.world.apply_settings(self.init_setting)

    def _make_setting(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(6.0)
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True) 
        traffic_manager.set_hybrid_physics_radius(100)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_boundaries_respawn_dormant_vehicles(21,70)
        # synchrony model and fixed time step
        init_setting = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 steps  per second
        settings.actor_active_distance = 100
        world.apply_settings(settings)
        return world, init_setting, client, traffic_manager

    def _span_player(self):
        """create our target vehicle"""
        my_vehicle_bp = self.blueprint_library.filter("model3")[0]
        my_vehicle_bp.set_attribute('role_name', 'hero' )
        location = carla.Location(190, 10, 0.5)
        rotation = carla.Rotation(0, 0, 0)
        transform_vehicle = carla.Transform(carla.Location(x=-48.642700, y=-19.353889, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.432304, roll=0.000000))
        my_vehicle = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
        my_vehicle.set_autopilot(True)
        k, my_camera, rsu_camera, rsuc_camera = self._span_sensor(my_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle
        return k, my_camera, rsu_camera, rsuc_camera#, rsuc2_camera

    def _span_sensor(self, player):
        """create camera, depth camera and lidar and attach to the target vehicle"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')

        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))

        camera_d_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_d_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))

        lidar_bp.set_attribute('range', '200')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '2')
        lidar_bp.set_attribute('lower_fov', '-24.8')
        lidar_bp.set_attribute('points_per_second', '2560000')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('horizontal_fov', '360')
        lidar_bp.set_attribute('dropoff_general_rate', '0.1')
        lidar_location = carla.Location(0, 0, 1.8)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)

        transform_sensor = carla.Transform(carla.Location(x=0, z=1.8))

        my_camera = self.world.spawn_actor(camera_bp, transform_sensor, attach_to=player,attachment_type=carla.AttachmentType.Rigid)
        my_camera_d = self.world.spawn_actor(camera_d_bp, transform_sensor, attach_to=player,attachment_type=carla.AttachmentType.Rigid)
        my_lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=player)

        rsu_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        rsu_camera_bp.set_attribute("image_size_x",str(WINDOW_WIDTH))
        rsu_camera_bp.set_attribute("image_size_y",str(WINDOW_HEIGHT))
        # camera relative position related to the vehicle
        rsu_camera_transform = carla.Transform(carla.Location(x=-65.000000, y=5.000000, z=4.000000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
        rsu_camera = self.world.spawn_actor(rsu_camera_bp, rsu_camera_transform)
        self.road = rsu_camera

        rsu_camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        rsu_camera_d_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        rsu_camera_d_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        rsu_camera_d_transform = carla.Transform(carla.Location(x=-65.000000, y=5.000000, z=4.000000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
        rsu_camera_d = self.world.spawn_actor(rsu_camera_d_bp, rsu_camera_d_transform)

        rsu_lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        rsu_lidar_bp.set_attribute('channels', str(64))
        rsu_lidar_bp.set_attribute('points_per_second', str(2560000))
        rsu_lidar_bp.set_attribute('rotation_frequency', str(20))
        rsu_lidar_bp.set_attribute('range', str(200))
        rsu_lidar_bp.set_attribute('dropoff_general_rate', '0.1')
        rsu_lidar_bp.set_attribute('horizontal_fov', '360')
        rsu_lidar_bp.set_attribute('upper_fov', str(0))
        rsu_lidar_bp.set_attribute('lower_fov', str(-40))
        rsu_lidar_location = carla.Location(-65, 5, 4)
        rsu_lidar_rotation = carla.Rotation(0, 0, 0)
        rsu_lidar_transform = carla.Transform(rsu_lidar_location, rsu_lidar_rotation)
        # spawn the lidar
        rsu_lidar = self.world.spawn_actor(rsu_lidar_bp, rsu_lidar_transform)
        
        rsuc_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        rsuc_camera_bp.set_attribute("image_size_x",str(WINDOW_WIDTH))
        rsuc_camera_bp.set_attribute("image_size_y",str(WINDOW_HEIGHT))
        # camera relative position related to the vehicle
        rsuc_camera_transform = carla.Transform(carla.Location(x=-36.000000, y=10.000000, z=4.000000), carla.Rotation(pitch=0.000000, yaw=135.000000, roll=0.000000))
        rsuc_camera = self.world.spawn_actor(rsuc_camera_bp, rsuc_camera_transform)
        self.roadc = rsuc_camera

        rsuc_camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        rsuc_camera_d_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        rsuc_camera_d_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        rsuc_camera_d_transform = carla.Transform(carla.Location(x=-36.000000, y=10.000000, z=4.000000), carla.Rotation(pitch=0.000000, yaw=135.000000, roll=0.000000))
        rsuc_camera_d = self.world.spawn_actor(rsuc_camera_d_bp, rsuc_camera_d_transform)

        rsuc_lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        rsuc_lidar_bp.set_attribute('channels', str(64))
        rsuc_lidar_bp.set_attribute('points_per_second', str(2560000))
        rsuc_lidar_bp.set_attribute('rotation_frequency', str(20))
        rsuc_lidar_bp.set_attribute('range', str(200))
        rsuc_lidar_bp.set_attribute('dropoff_general_rate', '0.1')
        rsuc_lidar_bp.set_attribute('horizontal_fov', '360')
        rsuc_lidar_bp.set_attribute('upper_fov', str(0))
        rsuc_lidar_bp.set_attribute('lower_fov', str(-40))
        rsuc_lidar_location = carla.Location(-36, 10, 4)
        rsuc_lidar_rotation = carla.Rotation(0, 135, 0)
        rsuc_lidar_transform = carla.Transform(rsuc_lidar_location, rsuc_lidar_rotation)
        # spawn the lidar
        rsuc_lidar = self.world.spawn_actor(rsuc_lidar_bp, rsuc_lidar_transform)
        
        self.actor_list.append(my_camera)
        self.actor_list.append(my_camera_d)
        self.actor_list.append(my_lidar)
        self.actor_list.append(rsu_camera)
        self.actor_list.append(rsu_camera_d)
        self.actor_list.append(rsu_lidar)
        self.actor_list.append(rsuc_camera)
        self.actor_list.append(rsuc_camera_d)
        self.actor_list.append(rsuc_lidar)
        self.sensors.append(my_camera)
        self.sensors.append(my_camera_d)
        self.sensors.append(my_lidar)
        self.sensors.append(rsu_camera)
        self.sensors.append(rsu_camera_d)
        self.sensors.append(rsu_lidar)
        self.sensors.append(rsuc_camera)
        self.sensors.append(rsuc_camera_d)
        self.sensors.append(rsuc_lidar)

        # camera intrinsic
        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k, my_camera,rsu_camera,rsuc_camera

    def _span_non_player(self):
        """create autonomous vehicles and people"""
        
        spawn_point_1 = carla.Transform(carla.Location(x=-48.642700, y=-8, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.432304, roll=0.000000))
        blueprint_1 = random.choice(self.blueprint_library.filter(FILTERV))
        if blueprint_1.has_attribute('color'):
        	color = random.choice(blueprint_1.get_attribute('color').recommended_values)
        	blueprint_1.set_attribute('color', color)
        if blueprint_1.has_attribute('driver_id'):
        	driver_id = random.choice(blueprint_1.get_attribute('driver_id').recommended_values)
        	blueprint_1.set_attribute('driver_id', driver_id)
        blueprint_1.set_attribute('role_name', 'autopilot')
        
        
        spawn_point_2=carla.Transform(carla.Location(x=-47.842700, y=0, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.432304, roll=0.000000))
        blueprint_2 = random.choice(self.blueprint_library.filter(FILTERV))
        if blueprint_2.has_attribute('color'):
        	color = random.choice(blueprint_2.get_attribute('color').recommended_values)
        	blueprint_2.set_attribute('color', color)
        if blueprint_2.has_attribute('driver_id'):
        	driver_id = random.choice(blueprint_2.get_attribute('driver_id').recommended_values)
        	blueprint_2.set_attribute('driver_id', driver_id)
        blueprint_2.set_attribute('role_name', 'autopilot')
        
        blueprints = self.world.get_blueprint_library().filter(FILTERV)
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        
        spawn_points_all = self.world.get_map().get_spawn_points()
        #number_of_spawn_points_all = len(spawn_points_all)
        spawn_points=[]
        for n, transform in enumerate(spawn_points_all):
            if (transform.location.x>-90) and (transform.location.x<-10) and (transform.location.y<60) and (transform.location.y>-30):
               spawn_points.append(spawn_points_all[n])
        number_of_spawn_points = len(spawn_points)
        if NUM_OF_VEHICLES <= number_of_spawn_points:
            random.shuffle(spawn_points)
            number_of_vehicles = NUM_OF_VEHICLES
        elif NUM_OF_VEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, NUM_OF_VEHICLES, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor

        batch = []
        batch.append(SpawnActor(blueprint_1, spawn_point_1))
        batch.append(SpawnActor(blueprint_2, spawn_point_2))
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot
            batch.append(SpawnActor(blueprint, transform))

        vehicles_id = []
        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_id.append(response.actor_id)
        vehicle_actors = self.world.get_actors(vehicles_id)
        self.non_player.extend(vehicle_actors)
        self.actor_list.extend(vehicle_actors)

        for i in vehicle_actors:
            i.set_autopilot(True, self.traffic_manager.get_port())

        blueprintsWalkers = self.world.get_blueprint_library().filter(FILTERW)
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(NUM_OF_WALKERS):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            while (loc.x>-30) or (loc.x<-80) or (loc.y<-20) or (loc.y>20):
                loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        walkers_list = []
        all_id = []
        walkers_id = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        for i in range(len(walkers_list)):
            walkers_id.append(walkers_list[i]["id"])
        walker_actors = self.world.get_actors(walkers_id)
        self.non_player.extend(walker_actors)
        self.actor_list.extend(all_actors)
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d walkers and %d vehicles, press Ctrl+C to exit.' % (len(walkers_id), len(vehicles_id)))

    def _save_training_files(self, datapoints, point_cloud, datapoints_r, point_cloud_r, datapoints_rc, point_cloud_rc,loc,datapoints_c,datapoints_c_r,datapoints_c_rc):#datapoints_rc2, point_cloud_rc2):
        """ Save data in Kitti dataset format """
        logging.info("Attempting to save at frame no {}, frame no: {}".format(self.frame, self.captured_frame_no))
        groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)
        kitti_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH.format(self.captured_frame_no)
        img_r_fname = IMAGE_R_PATH.format(self.captured_frame_no)
        lidar_r_fname = LIDAR_R_PATH.format(self.captured_frame_no)
        kitti_r_fname = LABEL_R_PATH.format(self.captured_frame_no)
        img_rc_fname = IMAGE_RC_PATH.format(self.captured_frame_no)
        lidar_rc_fname = LIDAR_RC_PATH.format(self.captured_frame_no)
        kitti_rc_fname = LABEL_RC_PATH.format(self.captured_frame_no)
        loc_fname = LOC_PATH.format(self.captured_frame_no)
        kitti_c_fname = LABEL_C_PATH.format(self.captured_frame_no)
        kitti_c_r_fname = LABEL_C_R_PATH.format(self.captured_frame_no)
        kitti_c_rc_fname = LABEL_C_RC_PATH.format(self.captured_frame_no)

        save_loc_data_0(loc_fname, loc)

        save_groundplanes(
            groundplane_fname, self.player, LIDAR_HEIGHT_POS)
        save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(
            img_fname, self.main_image)
        save_image_data(
            img_r_fname, self.main_image_r)
        save_image_data(
            img_rc_fname, self.main_image_rc)
        
        save_kitti_data(kitti_fname, datapoints)
        save_kitti_data(kitti_r_fname, datapoints_r)
        save_kitti_data(kitti_rc_fname, datapoints_rc)
        save_kitti_data(kitti_c_fname, datapoints_c)
        save_kitti_data(kitti_c_r_fname, datapoints_c_r)
        save_kitti_data(kitti_c_rc_fname, datapoints_c_rc)

        save_calibration_matrices(
            calib_filename, self.intrinsic, self.extrinsic,self.extrinsic_r,self.extrinsic_rc)
        save_lidar_data(lidar_fname, point_cloud)
        save_lidar_data(lidar_r_fname, point_cloud_r)
        save_lidar_data(lidar_rc_fname, point_cloud_rc)

    def generate_datapoints(self, image):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image """
        datapoints = []
        datapoints_c = []
        image = image.copy()
        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames
        for agent in self.non_player:
            if GEN_DATA:
                image, kitti_datapoint_c, kitti_datapoint = create_kitti_datapoint(
                    agent, self.intrinsic, self.extrinsic, image, self.depth_image, self.player, rotRP)
                if kitti_datapoint:
                    datapoints.append(kitti_datapoint)
                if kitti_datapoint_c:
                    datapoints_c.append(kitti_datapoint_c)

        return image, datapoints, datapoints_c

    def generate_datapoints_r(self, image):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image """
        datapoints = []
        datapoints_c = []
        image = image.copy()
        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames
        for agent in self.non_player:
            if GEN_DATA:
                image, kitti_datapoint_c, kitti_datapoint = create_kitti_datapoint(
                    agent, self.intrinsic, self.extrinsic_r, image, self.depth_image_r, self.road, rotRP)
                if kitti_datapoint:
                    datapoints.append(kitti_datapoint)
                if kitti_datapoint_c:
                    datapoints_c.append(kitti_datapoint_c)

        return image, datapoints, datapoints_c
        
    def generate_datapoints_rc(self, image):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image """
        datapoints = []
        datapoints_c = []
        image = image.copy()
        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames
        for agent in self.non_player:
            if GEN_DATA:
                image, kitti_datapoint_c, kitti_datapoint = create_kitti_datapoint(
                    agent, self.intrinsic, self.extrinsic_rc, image, self.depth_image_rc, self.roadc, rotRP)
                if kitti_datapoint:
                    datapoints.append(kitti_datapoint)
                if kitti_datapoint_c:
                    datapoints_c.append(kitti_datapoint_c)

        return image, datapoints, datapoints_c
    


def draw_image(surface, image, blend=False):
    array = image[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def main():

    pygame.init()
    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    with SynchronyModel() as sync_mode:
        try:
            step = 1
            stop_flag = 1
            while True:
                if should_quit():
                    break
                if stop_flag == 60:
                    break
                clock.tick()
                snapshot, sync_mode.main_image, sync_mode.depth_image, sync_mode.point_cloud, sync_mode.main_image_r, sync_mode.depth_image_r, sync_mode.point_cloud_r, sync_mode.main_image_rc, sync_mode.depth_image_rc, sync_mode.point_cloud_rc = sync_mode.tick(
                    timeout=2.0)
		
                sync_mode.loc_rot=sync_mode.player.get_transform()
                z=sync_mode.loc_rot.location.z#+1.8
                loc=[sync_mode.loc_rot.location.x,sync_mode.loc_rot.location.y,z,sync_mode.loc_rot.rotation.pitch,sync_mode.loc_rot.rotation.yaw,sync_mode.loc_rot.rotation.roll]      
                
                image = image_converter.to_rgb_array(sync_mode.main_image)
                image_r = image_converter.to_rgb_array(sync_mode.main_image_r)
                image_rc = image_converter.to_rgb_array(sync_mode.main_image_rc)
                sync_mode.extrinsic = np.mat(sync_mode.my_camera.get_transform().get_matrix())
                sync_mode.extrinsic_r = np.mat(sync_mode.rsu_camera.get_transform().get_matrix())
                sync_mode.extrinsic_rc = np.mat(sync_mode.rsuc_camera.get_transform().get_matrix())
                image, datapoints, datapoints_c = sync_mode.generate_datapoints(image)
                image_r, datapoints_r, datapoints_c_r = sync_mode.generate_datapoints_r(image_r)
                image_rc, datapoints_rc, datapoints_c_rc = sync_mode.generate_datapoints_rc(image_rc)

                if (datapoints or datapoints_r or datapoints_rc or datapoints_c or datapoints_c_r or datapoints_c_rc) and step % 10 == 0:
                    data = np.copy(np.frombuffer(sync_mode.point_cloud.raw_data, dtype=np.dtype('f4')))
                    data_r = np.copy(np.frombuffer(sync_mode.point_cloud_r.raw_data, dtype=np.dtype('f4')))
                    data_rc = np.copy(np.frombuffer(sync_mode.point_cloud_rc.raw_data, dtype=np.dtype('f4')))
                    data = np.reshape(data, (int(data.shape[0] / 4), 4))
                    data_r = np.reshape(data_r, (int(data_r.shape[0] / 4), 4))
                    data_rc = np.reshape(data_rc, (int(data_rc.shape[0] / 4), 4))
                    # Isolate the 3D data
                    points = data[:, :-1]
                    points_r = data_r[:, :-1]
                    points_rc = data_rc[:, :-1]
                    sync_mode._save_training_files(datapoints, points, datapoints_r, points_r, datapoints_rc, points_rc,loc,datapoints_c,datapoints_c_r,datapoints_c_rc)
                    sync_mode.captured_frame_no += 1
                    stop_flag += 1

                step = step+1
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                draw_image(display, image)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
        finally:
            print('destroying actors.')
            for actor in sync_mode.actor_list:
                actor.destroy()
            pygame.quit()
            print('done.')


if __name__ == '__main__':
    main()
