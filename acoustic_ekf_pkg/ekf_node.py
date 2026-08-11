#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geographic_msgs.msg import GeoPoint
import numpy as np
import threading
import time
import yaml
import os
import math
import utm
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Vector3Stamped
import tf_transformations


class EKF:
    def __init__(self, initial_state, Q=None, R=None, leader_velocities=False, dt=0, max_velocity=2.0):
        self.dt = np.float32(dt)
        self.Q = Q if Q is not None else np.eye(12, dtype=np.float32) * np.float32(0.01)
        self.leader_velocities = leader_velocities
        self.max_velocity = max_velocity  # Maximum allowed velocity (m/s)
        
        if self.leader_velocities:
            self.R = R if R is not None else np.eye(5, dtype=np.float32) * np.float32(0.05)
        else:
            self.R = R if R is not None else np.eye(3, dtype=np.float32) * np.float32(0.05)
            
    # State vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, vfx, vfy]
        self.x = initial_state.reshape((12, 1)).astype(np.float32)
        self.previous_x = self.x.copy()
        self.P = np.eye(12, dtype=np.float32) * 2

    def _clip_velocities(self):
        # Clip leader and follower velocities
        # Leader 1: vx1 (2), vy1 (3); Leader 2: vx2 (6), vy2 (7); Follower: vf (11)
        self.x[2] = np.clip(self.x[2], -self.max_velocity, self.max_velocity)
        self.x[3] = np.clip(self.x[3], -self.max_velocity, self.max_velocity)
        self.x[6] = np.clip(self.x[6], -self.max_velocity, self.max_velocity)
        self.x[7] = np.clip(self.x[7], -self.max_velocity, self.max_velocity)
        self.x[10] = np.clip(self.x[10], -self.max_velocity, self.max_velocity)
        self.x[11] = np.clip(self.x[11], -self.max_velocity, self.max_velocity)  # Follower speed should be >= 0

    def predict(self):
        # Store previous state
        self.previous_x = self.x.copy()
        
        # Extract state variables
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, vfx, vfy = self.x.flatten()
        
        # State transition matrix F
        F = np.array([
            [1, 0, self.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, self.dt, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, self.dt, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, self.dt, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Predict next state using non-linear model
        self.x = np.array([
            [x1 + vx1 * self.dt],
            [y1 + vy1 * self.dt],
            [vx1],
            [vy1],
            [x2 + vx2 * self.dt],
            [y2 + vy2 * self.dt],
            [vx2],
            [vy2],
            [xf + vfx * self.dt],
            [yf + vfy * self.dt],
            [vfx],
            [vfy]
        ], dtype=np.float32)
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        self._clip_velocities()  # <-- Add this line
        return self.x.flatten(), self.P

    def update(self, z, leader_id):
        # Measurement model Jacobian
        if self.leader_velocities:
            H = np.zeros((5, 12), dtype=np.float32)
        else:
            H = np.zeros((3, 12), dtype=np.float32)
        
        # Set position measurement gradients
        H[0, leader_id * 4] = np.float32(1)      # x position
        H[1, leader_id * 4 + 1] = np.float32(1)  # y position
        
        # Distance measurement gradients
        flat_x = self.x.flatten()
        dist = np.linalg.norm(flat_x[leader_id * 4:leader_id * 4 + 2] - flat_x[8:10])
        
        if self.leader_velocities:
            # Velocity measurements
            H[2, leader_id * 4 + 2] = np.float32(1)  # vx
            H[3, leader_id * 4 + 3] = np.float32(1)  # vy
            # Distance gradients
            H[4, leader_id * 4] = np.float32((flat_x[leader_id * 4] - flat_x[8]) / dist)
            H[4, leader_id * 4 + 1] = np.float32((flat_x[leader_id * 4 + 1] - flat_x[9]) / dist)
            H[4, 8] = np.float32((flat_x[8] - flat_x[leader_id * 4]) / dist)
            H[4, 9] = np.float32((flat_x[9] - flat_x[leader_id * 4 + 1]) / dist)
        else:
            # Distance gradients
            H[2, leader_id * 4] = np.float32((flat_x[leader_id * 4] - flat_x[8]) / dist)
            H[2, leader_id * 4 + 1] = np.float32((flat_x[leader_id * 4 + 1] - flat_x[9]) / dist)
            H[2, 8] = np.float32((flat_x[8] - flat_x[leader_id * 4]) / dist)
            H[2, 9] = np.float32((flat_x[9] - flat_x[leader_id * 4 + 1]) / dist)
        
        # Kalman update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        z_hat = self.get_expected_measurement(leader_id)
        y = z.astype(np.float32) - z_hat.astype(np.float32)
        y = y.reshape(-1, 1)
        
        self.x = self.x + K @ y
        I = np.eye(12, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        self._clip_velocities()  # <-- Add this line
        return self.x.flatten()

    def get_expected_measurement(self, leader_id):
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, vfx, vfy = self.x.flatten()
        
        if leader_id == 0:  # Leader 1 (first leader in EKF state)
            if self.leader_velocities:
                expected_msmt = np.array([x1, y1, vx1, vy1, np.linalg.norm(self.x[0:2] - self.x[8:10])])
            else:
                expected_msmt = np.array([x1, y1, np.linalg.norm(self.x[0:2] - self.x[8:10])])
        elif leader_id == 1:  # Leader 2 (second leader in EKF state)
            if self.leader_velocities:
                expected_msmt = np.array([x2, y2, vx2, vy2, np.linalg.norm(self.x[4:6] - self.x[8:10])])
            else:
                expected_msmt = np.array([x2, y2, np.linalg.norm(self.x[4:6] - self.x[8:10])])
        
        return expected_msmt.flatten()

    def get_state(self):
        return self.x.flatten()

    def get_covariance(self):
        return self.P


class AcousticEKFNode(Node):
    def __init__(self):
        super().__init__('acoustic_ekf_node')
        
        # Declare parameters
        self.declare_parameter('config_file', 'ekf_config.yaml')
        self.declare_parameter('follower_ns', '')
        # use_sim_time is often pre-declared by launch; default live=false.
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        
        # Get parameters
        config_file = self.get_parameter('config_file').value
        follower_ns = self.get_parameter('follower_ns').value
        
        # Load EKF config from specified file
        config_path = os.path.join(os.path.dirname(__file__), '../config', config_file)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Configuration parameters
        process_noise = config.get('process_noise', 0.01)
        measurement_noise = config.get('measurement_noise', 0.05)
        distance_measurement_noise = config.get('distance_measurement_noise', 0.1)  # Separate noise for distance
        self.process_noise = process_noise  # store for dynamic Q updates
        self.leader_init_samples = config.get('leader_init_samples', 10)
        self.follower_init_samples = config.get('follower_init_samples', 3)
        self.max_speed = config.get('max_speed', 2.0)
        self.dt = 0  # config.get('dt', 0.1)  # Read dt from config
        self.correction_factor = config.get('correction_factor', 1.0)  # Correction factor for distance measurements

        # Boat (antenna / GPS vs transducer) offset in UTM meters [dx, dy]; applied to leader positions
        self.boat_offset = np.array(config.get('boat_offset', [0.0, -0.6]), dtype=np.float32)
        
        # Modem to baselink offset parameters (for real-time transform)
        # The EKF estimates modem position, we transform back to baselink for publishing
        self.modem_to_baselink_offset_x = config.get('modem_to_baselink_offset_x', 1.96)  # Forward offset (m)
        self.modem_to_baselink_offset_y = config.get('modem_to_baselink_offset_y', 0.0)   # Lateral offset (m) 
        self.modem_to_baselink_offset_z = config.get('modem_to_baselink_offset_z', 0.475) # Vertical offset (m)
        
        # Heading update related parameters
        self.heading_measurement_noise = config.get('heading_measurement_noise', 15.0)  # degrees
        self.heading_perturbation = config.get('heading_perturbation', 0.0)  # degrees (+/- uniform)
        self.use_heading_updates = config.get('use_heading_updates', True)
        self.current_heading_deg = None  # Initialize current heading to None
        
        # Construct full topic paths from config (relative topics) and follower namespace
        # If topic starts with '/', use as-is (absolute), otherwise prepend follower_ns
        def build_topic(relative_topic):
            if not relative_topic:
                return ''
            if relative_topic.startswith('/'):
                return relative_topic  # Already absolute
            if follower_ns:
                return f'/{follower_ns}/{relative_topic}'
            return f'/{relative_topic}'
        
        publish_topic = build_topic(config.get('publish_topic', 'follower/ekf/state'))
        geopoint_topic = build_topic(config.get('geopoint_topic', 'follower/ekf/geopoint'))
        dist1_topic = build_topic(config.get('dist1_topic', 'follower/leader1/distance'))
        dist2_topic = build_topic(config.get('dist2_topic', 'follower/leader2/distance'))
        gps1_topic = build_topic(config.get('gps1_topic', '/leader1/smarc/latlon'))
        gps2_topic = build_topic(config.get('gps2_topic', '/leader2/smarc/latlon'))
        follower_gps_topic = build_topic(config.get('follower_gps_topic', 'smarc/latlon'))
        delta_pos_topic = build_topic(config.get('delta_pos_topic', 'ekf/delta_pos'))
        imu_topic = build_topic(config.get('imu_topic', 'core/imu'))
        odom_topic = config.get('odom_topic', '/lolo/smarc/odom')
        leader_gps_type = config.get('leader_gps_type', 'NavSatFix')
        self.leader_gps_type = leader_gps_type
        follower_gps_type = config.get('follower_gps_type', 'NavSatFix')
        follower_gps_type = config.get('follower_gps_type', 'NavSatFix')
        
        # Publishers
        self.state_pub = self.create_publisher(Float32MultiArray, publish_topic, 10)
        self.geopoint_pub = self.create_publisher(NavSatFix, geopoint_topic, 10)  # Changed to NavSatFix
        self.geopoint_pub_leader1 = self.create_publisher(NavSatFix, geopoint_topic + "_leader1", 10)
        self.geopoint_pub_leader2 = self.create_publisher(NavSatFix, geopoint_topic + "_leader2", 10)
        self.geopoint_pub_centroid = self.create_publisher(NavSatFix, geopoint_topic + "_centroid", 10)
        self.delta_pos_pub = self.create_publisher(Float32, delta_pos_topic, 10)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # UTM reference zone (will be set from first GPS message)
        self.utm_zone_number = None
        self.utm_zone_letter = None
        
        # Initialization tracking
        self.leader_gps_samples = {1: [], 2: []}  # Using leader 1 and 2
        self.follower_gps_samples = []
        self.initialization_phase = 'collecting_leaders'  # 'collecting_leaders', 'collecting_follower', 'ready'
        self.follower_utm_offset = None  # Offset in UTM coordinates for follower GPS
        self.last_truth_follower_utm = None  # Absolute UTM of latest ground-truth follower GPS
        
        # Current measurements in UTM coordinates
        self.current_leader_positions = {1: None, 2: None}  # In UTM coordinates
        self.last_distances = {1: None, 2: None}
        self.last_imu = None
        
        # EKF initialization
        self.ekf = None
        
        # Subscriptions
        self.create_subscription(Float32, dist1_topic, self.dist1_callback, 10)
        self.create_subscription(Float32, dist2_topic, self.dist2_callback, 10)
        if leader_gps_type == 'GeoPoint':
            from geographic_msgs.msg import GeoPoint
            self.create_subscription(GeoPoint, gps1_topic, self.gps1_geopoint_callback, 10)
            self.create_subscription(GeoPoint, gps2_topic, self.gps2_geopoint_callback, 10)
        else:
            self.create_subscription(NavSatFix, gps1_topic, self.gps1_callback, 10)
            self.create_subscription(NavSatFix, gps2_topic, self.gps2_callback, 10)
        
        # Subscribe to follower GPS based on type
        if follower_gps_type == 'GeoPoint':
            from geographic_msgs.msg import GeoPoint
            self.create_subscription(GeoPoint, follower_gps_topic, self.follower_gps_geopoint_callback, 10)
        else:
            self.create_subscription(NavSatFix, follower_gps_topic, self.follower_gps_callback, 10)
        
        # self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)  # <-- Commented out odom callback
        heading_topic = '/lolo/smarc/heading'
        if self.use_heading_updates:
            self.create_subscription(Float32, heading_topic, self.heading_callback, 10)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 10)  # <-- Added IMU subscription
        
        # Timer for prediction step (DISABLED: prediction now triggered by GPS timestamps)
        # self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # EKF matrices
        # Use WNOA-style block diagonal process noise (3 blocks of 4x4 for [x,y,vx,vy])
        self.Q = self.make_WNOA(process_noise, dt=max(self.dt, 0.1))
        # Measurement noise: first two (leader x,y) use measurement_noise^2, distance uses separate noise
        self.R = np.eye(3, dtype=np.float32)
        self.R[0, 0] = measurement_noise ** 2
        self.R[1, 1] = measurement_noise ** 2
        self.R[2, 2] = distance_measurement_noise ** 2
        
        self.update_cooldown = config.get('update_cooldown', 0.5)
        self.last_update_time = self.get_clock().now().nanoseconds / 1e9 - self.update_cooldown

        self.get_logger().info('AcousticEKFNode initialized, collecting leader GPS samples...')

        self.leader_depth = config.get('leader_depth', 0.5)
        depth_topic = build_topic(config.get('depth_topic', 'smarc/depth'))
        self.depth = None
        self.create_subscription(Float32, depth_topic, self.depth_callback, 10)

        # TF2 transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.imu_frame = 'lolo/base_link'  # Adjust if your IMU frame is different
        self.utm_frame = 'utm'            # UTM frame from your tf tree

        self.last_predict_time = None  # For variable dt prediction

        # use_sim_time is a declared ROS param; bag launches pass true explicitly.
        self.get_logger().info(
            f'AcousticEKFNode initialized (use_sim_time='
            f'{bool(self.get_parameter("use_sim_time").value)})')

    def gps_to_utm(self, lat, lon):
        """Convert GPS to UTM coordinates, establishing reference zone from first conversion, with higher precision."""
        try:
            x, y, zone_number, zone_letter = utm.from_latlon(float(lat), float(lon))
            if self.utm_zone_number is None:
                self.utm_zone_number = zone_number
                self.utm_zone_letter = zone_letter
                self.get_logger().info(f'UTM reference zone set to: {zone_number}{zone_letter}')
            # Format to 8 decimal places for internal use
            x = float(f"{x:.8f}")
            y = float(f"{y:.8f}")
            return x, y
        except Exception as e:
            self.get_logger().error(f'UTM conversion failed: {e}')
            return None, None

    def utm_to_gps(self, x, y):
        """Convert UTM coordinates back to GPS with higher precision."""
        if self.utm_zone_number is None or self.utm_zone_letter is None:
            return None, None
        try:
            # Use higher precision for UTM to lat/lon conversion
            lat, lon = utm.to_latlon(float(x), float(y), int(self.utm_zone_number), str(self.utm_zone_letter))
            # Format to 8 decimal places for publishing
            lat = float(f"{lat:.12f}")
            lon = float(f"{lon:.12f}")
            return lat, lon
        except Exception as e:
            self.get_logger().error(f'UTM to GPS conversion failed: {e}')
            return None, None

    def gps1_callback(self, msg):
        """Handle GPS data from leader 1"""
        self.get_logger().info(f'Received GPS1: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
        if self.initialization_phase == 'collecting_leaders':
            self.leader_gps_samples[1].append((msg.latitude, msg.longitude))
            self.get_logger().info(f'Leader 1 GPS sample {len(self.leader_gps_samples[1])}/{self.leader_init_samples}: '
                                   f'lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
            self._check_leader_initialization()
        elif self.initialization_phase == 'ready':
            x, y = self.gps_to_utm(msg.latitude, msg.longitude)
            if x is not None and y is not None and self.follower_utm_offset is not None:
                offset_x = x - self.follower_utm_offset[0] - self.boat_offset[0]
                offset_y = y - self.follower_utm_offset[1] - self.boat_offset[1]
                self.current_leader_positions[1] = (offset_x, offset_y)
                self.get_logger().info(f'Updated current_leader_positions[1] to ({offset_x:.2f}, {offset_y:.2f}) [offset applied]')
            else:
                self.get_logger().warn('Follower UTM offset not initialized or UTM conversion failed for leader 1!')

    def gps2_callback(self, msg):
        """Handle GPS data from leader 2"""
        self.get_logger().info(f'Received GPS2: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
        if self.initialization_phase == 'collecting_leaders':
            self.leader_gps_samples[2].append((msg.latitude, msg.longitude))
            self.get_logger().info(f'Leader 2 GPS sample {len(self.leader_gps_samples[2])}/{self.leader_init_samples}: '
                                   f'lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
            self._check_leader_initialization()
        elif self.initialization_phase == 'ready':
            x, y = self.gps_to_utm(msg.latitude, msg.longitude)
            if x is not None and y is not None and self.follower_utm_offset is not None:
                offset_x = x - self.follower_utm_offset[0] - self.boat_offset[0]
                offset_y = y - self.follower_utm_offset[1] - self.boat_offset[1]
                self.current_leader_positions[2] = (offset_x, offset_y)
                self.get_logger().info(f'Updated current_leader_positions[2] to ({offset_x:.2f}, {offset_y:.2f}) [offset applied]')
            else:
                self.get_logger().warn('Follower UTM offset not initialized or UTM conversion failed for leader 2!')

    def gps1_geopoint_callback(self, msg):
        """Handle GeoPoint data from leader 1"""
        lat, lon = msg.latitude, msg.longitude
        self.get_logger().debug(f'Received GPS1 (GeoPoint): lat={lat:.6f}, lon={lon:.6f}')
        if self.initialization_phase == 'collecting_leaders':
            self.leader_gps_samples[1].append((lat, lon))
            self.get_logger().info(f'Leader 1 GPS sample {len(self.leader_gps_samples[1])}/{self.leader_init_samples}: lat={lat:.6f}, lon={lon:.6f}')
            self._check_leader_initialization()
        elif self.initialization_phase == 'ready':
            x, y = self.gps_to_utm(lat, lon)
            if x is not None and y is not None and self.follower_utm_offset is not None:
                offset_x = x - self.follower_utm_offset[0] - self.boat_offset[0]
                offset_y = y - self.follower_utm_offset[1] - self.boat_offset[1]
                self.current_leader_positions[1] = (offset_x, offset_y)
                self.get_logger().debug(f'Updated current_leader_positions[1] to ({offset_x:.2f}, {offset_y:.2f}) [offset applied]')

    def gps2_geopoint_callback(self, msg):
        """Handle GeoPoint data from leader 2"""
        lat, lon = msg.latitude, msg.longitude
        self.get_logger().debug(f'Received GPS2 (GeoPoint): lat={lat:.6f}, lon={lon:.6f}')
        if self.initialization_phase == 'collecting_leaders':
            self.leader_gps_samples[2].append((lat, lon))
            self.get_logger().info(f'Leader 2 GPS sample {len(self.leader_gps_samples[2])}/{self.leader_init_samples}: lat={lat:.6f}, lon={lon:.6f}')
            self._check_leader_initialization()
        elif self.initialization_phase == 'ready':
            x, y = self.gps_to_utm(lat, lon)
            if x is not None and y is not None and self.follower_utm_offset is not None:
                offset_x = x - self.follower_utm_offset[0] - self.boat_offset[0]
                offset_y = y - self.follower_utm_offset[1] - self.boat_offset[1]
                self.current_leader_positions[2] = (offset_x, offset_y)
                self.get_logger().debug(f'Updated current_leader_positions[2] to ({offset_x:.2f}, {offset_y:.2f}) [offset applied]')

    def follower_gps_callback(self, msg):
        """Handle GPS data from follower for initialization and trigger EKF prediction with variable dt"""
        # self.get_logger().info(f'Received follower GPS: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
        # Use message time for dt
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.initialization_phase == 'collecting_follower':
            self.follower_gps_samples.append((msg.latitude, msg.longitude))
            self.get_logger().info(f'Follower GPS sample {len(self.follower_gps_samples)}/{self.follower_init_samples}: '
                                   f'lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
            if len(self.follower_gps_samples) >= self.follower_init_samples:
                self._initialize_ekf()
        elif self.initialization_phase == 'ready' and self.ekf is not None:
            # Variable dt prediction step
            if self.last_predict_time is not None:
                dt = t - self.last_predict_time
                if dt > 0:
                    with self.lock:
                        # Update latest ground-truth UTM position
                        gx, gy = self.gps_to_utm(msg.latitude, msg.longitude)
                        if gx is not None and gy is not None:
                            self.last_truth_follower_utm = (gx, gy)
                        self.ekf.dt = np.float32(dt)
                        # Update dynamic process noise matrix with new dt
                        self.ekf.Q = self.make_WNOA(self.process_noise, dt)
                        self.ekf.predict()
                        self._publish_state_and_geopoint()
            self.last_predict_time = t
        # Also update truth position even if not ready for prediction
        else:
            gx, gy = self.gps_to_utm(msg.latitude, msg.longitude)
            if gx is not None and gy is not None:
                self.last_truth_follower_utm = (gx, gy)

    def follower_gps_geopoint_callback(self, msg):
        """Handle GeoPoint GPS data from follower for initialization and trigger EKF prediction"""
        lat, lon = msg.latitude, msg.longitude
        self.get_logger().debug(f'Received follower GPS (GeoPoint): lat={lat:.6f}, lon={lon:.6f}')
        
        # Use ROS time consistently (convert to float seconds)
        t = self.get_clock().now().nanoseconds / 1e9
        
        if self.initialization_phase == 'collecting_follower':
            self.follower_gps_samples.append((lat, lon))
            self.get_logger().info(f'Follower GPS sample {len(self.follower_gps_samples)}/{self.follower_init_samples}: '
                                   f'lat={lat:.6f}, lon={lon:.6f}')
            if len(self.follower_gps_samples) >= self.follower_init_samples:
                self._initialize_ekf()
                # Initialize time tracking after EKF is ready
                self.last_predict_time = t
        elif self.initialization_phase == 'ready' and self.ekf is not None:
            # Compute dt from ROS time between message arrivals
            if self.last_predict_time is not None:
                dt = t - self.last_predict_time
                # Sanity check dt (should be between 0.01s and 2s for GPS updates)
                if dt > 0.01 and dt < 2.0:
                    with self.lock:
                        # Update latest ground-truth UTM position
                        gx, gy = self.gps_to_utm(lat, lon)
                        if gx is not None and gy is not None:
                            self.last_truth_follower_utm = (gx, gy)
                        self.ekf.dt = np.float32(dt)
                        # Update dynamic process noise matrix with measured dt
                        self.ekf.Q = self.make_WNOA(self.process_noise, dt)
                        self.ekf.predict()
                        self._publish_state_and_geopoint()
                elif dt >= 2.0:
                    self.get_logger().warn(f'Large dt gap: {dt:.3f}s, skipping prediction')
            else:
                # First prediction after initialization
                self.get_logger().info('First GeoPoint after init, waiting for next message to compute dt')
            self.last_predict_time = t
        # Also update truth position
        else:
            gx, gy = self.gps_to_utm(lat, lon)
            if gx is not None and gy is not None:
                self.last_truth_follower_utm = (gx, gy)

    def dist1_callback(self, msg):
        """Handle distance measurement from leader 1"""
        self.get_logger().debug(f'Received distance from leader 1: {msg.data:.2f}')
        if self.initialization_phase != 'ready':
            self.get_logger().info('dist1_callback ignored: not ready')
            return
        with self.lock:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_update_time < self.update_cooldown:
                self.get_logger().debug('EKF update skipped due to cooldown')
                return
            # Calculate horizontal distance if depth is available
            if self.depth is not None:
                try:
                    d = msg.data * self.correction_factor
                    z_depth = max(self.depth - self.leader_depth, 0.0)
                    horiz_dist = math.sqrt(max(d**2 - z_depth**2, 0.0))
                    self.last_distances[1] = horiz_dist
                except Exception as e:
                    self.get_logger().error(f'Error computing horizontal distance: {e}')
                    self.last_distances[1] = msg.data * self.correction_factor
            else:
                self.last_distances[1] = msg.data * self.correction_factor
            if self.current_leader_positions[1] is not None:
                z = np.array([
                    self.current_leader_positions[1][0],
                    self.current_leader_positions[1][1],
                    self.last_distances[1]
                ], dtype=np.float32)
                self.get_logger().info(f'Updating EKF with leader 1 measurement: {z}')
                self.ekf.update(z, 0)
                self.last_update_time = now
                self._publish_state_and_geopoint()

    def dist2_callback(self, msg):
        """Handle distance measurement from leader 2"""
        self.get_logger().debug(f'Received distance from leader 2: {msg.data:.2f}')
        if self.initialization_phase != 'ready':
            self.get_logger().info('dist2_callback ignored: not ready')
            return
        with self.lock:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_update_time < self.update_cooldown:
                self.get_logger().debug('EKF update skipped due to cooldown')
                return
            # Calculate horizontal distance if depth is available
            if self.depth is not None:
                try:
                    d = msg.data * self.correction_factor
                    z_depth = max(self.depth - self.leader_depth, 0.0)
                    horiz_dist = math.sqrt(max(d**2 - z_depth**2, 0.0))
                    self.last_distances[2] = horiz_dist
                except Exception as e:
                    self.get_logger().error(f'Error computing horizontal distance: {e}')
                    self.last_distances[2] = msg.data * self.correction_factor
            else:
                self.last_distances[2] = msg.data * self.correction_factor
            if self.current_leader_positions[2] is not None:
                z = np.array([
                    self.current_leader_positions[2][0],
                    self.current_leader_positions[2][1],
                    self.last_distances[2]
                ], dtype=np.float32)
                self.get_logger().info(f'Updating EKF with leader 2 measurement: {z}')
                self.ekf.update(z, 1)
                self.last_update_time = now
                self._publish_state_and_geopoint()

    def heading_callback(self, msg):
        """EKF update using heading measurement with proper Jacobians."""
        if not self.use_heading_updates:
            return
        eps = float(self.heading_perturbation)
        if eps > 0:
            perturb = np.random.uniform(-eps, eps)
        else:
            perturb = 0.0
        self.current_heading_deg = msg.data + perturb  # Optional perturbation for robustness testing
        # Only update if EKF is ready
        if self.initialization_phase != 'ready' or self.ekf is None:
            return
        
        # Perform EKF update with heading measurement
        with self.lock:
            self._update_heading_measurement(self.current_heading_deg)
    
    def _update_heading_measurement(self, heading_deg):
        """Perform EKF update using heading measurement with proper Jacobians."""
        # Convert heading to radians (NED: 0=N, 90=E)
        heading_rad = math.radians(heading_deg)
        
        # Get current velocity estimates
        vx = float(self.ekf.x[10])  # Velocity in UTM x (east)
        vy = float(self.ekf.x[11])  # Velocity in UTM y (north)
        
        # Compute current speed
        speed = math.sqrt(vx**2 + vy**2)
        
        # Avoid division by zero for very small speeds
        if speed < 1e-6:
            self.get_logger().debug('Speed too small for heading update, skipping')
            return
        
        # Measurement model: heading = atan2(vx, vy) (NED convention: atan2(east, north))
        # Expected measurement (predicted heading from current velocity)
        predicted_heading = math.atan2(vx, vy)
        
        # Normalize angles to [-pi, pi]
        def normalize_angle(angle):
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle
        
        # Innovation (measurement residual)
        innovation = np.array(normalize_angle(heading_rad - predicted_heading)).reshape(-1, 1)
        
        # Measurement Jacobian H (1x12 vector, only non-zero for velocity states)
        H = np.zeros((1, 12), dtype=np.float32)
        
        # Partial derivatives of atan2(vx, vy) w.r.t. vx and vy
        # d/dvx [atan2(vx, vy)] = vy / (vx^2 + vy^2)
        # d/dvy [atan2(vx, vy)] = -vx / (vx^2 + vy^2)
        speed_sq = vx**2 + vy**2
        H[0, 10] = vy / speed_sq  # derivative w.r.t. vx (state index 10)
        H[0, 11] = -vx / speed_sq  # derivative w.r.t. vy (state index 11)
        
        # Measurement noise covariance (heading uncertainty in radians)
        R_heading = np.array([[math.radians(self.heading_measurement_noise)**2]], dtype=np.float32)
        
        # Kalman update equations
        try:
            # Innovation covariance
            S = H @ self.ekf.P @ H.T + R_heading
            
            # Kalman gain
            K = self.ekf.P @ H.T @ np.linalg.inv(S)            
            # State update
            self.ekf.x = self.ekf.x + K @ innovation
            
            # Covariance update (Joseph form for numerical stability)
            I = np.eye(12, dtype=np.float32)
            IKH = I - K @ H
            self.ekf.P = IKH @ self.ekf.P
            
            # Apply velocity clipping
            # self.ekf._clip_velocities()
            
            self.get_logger().debug(f'Heading EKF update: measured={heading_deg:.1f}°, '
                                  f'predicted={math.degrees(predicted_heading):.1f}°, '
                                  f'innovation={math.degrees(innovation):.1f}°, '
                                  f'updated_vx={float(self.ekf.x[10]):.3f}, '
                                  f'updated_vy={float(self.ekf.x[11]):.3f}')
            
        except Exception as e:
            self.get_logger().error(f'Heading EKF update failed: {e}')

    def imu_callback(self, msg):
        """Trigger EKF prediction step on IMU message."""
        if self.initialization_phase != 'ready' or self.ekf is None:
            return
        # Use IMU timestamp for dt if available
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if hasattr(self, 'last_predict_time') and self.last_predict_time is not None:
            dt = t - self.last_predict_time
            if dt > 0:
                with self.lock:
                    self.ekf.dt = np.float32(dt)
                    # Update dynamic process noise matrix with new dt
                    self.ekf.Q = self.make_WNOA(self.process_noise, dt)
                    self.ekf.predict()
                    self._publish_state_and_geopoint()
        self.last_predict_time = t

    def timer_callback(self):
        """Prediction step of EKF and always publish state."""
        if self.initialization_phase != 'ready' or self.ekf is None:
            return
        with self.lock:
            self.ekf.predict()
            self._publish_state_and_geopoint()  # Always publish after prediction

    def depth_callback(self, msg):
        self.depth = msg.data

    def _check_leader_initialization(self):
        """Check if we have enough leader GPS samples to initialize"""
        if (len(self.leader_gps_samples[1]) >= self.leader_init_samples and 
            len(self.leader_gps_samples[2]) >= self.leader_init_samples):
            
            self.initialization_phase = 'collecting_follower'
            self.get_logger().info('Leaders initialized, now collecting follower GPS samples...')

    def _initialize_ekf(self):
        """Initialize EKF with collected GPS samples in UTM coordinates and store follower UTM offset."""
        self.get_logger().info('Initializing EKF...')
        # Use only the last 10 samples if more are available
        leader1_samples = self.leader_gps_samples[1][-self.leader_init_samples:] if len(self.leader_gps_samples[1]) > self.leader_init_samples else self.leader_gps_samples[1]
        leader2_samples = self.leader_gps_samples[2][-self.leader_init_samples:] if len(self.leader_gps_samples[2]) > self.leader_init_samples else self.leader_gps_samples[2]
        if len(leader1_samples) < self.leader_init_samples or len(leader2_samples) < self.leader_init_samples:
            self.get_logger().error('Not enough leader GPS samples to initialize EKF')
            return
        if len(self.follower_gps_samples) < self.follower_init_samples:
            self.get_logger().error('Not enough follower GPS samples to initialize EKF')
            return

        # Use only the last follower samples if more are available
        follower_samples = self.follower_gps_samples[-self.follower_init_samples:] if len(self.follower_gps_samples) > self.follower_init_samples else self.follower_gps_samples

        avg_lat1 = sum(sample[0] for sample in leader1_samples) / len(leader1_samples)
        avg_lon1 = sum(sample[1] for sample in leader1_samples) / len(leader1_samples)
        
        avg_lat2 = sum(sample[0] for sample in leader2_samples) / len(leader2_samples)
        avg_lon2 = sum(sample[1] for sample in leader2_samples) / len(leader2_samples)
        
        avg_follower_lat = sum(sample[0] for sample in self.follower_gps_samples) / len(self.follower_gps_samples)
        avg_follower_lon = sum(sample[1] for sample in self.follower_gps_samples) / len(self.follower_gps_samples)
        
        # Convert to UTM coordinates
        leader1_x, leader1_y = self.gps_to_utm(avg_lat1, avg_lon1)
        leader2_x, leader2_y = self.gps_to_utm(avg_lat2, avg_lon2)
        follower_x, follower_y = self.gps_to_utm(avg_follower_lat, avg_follower_lon)
        if None in [leader1_x, leader1_y, leader2_x, leader2_y, follower_x, follower_y]:
            self.get_logger().error('Failed to convert GPS to UTM coordinates')
            return
        # Store follower UTM offset for later use
        self.follower_utm_offset = np.array([follower_x, follower_y], dtype=np.float64)
        # Apply boat offset to leader positions (subtracting so that we add back when publishing)
        leader1_x_adj = leader1_x - self.boat_offset[0]
        leader1_y_adj = leader1_y - self.boat_offset[1]
        leader2_x_adj = leader2_x - self.boat_offset[0]
        leader2_y_adj = leader2_y - self.boat_offset[1]
        # Initialize EKF state vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, vfx, vfy]
        initial_state = np.array([
            leader1_x_adj - follower_x, leader1_y_adj - follower_y, 0.0, 0.0,  # Leader 1 (offset & boat adjustment)
            leader2_x_adj - follower_x, leader2_y_adj - follower_y, 0.0, 0.0,  # Leader 2 (offset & boat adjustment)
            0.0, 0.0, 0.0, 0.0 # Follower position and velocity (origin)
        ], dtype=np.float32)
        self.ekf = EKF(initial_state, Q=self.Q, R=self.R, dt=self.dt, max_velocity=self.max_speed)
        self.initialization_phase = 'ready'
        self.get_logger().info(f'EKF initialized with UTM offset: {self.follower_utm_offset}')
        self.get_logger().info('EKF ready for operation!')

    def _publish_state_and_geopoint(self):
        """Publish EKF state and NavSatFix, adding back follower UTM offset."""
        if self.ekf is None or self.follower_utm_offset is None:
            self.get_logger().warn('Cannot publish: EKF or follower_utm_offset not initialized!')
            return
        state = self.ekf.get_state()
        # Publish state vector (still offset)
        state_msg = Float32MultiArray()
        state_msg.data = state.tolist()
        self.state_pub.publish(state_msg)
        # Add offset back for publishing
        follower_modem_x = state[8] + self.follower_utm_offset[0]
        follower_modem_y = state[9] + self.follower_utm_offset[1]
        
        # Apply reverse transform from modem position to baselink position
        # The EKF estimates modem position, but we want to publish baselink position
        if hasattr(self, 'current_heading_deg') and self.current_heading_deg is not None:
            # Convert heading to radians (NED convention: 0=North, 90=East)
            heading_rad = math.radians(self.current_heading_deg)
            
            # Reverse transform: baselink = modem - offset * [sin(heading), cos(heading)]
            # This undoes the offset applied in post-processing
            follower_x = follower_modem_x - self.modem_to_baselink_offset_x * math.sin(heading_rad)
            follower_y = follower_modem_y - self.modem_to_baselink_offset_x * math.cos(heading_rad)
        else:
            # If no heading available, publish modem position (fallback)
            self.get_logger().warn('No heading available for modem-to-baselink transform, publishing modem position', throttle_duration_sec=5.0)
            follower_x = follower_modem_x
            follower_y = follower_modem_y
        # Sanity check for out-of-range UTM values
        if abs(follower_x) > 1e7 or abs(follower_y) > 1e7:
            self.get_logger().warn(f'Follower UTM values out of range: x={follower_x}, y={follower_y}. Skipping publish.')
            return
        follower_lat, follower_lon = self.utm_to_gps(follower_x, follower_y)
        if follower_lat is not None and follower_lon is not None:
            navsat_msg = NavSatFix()
            navsat_msg.latitude = follower_lat
            navsat_msg.longitude = follower_lon
            navsat_msg.altitude = 0.0
            navsat_msg.header.stamp = self.get_clock().now().to_msg()
            navsat_msg.header.frame_id = "map"
            navsat_msg.status.status = 0
            navsat_msg.status.service = 1
            self.geopoint_pub.publish(navsat_msg)
        # Publish delta distance to ground truth if available
        if self.last_truth_follower_utm is not None:
            truth_x, truth_y = self.last_truth_follower_utm
            delta = math.sqrt((follower_x - truth_x)**2 + (follower_y - truth_y)**2)
            delta_msg = Float32()
            delta_msg.data = float(delta)
            self.delta_pos_pub.publish(delta_msg)
        # Leader 1 position
        leader1_x = state[0] + self.follower_utm_offset[0] + self.boat_offset[0]
        leader1_y = state[1] + self.follower_utm_offset[1] + self.boat_offset[1]
        if abs(leader1_x) > 1e7 or abs(leader1_y) > 1e7:
            self.get_logger().warn(f'Leader1 UTM values out of range: x={leader1_x}, y={leader1_y}. Skipping publish.')
        else:
            leader1_lat, leader1_lon = self.utm_to_gps(leader1_x, leader1_y)
            if leader1_lat is not None and leader1_lon is not None:
                navsat_leader1 = NavSatFix()
                navsat_leader1.latitude = leader1_lat
                navsat_leader1.longitude = leader1_lon
                navsat_leader1.altitude = 0.0
                navsat_leader1.header.stamp = self.get_clock().now().to_msg()
                navsat_leader1.header.frame_id = "map"
                navsat_leader1.status.status = 0
                navsat_leader1.status.service = 1
                self.geopoint_pub_leader1.publish(navsat_leader1)
        # Leader 2 position
        leader2_x = state[4] + self.follower_utm_offset[0] + self.boat_offset[0]
        leader2_y = state[5] + self.follower_utm_offset[1] + self.boat_offset[1]
        if abs(leader2_x) > 1e7 or abs(leader2_y) > 1e7:
            self.get_logger().warn(f'Leader2 UTM values out of range: x={leader2_x}, y={leader2_y}. Skipping publish.')
        else:
            leader2_lat, leader2_lon = self.utm_to_gps(leader2_x, leader2_y)
            if leader2_lat is not None and leader2_lon is not None:
                navsat_leader2 = NavSatFix()
                navsat_leader2.latitude = leader2_lat
                navsat_leader2.longitude = leader2_lon
                navsat_leader2.altitude = 0.0
                navsat_leader2.header.stamp = self.get_clock().now().to_msg()
                navsat_leader2.header.frame_id = "map"
                navsat_leader2.status.status = 0
                navsat_leader2.status.service = 1
                self.geopoint_pub_leader2.publish(navsat_leader2)
        
        # Publish centroid of all three positions
        if (follower_lat is not None and follower_lon is not None and
            abs(leader1_x) <= 1e7 and abs(leader1_y) <= 1e7 and
            abs(leader2_x) <= 1e7 and abs(leader2_y) <= 1e7):
            
            # Calculate centroid in UTM coordinates
            centroid_x = (follower_x + leader1_x + leader2_x) / 3.0
            centroid_y = (follower_y + leader1_y + leader2_y) / 3.0
            
            centroid_lat, centroid_lon = self.utm_to_gps(centroid_x, centroid_y)
            if centroid_lat is not None and centroid_lon is not None:
                navsat_centroid = NavSatFix()
                navsat_centroid.latitude = centroid_lat
                navsat_centroid.longitude = centroid_lon
                navsat_centroid.altitude = 0.0
                navsat_centroid.header.stamp = self.get_clock().now().to_msg()
                navsat_centroid.header.frame_id = "map"
                navsat_centroid.status.status = 0
                navsat_centroid.status.service = 1
                self.geopoint_pub_centroid.publish(navsat_centroid)

    def make_WNOA(self, process_noise, dt):
        """Create block diagonal WNOA process noise matrix (3 x 4x4 blocks for [x,y,vx,vy])."""
        dt = float(max(dt, 1e-3))  # guard against zero
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        block = (process_noise ** 2) * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2]
        ], dtype=np.float32)
        Q = np.zeros((12, 12), dtype=np.float32)
        for i in range(3):
            Q[i*4:(i+1)*4, i*4:(i+1)*4] = block
        return Q


def main(args=None):
    rclpy.init(args=args)
    node = AcousticEKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
