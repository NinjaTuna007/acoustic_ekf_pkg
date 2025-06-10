#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, NavSatFix
import numpy as np
import threading
import time
import yaml
import os
import math
import utm


class EKF:
    def __init__(self, initial_state, Q=None, R=None, leader_velocities=False, dt=0.1, max_velocity=2.0):
        self.dt = np.float32(dt)
        self.Q = Q if Q is not None else np.eye(12, dtype=np.float32) * np.float32(0.01)
        self.leader_velocities = leader_velocities
        self.max_velocity = max_velocity  # Maximum allowed velocity (m/s)
        
        if self.leader_velocities:
            self.R = R if R is not None else np.eye(5, dtype=np.float32) * np.float32(0.05)
        else:
            self.R = R if R is not None else np.eye(3, dtype=np.float32) * np.float32(0.05)
            
        # State vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, thetaf, vf]
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
        self.x[11] = np.clip(self.x[11], 0.0, self.max_velocity)  # Follower speed should be >= 0

    def predict(self):
        # Store previous state
        self.previous_x = self.x.copy()
        
        # Extract state variables
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, thetaf, vf = self.x.flatten()
        
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
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, self.dt * -np.sin(thetaf) * vf, -self.dt * np.cos(thetaf)],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, self.dt * np.cos(thetaf) * vf, self.dt * np.sin(thetaf)],
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
            [xf + vf * np.cos(thetaf) * self.dt],
            [yf + vf * np.sin(thetaf) * self.dt],
            [thetaf],
            [vf]
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
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, thetaf, vf = self.x.flatten()
        
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
        
        # Load EKF config
        config_path = os.path.join(os.path.dirname(__file__), '../config/ekf_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Configuration parameters
        process_noise = config.get('process_noise', 0.01)
        measurement_noise = config.get('measurement_noise', 0.05)
        self.leader_init_samples = config.get('leader_init_samples', 10)
        self.follower_init_samples = config.get('follower_init_samples', 3)
        self.max_speed = config.get('max_speed', 2.0)
        self.dt = config.get('dt', 0.1)  # Read dt from config
        self.correction_factor = config.get('correction_factor', 1.0)  # Correction factor for distance measurements
        
        # Topic names from config
        publish_topic = config.get('publish_topic', '/follower/ekf/state')
        geopoint_topic = config.get('geopoint_topic', '/follower/ekf/geopoint')
        dist1_topic = config.get('dist1_topic', '/follower/leader1/distance')
        dist2_topic = config.get('dist2_topic', '/follower/leader2/distance')
        gps1_topic = config.get('gps1_topic', '/follower/leader1/core/gps')
        gps2_topic = config.get('gps2_topic', '/follower/leader2/core/gps')
        follower_gps_topic = config.get('follower_gps_topic', '/follower/core/gps')
        imu_topic = config.get('imu_topic', '/follower/core/imu')
        
        # Publishers
        self.state_pub = self.create_publisher(Float32MultiArray, publish_topic, 10)
        self.geopoint_pub = self.create_publisher(NavSatFix, geopoint_topic, 10)  # Changed to NavSatFix
        self.geopoint_pub_leader1 = self.create_publisher(NavSatFix, geopoint_topic + "_leader1", 10)
        self.geopoint_pub_leader2 = self.create_publisher(NavSatFix, geopoint_topic + "_leader2", 10)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # UTM reference zone (will be set from first GPS message)
        self.utm_zone_number = None
        self.utm_zone_letter = None
        
        # Initialization tracking
        self.leader_gps_samples = {1: [], 2: []}  # Using leader 1 and 2
        self.follower_gps_samples = []
        self.initialization_phase = 'collecting_leaders'  # 'collecting_leaders', 'collecting_follower', 'ready'
        
        # Current measurements in UTM coordinates
        self.current_leader_positions = {1: None, 2: None}  # In UTM coordinates
        self.last_distances = {1: None, 2: None}
        self.last_imu = None
        
        # EKF initialization
        self.ekf = None
        
        # Subscriptions
        self.create_subscription(Float32, dist1_topic, self.dist1_callback, 10)
        self.create_subscription(Float32, dist2_topic, self.dist2_callback, 10)
        self.create_subscription(NavSatFix, gps1_topic, self.gps1_callback, 10)
        self.create_subscription(NavSatFix, gps2_topic, self.gps2_callback, 10)
        self.create_subscription(NavSatFix, follower_gps_topic, self.follower_gps_callback, 10)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        
        # Timer for prediction step
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # EKF matrices
        self.Q = np.eye(12, dtype=np.float32) * np.float32(process_noise)
        self.R = np.eye(3, dtype=np.float32) * np.float32(measurement_noise)
        
        self.get_logger().info('AcousticEKFNode initialized, collecting leader GPS samples...')

    def gps_to_utm(self, lat, lon):
        """Convert GPS to UTM coordinates, establishing reference zone from first conversion"""
        try:
            x, y, zone_number, zone_letter = utm.from_latlon(lat, lon)
            
            # Set reference zone from first GPS message
            if self.utm_zone_number is None:
                self.utm_zone_number = zone_number
                self.utm_zone_letter = zone_letter
                self.get_logger().info(f'UTM reference zone set to: {zone_number}{zone_letter}')
            
            return x, y
        except Exception as e:
            self.get_logger().error(f'UTM conversion failed: {e}')
            return None, None

    def utm_to_gps(self, x, y):
        """Convert UTM coordinates back to GPS"""
        if self.utm_zone_number is None or self.utm_zone_letter is None:
            return None, None
        
        try:
            lat, lon = utm.to_latlon(x, y, self.utm_zone_number, self.utm_zone_letter)
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
            # Convert GPS to UTM coordinates
            x, y = self.gps_to_utm(msg.latitude, msg.longitude)
            if x is not None and y is not None:
                self.current_leader_positions[1] = (x, y)
                self.get_logger().info(f'Updated current_leader_positions[1] to ({x:.2f}, {y:.2f})')

    def gps2_callback(self, msg):
        """Handle GPS data from leader 2"""
        self.get_logger().info(f'Received GPS2: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
        if self.initialization_phase == 'collecting_leaders':
            self.leader_gps_samples[2].append((msg.latitude, msg.longitude))
            self.get_logger().info(f'Leader 2 GPS sample {len(self.leader_gps_samples[2])}/{self.leader_init_samples}: '
                                   f'lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
            self._check_leader_initialization()
        elif self.initialization_phase == 'ready':
            # Convert GPS to UTM coordinates
            x, y = self.gps_to_utm(msg.latitude, msg.longitude)
            if x is not None and y is not None:
                self.current_leader_positions[2] = (x, y)
                self.get_logger().info(f'Updated current_leader_positions[2] to ({x:.2f}, {y:.2f})')

    def follower_gps_callback(self, msg):
        """Handle GPS data from follower for initialization"""
        self.get_logger().info(f'Received follower GPS: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
        if self.initialization_phase == 'collecting_follower':
            self.follower_gps_samples.append((msg.latitude, msg.longitude))
            self.get_logger().info(f'Follower GPS sample {len(self.follower_gps_samples)}/{self.follower_init_samples}: '
                                   f'lat={msg.latitude:.6f}, lon={msg.longitude:.6f}')
            
            if len(self.follower_gps_samples) >= self.follower_init_samples:
                self._initialize_ekf()

    def dist1_callback(self, msg):
        """Handle distance measurement from leader 1"""
        self.get_logger().info(f'Received distance from leader 1: {msg.data:.2f}')
        if self.initialization_phase != 'ready':
            self.get_logger().info('dist1_callback ignored: not ready')
            return
            
        with self.lock:
            self.last_distances[1] = msg.data * self.correction_factor  # Apply correction factor
            if self.current_leader_positions[1] is not None:
                # Create measurement vector [x1, y1, distance]
                z = np.array([
                    self.current_leader_positions[1][0],
                    self.current_leader_positions[1][1],
                    self.last_distances[1]
                ], dtype=np.float32)
                self.get_logger().info(f'Updating EKF with leader 1 measurement: {z}')
                self.ekf.update(z, 0)  # Use 0 for leader 1 (first leader)
                self._publish_state_and_geopoint()

    def dist2_callback(self, msg):
        """Handle distance measurement from leader 2"""
        self.get_logger().info(f'Received distance from leader 2: {msg.data:.2f}')
        if self.initialization_phase != 'ready':
            self.get_logger().info('dist2_callback ignored: not ready')
            return
            
        with self.lock:
            self.last_distances[2] = msg.data * self.correction_factor  # Apply correction factor
            if self.current_leader_positions[2] is not None:
                # Create measurement vector [x2, y2, distance]
                z = np.array([
                    self.current_leader_positions[2][0],
                    self.current_leader_positions[2][1],
                    self.last_distances[2]
                ], dtype=np.float32)
                self.get_logger().info(f'Updating EKF with leader 2 measurement: {z}')
                self.ekf.update(z, 1)  # Use 1 for leader 2 (second leader)
                self._publish_state_and_geopoint()

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.get_logger().info('Received IMU data')
        self.last_imu = msg

    def timer_callback(self):
        """Prediction step of EKF"""
        self.get_logger().debug('Timer callback triggered')
        if self.initialization_phase != 'ready' or self.ekf is None:
            self.get_logger().debug('Timer callback ignored: not ready or EKF not initialized')
            return
            
        with self.lock:
            self.ekf.predict()
            self._publish_state_and_geopoint()

    def _check_leader_initialization(self):
        """Check if we have enough leader GPS samples to initialize"""
        if (len(self.leader_gps_samples[1]) >= self.leader_init_samples and 
            len(self.leader_gps_samples[2]) >= self.leader_init_samples):
            
            self.initialization_phase = 'collecting_follower'
            self.get_logger().info('Leaders initialized, now collecting follower GPS samples...')

    def _initialize_ekf(self):
        """Initialize EKF with collected GPS samples in UTM coordinates"""
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
        
        # Initialize EKF state vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, thetaf, vf]
        initial_state = np.array([
            leader1_x, leader1_y, 0.0, 0.0,  # Leader 1 position and velocity
            leader2_x, leader2_y, 0.0, 0.0,  # Leader 2 position and velocity
            follower_x, follower_y, 0.0, 0.0  # Follower position, heading, and velocity
        ], dtype=np.float32)
        
        # Initialize EKF
        self.ekf = EKF(initial_state, Q=self.Q, R=self.R, dt=self.dt, max_velocity=self.max_speed)
        self.initialization_phase = 'ready'
        
        self.get_logger().info(f'EKF initialized with UTM coordinates:')
        self.get_logger().info(f'  Leader 1: ({leader1_x:.2f}, {leader1_y:.2f})')
        self.get_logger().info(f'  Leader 2: ({leader2_x:.2f}, {leader2_y:.2f})')
        self.get_logger().info(f'  Follower: ({follower_x:.2f}, {follower_y:.2f})')
        self.get_logger().info('EKF ready for operation!')

    def _publish_state_and_geopoint(self):
        """Publish EKF state and NavSatFix"""
        if self.ekf is None:
            return
            
        state = self.ekf.get_state()
        
        # Publish state vector
        state_msg = Float32MultiArray()
        state_msg.data = state.tolist()
        self.state_pub.publish(state_msg)
        
        # Follower position
        follower_x, follower_y = state[8], state[9]
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

        # Leader 1 position
        leader1_x, leader1_y = state[0], state[1]
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
        leader2_x, leader2_y = state[4], state[5]
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


def main(args=None):
    rclpy.init(args=args)
    node = AcousticEKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
