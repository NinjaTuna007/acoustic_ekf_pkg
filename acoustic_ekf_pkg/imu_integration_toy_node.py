import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
import numpy as np
import math
import os
import yaml
import utm
from tf2_ros import Buffer, TransformListener
import tf_transformations
import matplotlib.pyplot as plt
import signal

class ImuIntegrationToyNode(Node):
    def __init__(self):
        super().__init__('imu_integration_toy_node')
        # Load config from YAML
        config_path = os.path.join(os.path.dirname(__file__), '../config/ekf_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.imu_topic = config.get('imu_topic', '/lolo/standard/imu')
        self.gps_topic = config.get('follower_gps_topic', '/lolo/standard/navsatfix')
        self.publish_topic = config.get('imu_integration_publish_topic', '/imu_integration/point')
        self.odom_topic = config.get('odom_topic', '/lolo/smarc/odom')
        self.odom_initialized = False
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)
        self.create_subscription(NavSatFix, self.gps_topic, self.gps_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.pub = self.create_publisher(NavSatFix, self.publish_topic, 10)

        self.last_time = None
        self.utm_zone_number = None
        self.utm_zone_letter = None
        self.position = None  # [x, y] in UTM meters
        self.velocity = np.zeros(2)  # [vx, vy] in m/s
        self.gps_initialized = False
        self.imu_frame = config.get('imu_frame', 'lolo/base_link')
        self.utm_frame = config.get('utm_frame', 'utm')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.trajectory = []  # Store UTM positions for plotting
        self._plot_on_exit = False
        signal.signal(signal.SIGINT, self.handle_sigint)
        signal.signal(signal.SIGTERM, self.handle_sigint)

    def gps_callback(self, msg):
        if not self.gps_initialized:
            # Convert lat/lon to UTM and store reference zone
            x, y, zone_number, zone_letter = utm.from_latlon(msg.latitude, msg.longitude)
            self.position = np.array([x, y])
            self.utm_zone_number = zone_number
            self.utm_zone_letter = zone_letter
            self.gps_initialized = True
            self.get_logger().info(f'Initialized UTM position from GPS: {self.position}, zone: {zone_number}{zone_letter}')

    def imu_callback(self, msg):
        if not self.gps_initialized or self.utm_zone_number is None or self.utm_zone_letter is None:
            return
        # Use IMU message timestamp for integration
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = t
            self.get_logger().info(f'IMU integration started at t={t:.3f}')
            return
        dt = t - self.last_time
        self.last_time = t
        # Only attempt TF lookup if transform is available
        if not self.tf_buffer.can_transform(self.utm_frame, self.imu_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
            if not hasattr(self, '_tf_warned'):
                self.get_logger().warn(f'TF transform from {self.imu_frame} to {self.utm_frame} not available yet. Will retry.')
                self._tf_warned = True
            return
        # Log success after previous warning
        if hasattr(self, '_tf_warned'):
            self.get_logger().info(f'TF transform from {self.imu_frame} to {self.utm_frame} is now available.')
            del self._tf_warned
        try:
            tform = self.tf_buffer.lookup_transform(
                self.utm_frame, self.imu_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            q = tform.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
            rot_mat = tf_transformations.quaternion_matrix(quat)[:3, :3]
            acc_vec = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            acc_utm = rot_mat @ acc_vec
            ax = acc_utm[0]
            ay = acc_utm[1]
            az = acc_utm[2]
        except Exception as e:
            self.get_logger().warn(f'IMU tf transform failed: {e}')
            return
        # Integrate velocity
        self.velocity[0] += ax * dt
        self.velocity[1] += ay * dt
        # Integrate position in UTM
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.trajectory.append(self.position.copy())  # Store for plotting
        # Log for validation
        # self.get_logger().info(f'IMU t={t:.3f}, dt={dt:.3f}, acc_utm=({ax:.3f}, {ay:.3f}), vel=({self.velocity[0]:.3f}, {self.velocity[1]:.3f}), pos=({self.position[0]:.3f}, {self.position[1]:.3f})')
        # Convert UTM back to lat/lon for publishing
        try:
            lat, lon = utm.to_latlon(self.position[0], self.position[1], self.utm_zone_number, self.utm_zone_letter)
        except Exception as e:
            self.get_logger().warn(f'UTM to lat/lon conversion failed: {e}')
            return
        navsat = NavSatFix()
        navsat.header.stamp = msg.header.stamp
        navsat.header.frame_id = 'imu_integration'
        navsat.latitude = lat
        navsat.longitude = lon
        navsat.altitude = 0.0
        self.pub.publish(navsat)
        self.get_logger().debug(f'Published lat/lon: ({lat}, {lon})')

    def odom_callback(self, msg):
        """
        Callback to initialize velocity from odometry, transforming to UTM frame if needed.
        Only sets velocity once at startup. Waits for transform to be available, does NOT initialize with raw odom velocity.
        """
        if not self.odom_initialized and self.gps_initialized:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            # Only initialize if transform is available
            if self.tf_buffer.can_transform(self.utm_frame, 'lolo/base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
                try:
                    t = self.tf_buffer.lookup_transform(
                        self.utm_frame, 'lolo/base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    q = t.transform.rotation
                    quat = [q.x, q.y, q.z, q.w]
                    rot_mat = tf_transformations.quaternion_matrix(quat)[:3, :3]
                    vel_vec = np.array([vx, vy, 0.0])
                    vel_utm = rot_mat @ vel_vec
                    vx_utm = vel_utm[0]
                    vy_utm = vel_utm[1]
                    self.velocity = np.array([vx_utm, vy_utm])
                    self.odom_initialized = True
                    self.get_logger().info(f'Initialized velocity from odometry (transformed to UTM): vx={vx_utm:.3f}, vy={vy_utm:.3f}')
                except Exception as e:
                    self.get_logger().warn(f'Odom tf transform failed: {e}. Will retry.')
            else:
                self.get_logger().warn(f'TF transform from lolo/base_link to {self.utm_frame} not available. Waiting to initialize velocity.')

    def handle_sigint(self, signum, frame):
        self._plot_on_exit = True
        # Do NOT call rclpy.shutdown() here! Let main() handle shutdown and plotting.

    def plot_trajectory(self):
        if self.trajectory:
            traj = np.array(self.trajectory)
            plt.figure(figsize=(8, 6))
            plt.plot(traj[:, 0], traj[:, 1], '-', color='royalblue', linewidth=2, label='UTM Trajectory')
            plt.scatter(traj[0, 0], traj[0, 1], color='green', s=60, label='Start')
            plt.scatter(traj[-1, 0], traj[-1, 1], color='red', s=60, label='End')
            plt.xlabel('UTM X (meters)')
            plt.ylabel('UTM Y (meters)')
            plt.title('IMU Integration UTM Trajectory')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show(block=True)

def main(args=None):
    rclpy.init(args=args)
    node = ImuIntegrationToyNode()
    try:
        rclpy.spin(node)
    finally:
        # Clean up ROS node resources first
        node.destroy_node()
        # Now safe to plot using plain Python attributes
        if getattr(node, '_plot_on_exit', False):
            node.plot_trajectory()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
