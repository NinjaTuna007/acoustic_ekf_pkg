import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import numpy as np
import utm
import os
import yaml
import tf_transformations
from tf2_ros import Buffer, TransformListener

class OdomIntegrationNode(Node):
    def __init__(self):
        super().__init__('odom_integration_node')
        # Load config from YAML
        config_path = os.path.join(os.path.dirname(__file__), '../config/ekf_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.odom_topic = config.get('odom_topic', '/lolo/smarc/odom')
        self.gps_topic = config.get('follower_gps_topic', '/lolo/standard/navsatfix')
        self.publish_topic = config.get('odom_integration_publish_topic', '/odom_integration/point')
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(NavSatFix, self.gps_topic, self.gps_callback, 10)
        self.pub = self.create_publisher(NavSatFix, self.publish_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_time = None
        self.utm_zone_number = None
        self.utm_zone_letter = None
        self.position = None  # [x, y] in UTM meters
        self.velocity = np.zeros(2)  # [vx, vy] in m/s
        self.gps_initialized = False

    def gps_callback(self, msg):
        if not self.gps_initialized:
            x, y, zone_number, zone_letter = utm.from_latlon(msg.latitude, msg.longitude)
            self.position = np.array([x, y])
            self.utm_zone_number = zone_number
            self.utm_zone_letter = zone_letter
            self.gps_initialized = True
            self.get_logger().info(f'Initialized UTM position from GPS: {self.position}, zone: {zone_number}{zone_letter}')

    def odom_callback(self, msg):
        if not self.gps_initialized:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z  # Assuming z velocity is also available
        # Transform odom velocity to UTM frame if needed
        if hasattr(msg, 'header') and hasattr(msg.header, 'frame_id') and msg.header.frame_id != '':
            if self.tf_buffer.can_transform('utm_33_V', msg.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
                try:
                    # log the frame id for debugging
                    # self.get_logger().warn(f'Attempting to transform from {msg.header.frame_id} to utm_33_V')
                    tform = self.tf_buffer.lookup_transform('utm_33_V', 'lolo/base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
                    q = tform.transform.rotation
                    quat = [q.x, q.y, q.z, q.w]
                    rot_mat = tf_transformations.quaternion_matrix(quat)[:3, :3]
                    vel_vec = np.array([vx, vy, vz])
                    vel_utm = rot_mat @ vel_vec
                    vx_utm = vel_utm[0]
                    vy_utm = vel_utm[1]
                    self.get_logger().info(f'Transformed odom velocity from {msg.header.frame_id} to UTM: ({vx}, {vy}) -> ({vx_utm}, {vy_utm})')
                    self.velocity = np.array([vx_utm, vy_utm])
                except Exception as e:
                    self.get_logger().warn(f'Odom tf transform failed: {e}. Using raw odom velocity.')
                    self.velocity = np.array([vx, vy])
            else:
                self.get_logger().warn(f'TF transform from {msg.header.frame_id} to utm not available. Using raw odom velocity.')
                self.velocity = np.array([vx, vy])
        else:
            self.velocity = np.array([vx, vy])
        # Integrate position in UTM
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        # Convert UTM back to lat/lon for publishing
        try:
            lat, lon = utm.to_latlon(self.position[0], self.position[1], self.utm_zone_number, self.utm_zone_letter)
        except Exception as e:
            self.get_logger().warn(f'UTM to lat/lon conversion failed: {e}')
            return
        navsat = NavSatFix()
        navsat.header.stamp = msg.header.stamp
        navsat.header.frame_id = 'odom_integration'
        navsat.latitude = lat
        navsat.longitude = lon
        navsat.altitude = 0.0
        self.pub.publish(navsat)
        self.get_logger().debug(f'Integrated UTM position: {self.position}, velocity (UTM): {self.velocity}, published lat/lon: ({lat}, {lon})')

def main(args=None):
    rclpy.init(args=args)
    node = OdomIntegrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
