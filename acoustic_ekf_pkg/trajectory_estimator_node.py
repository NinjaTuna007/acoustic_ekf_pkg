import rclpy
from rclpy.node import Node
import numpy as np
import utm
import matplotlib.pyplot as plt

from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import tf2_ros

class TrajectoryEstimator(Node):
    def __init__(self):
        super().__init__('trajectory_estimator')

        self.follower_gps = None

        self.follower_initialized = True
        self.odom_initialized = False
        self.odom_origin_x = 0.0
        self.odom_origin_y = 0.0
        self.last_odom_time = None
        self.last_x_clean = 0.0
        self.last_y_clean = 0.0
        self.follower_odom_traj_clean = []

        self.imu_initialized = False
        self.imu_position = np.zeros(2)
        self.imu_velocity = np.zeros(2)
        self.last_imu_time = None
        self.imu_traj = []

        self.noise_std = 0.0
        self.bias_x = 0.0
        self.bias_y = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(NavSatFix, '/lolo/standard/navsatfix', self.cb_gps, 10)
        self.create_subscription(Odometry, '/lolo/smarc/odom', self.cb_odom, 10)
        self.create_subscription(Imu, '/lolo/standard/imu', self.cb_imu, 10)

    def cb_gps(self, msg):
        # Always store GPS as UTM for internal use
        if msg.latitude != 0.0 and msg.longitude != 0.0:
            x, y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
            self.follower_gps = [(x, y)]

    def cb_odom(self, msg):
        if not self.follower_initialized:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.odom_initialized:
            if not self.follower_gps:
                return
            x, y = self.follower_gps[0]
            self.odom_origin_x = x - msg.pose.pose.position.x
            self.odom_origin_y = y - msg.pose.pose.position.y
            self.odom_initialized = True
            self.last_odom_time = t
            self.last_x_clean = msg.pose.pose.position.x + self.odom_origin_x
            self.last_y_clean = msg.pose.pose.position.y + self.odom_origin_y
            self.get_logger().info("Odometry initialized from GPS (UTM).")
            return
        x_clean = msg.pose.pose.position.x + self.odom_origin_x
        y_clean = msg.pose.pose.position.y + self.odom_origin_y
        self.follower_odom_traj_clean.append((x_clean, y_clean))
        self.last_x_clean = x_clean
        self.last_y_clean = y_clean
        self.last_odom_time = t

    def cb_imu(self, msg):
        if not self.odom_initialized or not self.follower_gps:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            tf = self.tf_buffer.lookup_transform('utm', 'lolo/base_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return
        acc_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            0.0
        ])
        q = tf.transform.rotation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        # Use rotation.apply instead of matmul for vector rotation
        acc_utm = rotation.apply(acc_body)
        if not self.imu_initialized:
            x, y = self.follower_gps[0]
            self.imu_position = np.array([x, y])
            self.last_imu_time = t
            self.imu_initialized = True
            self.get_logger().info("IMU preintegration initialized from GPS (UTM).")
            return
        dt = t - self.last_imu_time
        if dt <= 0 or dt > 1.0:
            return
        self.imu_velocity += acc_utm[:2] * dt
        self.imu_position += self.imu_velocity * dt
        self.imu_traj.append(tuple(self.imu_position))
        self.last_imu_time = t

    def plot_trajectories(self):
        if not self.follower_odom_traj_clean or not self.imu_traj:
            self.get_logger().info("Not enough data to plot.")
            return

        odom_x, odom_y = zip(*self.follower_odom_traj_clean)
        imu_x, imu_y = zip(*self.imu_traj)

        plt.figure()
        plt.plot(odom_x, odom_y, label='Clean Odometry (UTM)', linewidth=2)
        plt.plot(imu_x, imu_y, label='IMU Preintegration', linestyle='--', linewidth=2)
        plt.xlabel('UTM X')
        plt.ylabel('UTM Y')
        plt.title('Odometry vs IMU Preintegration')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.plot_trajectories()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()