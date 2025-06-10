# Acoustic EKF ROS 2 Package

This package provides real-time localization for a vehicle using an Extended Kalman Filter (EKF) based on acoustic distance measurements, GPS, and IMU data. The EKF state and update logic are based on the definitions in `ekf_test.py`.

## Subscribed Topics
- `/follower/leader1/distance` (std_msgs/Float64): Distance to leader 0
- `/follower/leader2/distance` (std_msgs/Float64): Distance to leader 1
- `/follower/core/imu` (sensor_msgs/Imu): IMU data
- `/follower/leader1/core/gps` (sensor_msgs/NavSatFix): GPS fix for leader 0
- `/follower/leader2/core/gps` (sensor_msgs/NavSatFix): GPS fix for leader 1

## Published Topics
- `/follower/ekf/state` (custom message or std_msgs/Float64MultiArray): EKF estimated state

## Usage
1. Build the package with colcon:
   ```bash
   colcon build --packages-select acoustic_ekf_pkg
   source install/setup.bash
   ```
2. Run the EKF node:
   ```bash
   ros2 run acoustic_ekf_pkg ekf_node.py
   ```

## Node Details
- The node runs prediction at a fixed interval (e.g., 10 Hz).
- Updates are performed when new distance or IMU data is received.
- The EKF state is published after each update.

## Requirements
- ROS 2 (Foxy or newer)
- Python 3.8+
- numpy, rclpy, std_msgs, sensor_msgs

---

For more details, see the code in `ekf_node.py`.
