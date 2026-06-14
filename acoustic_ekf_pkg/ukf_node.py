#!/usr/bin/env python3
"""Unscented Kalman Filter (UKF) node for acoustic follower localization.

This node estimates the relative positions/velocities of two leader vehicles and
one follower vehicle from:
  - leader GPS (NavSatFix / GeoPoint)  -> leader position pseudo-measurements
  - acoustic range (Float32)           -> leader<->follower distance measurements
  - follower heading (Float32)         -> follower velocity-direction measurement
  - IMU / follower GPS timestamps      -> drive the constant-velocity prediction

State vector (12), all in metres relative to the follower's UTM origin:
    [x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, vfx, vfy]

The process model is constant-velocity (WNOA process noise). The measurement
models (acoustic range and heading) are non-linear, which is exactly where the
UKF helps: instead of linearising with Jacobians (as the EKF does), it
propagates a deterministic set of sigma points through the non-linear functions.

Subscriptions and topic types are identical to ekf_node.py so the two nodes are
drop-in interchangeable.

================================================================================
ROS INTERFACE
================================================================================
All topic names below are the defaults from config/ukf_config.yaml; every name is
configurable in the YAML, and relative names (no leading '/') are prefixed with
the ``follower_ns`` parameter.

Parameters
----------
  config_file  (str)  : YAML file under config/ to load. Default 'ukf_config.yaml'.
  follower_ns  (str)  : namespace prefix for relative topic names. Default ''.

Subscriptions
-------------
  /leader1/distance            std_msgs/Float32      acoustic slant range to leader 1
  /leader2/distance            std_msgs/Float32      acoustic slant range to leader 2
  /leader1/gps                 sensor_msgs/NavSatFix leader 1 GPS  (or geographic_msgs/GeoPoint
                                                      if leader_gps_type='GeoPoint')
  /leader2/gps                 sensor_msgs/NavSatFix leader 2 GPS  (or GeoPoint)
  /lolo/standard/navsatfix     sensor_msgs/NavSatFix follower GPS  (or GeoPoint if
                                                      follower_gps_type='GeoPoint');
                                                      also triggers a prediction step
  /lolo/standard/imu           sensor_msgs/Imu       timestamp source for prediction steps
  /lolo/smarc/depth            std_msgs/Float32      follower depth (slant->horizontal range)
  /lolo/smarc/heading          std_msgs/Float32      external heading; subscribed ONLY when
                                                      use_heading_updates=true (off by default)

Publications
------------
  /follower/ukf/state            std_msgs/Float32MultiArray
                                   full 12-element state vector (relative UTM metres)
  /follower/ukf/navsatfix          sensor_msgs/NavSatFix  follower lat/lon (baselink)
  /follower/ukf/navsatfix_leader1  sensor_msgs/NavSatFix  leader 1 lat/lon
  /follower/ukf/navsatfix_leader2  sensor_msgs/NavSatFix  leader 2 lat/lon
  /follower/ukf/navsatfix_centroid sensor_msgs/NavSatFix  centroid of the three vehicles
  /follower/ukf/pose             geometry_msgs/PoseWithCovarianceStamped
                                   follower pose in absolute UTM metres (frame 'utm') with the
                                   position covariance block P[8:10, 8:10]; yaw from estimated heading
  /follower/ukf/setpoint         geographic_msgs/GeoPoint  desired follower position (lat/lon/alt):
                                   apex of the semicircle behind the two leaders (diameter = leader1<->leader2)
  /follower/ukf/delta_pos        std_msgs/Float32       distance between estimate and follower GPS truth

TF
--
  Listens to the /tf tree (utm <-> lolo/base_link) via a TransformListener (reserved
  for future use; not required for the current estimator).
================================================================================
"""

import os
import math
import threading

import numpy as np
import yaml
import utm

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped
from geographic_msgs.msg import GeoPoint
from tf2_ros import Buffer, TransformListener


class UKF:
    """Unscented Kalman Filter for 2 leaders + 1 follower (12-state).

    Computations are done in float64 for numerical stability (the unscented
    transform relies on a Cholesky factorisation of the covariance, which is
    sensitive to round-off in float32).
    """

    N = 12  # state dimension

    def __init__(self, initial_state, Q, R, dt=0.0, max_velocity=500.0,
                 alpha=1.0, beta=2.0, kappa=0.0):
        self.dt = float(dt)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.max_velocity = float(max_velocity)

        self.x = initial_state.reshape((self.N, 1)).astype(np.float64)
        self.P = np.eye(self.N, dtype=np.float64) * 4.0

        # Van der Merwe scaled unscented-transform weights.
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lambda_ = self.alpha ** 2 * (self.N + self.kappa) - self.N
        self._gamma = math.sqrt(self.N + self.lambda_)  # sigma-point spread

        c = self.N + self.lambda_
        self.Wm = np.full(2 * self.N + 1, 1.0 / (2.0 * c))
        self.Wc = np.full(2 * self.N + 1, 1.0 / (2.0 * c))
        self.Wm[0] = self.lambda_ / c
        self.Wc[0] = self.lambda_ / c + (1.0 - self.alpha ** 2 + self.beta)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _wrap(angle):
        """Wrap an angle to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _symmetrize(self):
        self.P = 0.5 * (self.P + self.P.T)

    def _clip_velocities(self):
        """Clip all velocity components to +/- max_velocity (parity with EKF)."""
        for idx in (2, 3, 6, 7, 10, 11):
            self.x[idx] = np.clip(self.x[idx], -self.max_velocity, self.max_velocity)

    def _sigma_points(self):
        """Generate the 2N+1 sigma points from the current (x, P)."""
        P = 0.5 * (self.P + self.P.T)
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Numerical drift broke positive-definiteness: repair to the nearest
            # PSD matrix by clamping eigenvalues, then refactor.
            w, V = np.linalg.eigh(P)
            w = np.clip(w, 1e-9, None)
            P = (V * w) @ V.T
            self.P = P
            S = np.linalg.cholesky(P + np.eye(self.N) * 1e-9)

        x = self.x.flatten()
        sigmas = np.zeros((2 * self.N + 1, self.N))
        sigmas[0] = x
        for i in range(self.N):
            sigmas[i + 1] = x + self._gamma * S[:, i]
            sigmas[self.N + i + 1] = x - self._gamma * S[:, i]
        return sigmas

    # ------------------------------------------------------------------ models
    def _fx(self, s):
        """Constant-velocity process model for a single sigma point."""
        dt = self.dt
        out = s.copy()
        out[0] = s[0] + s[2] * dt   # x1
        out[1] = s[1] + s[3] * dt   # y1
        out[4] = s[4] + s[6] * dt   # x2
        out[5] = s[5] + s[7] * dt   # y2
        out[8] = s[8] + s[10] * dt  # xf
        out[9] = s[9] + s[11] * dt  # yf
        return out

    @staticmethod
    def _hx_leader(s, leader_id):
        """Measurement model: [leader_x, leader_y, range(leader, follower)]."""
        lx = s[leader_id * 4]
        ly = s[leader_id * 4 + 1]
        dist = math.hypot(lx - s[8], ly - s[9])
        return np.array([lx, ly, dist])

    # ------------------------------------------------------------------ steps
    def predict(self):
        """Unscented prediction step using the current dt and Q."""
        sigmas = self._sigma_points()
        prop = np.array([self._fx(s) for s in sigmas])

        xmean = self.Wm @ prop
        P = self.Q.copy()
        for i in range(prop.shape[0]):
            d = (prop[i] - xmean).reshape(-1, 1)
            P += self.Wc[i] * (d @ d.T)

        self.x = xmean.reshape(-1, 1)
        self.P = P
        self._symmetrize()
        self._clip_velocities()
        return self.x.flatten(), self.P

    def update(self, z, leader_id):
        """Range/position update for the given leader (z = [lx, ly, dist]).

        This step uses an EXTENDED-Kalman (analytic Jacobian) update rather than
        the unscented form. The acoustic range ``dist = ||leader - follower||`` is
        nonlinear AND ambiguous: two leaders give a two-circle intersection with a
        mirror solution. The unscented transform spreads sigma points by
        ``sqrt(N + lambda) * sqrt(P)`` (~5-7 m here), which straddles both
        branches; the resulting mean/cross-covariance get contaminated by the
        mirror solution and walk the follower onto the wrong (ghost) intersection,
        especially under the degenerate same-direction leader geometry seen in the
        field data. The EKF linearises locally, so the update always steps along
        the local gradient and stays on the correct branch. The constant-velocity
        prediction remains unscented (it is exact for a linear model).
        """
        z = np.asarray(z, dtype=np.float64).flatten()
        fx = self.x.flatten()
        lx = fx[leader_id * 4]
        ly = fx[leader_id * 4 + 1]
        xf = fx[8]
        yf = fx[9]
        dist = math.hypot(lx - xf, ly - yf)
        if dist < 1e-6:
            dist = 1e-6

        # Measurement Jacobian H (3 x N) for h(x) = [lx, ly, ||leader-follower||].
        H = np.zeros((3, self.N), dtype=np.float64)
        H[0, leader_id * 4] = 1.0
        H[1, leader_id * 4 + 1] = 1.0
        H[2, leader_id * 4] = (lx - xf) / dist
        H[2, leader_id * 4 + 1] = (ly - yf) / dist
        H[2, 8] = (xf - lx) / dist
        H[2, 9] = (yf - ly) / dist

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        zhat = np.array([lx, ly, dist])
        self.x = self.x + K @ (z - zhat).reshape(-1, 1)
        self.P = (np.eye(self.N) - K @ H) @ self.P
        self._symmetrize()
        self._clip_velocities()
        return self.x.flatten()

    def heading_update(self, heading_deg, heading_noise_deg=10.0):
        """Heading update: measures the direction of the follower velocity.

        Measurement model (NED convention, 0=North, 90=East):
            h(x) = atan2(vfx, vfy)
        Angles are handled circularly (sigma-point mean via sin/cos, residuals
        wrapped to [-pi, pi]).
        """
        vfx = float(self.x[10])
        vfy = float(self.x[11])
        if math.hypot(vfx, vfy) < 1e-3:
            return self.x.flatten()  # direction is unobservable at near-zero speed

        z = math.radians(heading_deg)
        sigmas = self._sigma_points()
        Zsig = np.array([math.atan2(s[10], s[11]) for s in sigmas])

        # Circular (weighted) mean of the predicted heading sigma points.
        zmean = math.atan2(float(self.Wm @ np.sin(Zsig)),
                           float(self.Wm @ np.cos(Zsig)))

        S = math.radians(heading_noise_deg) ** 2
        Cxz = np.zeros((self.N, 1))
        xmean = self.x.flatten()
        for i in range(Zsig.shape[0]):
            dz = self._wrap(Zsig[i] - zmean)
            dx = (sigmas[i] - xmean).reshape(-1, 1)
            S += self.Wc[i] * dz * dz
            Cxz += self.Wc[i] * dx * dz

        K = Cxz / S
        innovation = self._wrap(z - zmean)
        self.x = self.x + K * innovation
        self.P = self.P - S * (K @ K.T)
        self._symmetrize()
        self._clip_velocities()
        return self.x.flatten()

    def get_state(self):
        return self.x.flatten()

    def get_covariance(self):
        return self.P


class AcousticUKFNode(Node):
    """ROS 2 node wrapping the UKF. Mirrors AcousticEKFNode I/O exactly."""

    def __init__(self):
        super().__init__('acoustic_ukf_node')

        # Declare parameters
        self.declare_parameter('config_file', 'ukf_config.yaml')
        self.declare_parameter('follower_ns', '')

        config_file = self.get_parameter('config_file').value
        follower_ns = self.get_parameter('follower_ns').value

        # Load UKF config, resolving the path robustly (installed share dir,
        # source tree, or an absolute path given via the parameter).
        config_path = self._resolve_config_path(config_file)
        self.get_logger().info(f'Loading config from: {config_path}')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Configuration parameters
        process_noise = config.get('process_noise', 4.0)
        measurement_noise = config.get('measurement_noise', 2.0)
        distance_measurement_noise = config.get('distance_measurement_noise', 0.1)
        self.process_noise = process_noise  # stored for dynamic Q updates
        self.leader_init_samples = config.get('leader_init_samples', 1)
        self.follower_init_samples = config.get('follower_init_samples', 1)
        self.max_speed = config.get('max_speed', 500.0)
        self.dt = 0
        self.correction_factor = config.get('correction_factor', 0.9824)
        # Constant range-bias correction (metres) added to every acoustic range
        # to de-bias the OWTT measurement (e.g. clock-sync / sound-speed offset).
        # Use a single global value: different per-leader biases make the two
        # ranges geometrically inconsistent and destabilise the filter.
        self.range_bias = float(config.get('range_bias', 0.0))

        # Boat (antenna/GPS vs transducer) offset in UTM metres [dx, dy].
        self.boat_offset = np.array(config.get('boat_offset', [0.0, -0.6]), dtype=np.float32)

        # Modem<->baselink offset (the UKF estimates modem position; we transform
        # back to baselink for publishing).
        self.modem_to_baselink_offset_x = config.get('modem_to_baselink_offset_x', 1.96)
        self.modem_to_baselink_offset_y = config.get('modem_to_baselink_offset_y', 0.0)
        self.modem_to_baselink_offset_z = config.get('modem_to_baselink_offset_z', 0.475)

        # Heading update parameters
        self.heading_measurement_noise = config.get('heading_measurement_noise', 10.0)  # deg
        self.heading_perturbation = config.get('heading_perturbation', 0.0)  # deg (+/- uniform)
        self.use_heading_updates = config.get('use_heading_updates', True)
        self.current_heading_deg = None
        # Minimum follower speed (m/s) for the filter-estimated heading (used by
        # the modem->baselink offset correction) to be considered reliable.
        self.heading_min_speed = config.get('heading_min_speed', 0.2)

        # Setpoint generation: weight bias toward leader 1's orientation estimate
        # when blending the two leaders' headings to decide the "behind" side.
        # Must be in [0, 1]; 0.5 = equal trust, 1.0 = leader 1 only.
        self.setpoint_leader1_weight = float(config.get('setpoint_leader1_weight', 0.7))

        # UKF unscented-transform tuning parameters.
        # alpha=1.0, kappa=0.0 -> lambda=0, which is numerically well-conditioned
        # for this 12-dim state (small alpha makes the central weight blow up and
        # can destroy positive-definiteness of the covariance).
        self.ukf_alpha = config.get('ukf_alpha', 1.0)
        self.ukf_beta = config.get('ukf_beta', 2.0)
        self.ukf_kappa = config.get('ukf_kappa', 0.0)

        # Build full topic paths from config (relative topics + follower namespace).
        def build_topic(relative_topic):
            if not relative_topic:
                return ''
            if relative_topic.startswith('/'):
                return relative_topic  # already absolute
            if follower_ns:
                return f'/{follower_ns}/{relative_topic}'
            return f'/{relative_topic}'

        publish_topic = build_topic(config.get('publish_topic', 'follower/ukf/state'))
        # NavSatFix outputs (follower + leaders + centroid). The base name gets
        # the '_leader1' / '_leader2' / '_centroid' suffixes appended below.
        navsatfix_topic = build_topic(config.get('navsatfix_topic', 'follower/ukf/navsatfix'))
        dist1_topic = build_topic(config.get('dist1_topic', 'follower/leader1/distance'))
        dist2_topic = build_topic(config.get('dist2_topic', 'follower/leader2/distance'))
        gps1_topic = build_topic(config.get('gps1_topic', '/leader1/smarc/latlon'))
        gps2_topic = build_topic(config.get('gps2_topic', '/leader2/smarc/latlon'))
        follower_gps_topic = build_topic(config.get('follower_gps_topic', 'smarc/latlon'))
        delta_pos_topic = build_topic(config.get('delta_pos_topic', 'ukf/delta_pos'))
        pose_cov_topic = build_topic(config.get('pose_cov_topic', 'follower/ukf/pose'))
        setpoint_topic = build_topic(config.get('setpoint_topic', 'follower/ukf/setpoint'))
        imu_topic = build_topic(config.get('imu_topic', 'core/imu'))
        leader_gps_type = config.get('leader_gps_type', 'NavSatFix')
        self.leader_gps_type = leader_gps_type
        follower_gps_type = config.get('follower_gps_type', 'NavSatFix')

        # Publishers
        self.state_pub = self.create_publisher(Float32MultiArray, publish_topic, 10)
        self.navsatfix_pub = self.create_publisher(NavSatFix, navsatfix_topic, 10)
        self.navsatfix_pub_leader1 = self.create_publisher(NavSatFix, navsatfix_topic + "_leader1", 10)
        self.navsatfix_pub_leader2 = self.create_publisher(NavSatFix, navsatfix_topic + "_leader2", 10)
        self.navsatfix_pub_centroid = self.create_publisher(NavSatFix, navsatfix_topic + "_centroid", 10)
        self.delta_pos_pub = self.create_publisher(Float32, delta_pos_topic, 10)
        self.pose_cov_pub = self.create_publisher(PoseWithCovarianceStamped, pose_cov_topic, 10)
        # Setpoint is a target coordinate -> geographic_msgs/GeoPoint (lat/lon/alt).
        self.setpoint_pub = self.create_publisher(GeoPoint, setpoint_topic, 10)

        # Thread safety
        self.lock = threading.Lock()

        # UTM reference zone (set from the first GPS message).
        self.utm_zone_number = None
        self.utm_zone_letter = None

        # Initialization tracking
        self.leader_gps_samples = {1: [], 2: []}
        self.follower_gps_samples = []
        self.initialization_phase = 'collecting_leaders'  # -> 'collecting_follower' -> 'ready'
        self.follower_utm_offset = None
        self.last_truth_follower_utm = None

        # Latest measurements in UTM coordinates
        self.current_leader_positions = {1: None, 2: None}
        self.last_distances = {1: None, 2: None}

        # UKF instance
        self.ukf = None

        # Subscriptions (identical to ekf_node.py)
        self.create_subscription(Float32, dist1_topic, self.dist1_callback, 10)
        self.create_subscription(Float32, dist2_topic, self.dist2_callback, 10)
        if leader_gps_type == 'GeoPoint':
            self.create_subscription(GeoPoint, gps1_topic, self.gps1_geopoint_callback, 10)
            self.create_subscription(GeoPoint, gps2_topic, self.gps2_geopoint_callback, 10)
        else:
            self.create_subscription(NavSatFix, gps1_topic, self.gps1_callback, 10)
            self.create_subscription(NavSatFix, gps2_topic, self.gps2_callback, 10)

        if follower_gps_type == 'GeoPoint':
            self.create_subscription(GeoPoint, follower_gps_topic, self.follower_gps_geopoint_callback, 10)
        else:
            self.create_subscription(NavSatFix, follower_gps_topic, self.follower_gps_callback, 10)

        # Always subscribe to the follower heading so we can place the modem
        # correctly at initialization (and, if enabled, run heading updates).
        self.follower_heading_deg = None
        heading_topic = '/lolo/smarc/heading'
        self.create_subscription(Float32, heading_topic, self.heading_callback, 10)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 10)

        # Noise matrices.
        # Q: WNOA process noise (block-diagonal, 3 x 4x4 blocks for [x, y, vx, vy]).
        self.Q = self.make_WNOA(process_noise, dt=max(self.dt, 0.1))
        # R: leader x/y use measurement_noise^2, distance uses its own noise.
        self.R = np.eye(3, dtype=np.float32)
        self.R[0, 0] = measurement_noise ** 2
        self.R[1, 1] = measurement_noise ** 2
        self.R[2, 2] = distance_measurement_noise ** 2

        self.update_cooldown = config.get('update_cooldown', 0.0)
        self.last_update_time = self.get_clock().now().nanoseconds / 1e9 - self.update_cooldown

        self.leader_depth = config.get('leader_depth', 1.0)
        depth_topic = build_topic(config.get('depth_topic', 'smarc/depth'))
        self.depth = None
        self.create_subscription(Float32, depth_topic, self.depth_callback, 10)

        # TF2 transform listener (kept for parity / future use).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.imu_frame = 'lolo/base_link'
        self.utm_frame = 'utm'

        self.last_predict_time = None  # for variable-dt prediction

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.get_logger().info('AcousticUKFNode initialized, collecting leader GPS samples...')

    def _resolve_config_path(self, config_file):
        """Locate the config YAML across install and source layouts.

        Tries, in order: an absolute path; the installed package share dir
        (proper ROS location); and the source tree relative to this file.
        """
        if os.path.isabs(config_file):
            return config_file

        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('acoustic_ekf_pkg'), 'config', config_file))
        except Exception:
            pass
        candidates.append(os.path.join(os.path.dirname(__file__), '../config', config_file))

        for path in candidates:
            if os.path.exists(path):
                return path
        # Nothing found: return the best candidate so the error names a real path.
        return candidates[0]

    # ----------------------------------------------------------- UTM helpers
    def gps_to_utm(self, lat, lon):
        """Convert GPS to UTM, fixing the reference zone on first conversion."""
        try:
            x, y, zone_number, zone_letter = utm.from_latlon(float(lat), float(lon))
            if self.utm_zone_number is None:
                self.utm_zone_number = zone_number
                self.utm_zone_letter = zone_letter
                self.get_logger().info(f'UTM reference zone set to: {zone_number}{zone_letter}')
            return float(f"{x:.8f}"), float(f"{y:.8f}")
        except Exception as e:
            self.get_logger().error(f'UTM conversion failed: {e}')
            return None, None

    def utm_to_gps(self, x, y):
        """Convert UTM coordinates back to GPS."""
        if self.utm_zone_number is None or self.utm_zone_letter is None:
            return None, None
        try:
            lat, lon = utm.to_latlon(float(x), float(y), int(self.utm_zone_number), str(self.utm_zone_letter))
            return float(f"{lat:.12f}"), float(f"{lon:.12f}")
        except Exception as e:
            self.get_logger().error(f'UTM to GPS conversion failed: {e}')
            return None, None

    # ----------------------------------------------------------- leader GPS
    def _handle_leader_gps(self, leader_id, lat, lon):
        """Shared logic for leader GPS (NavSatFix or GeoPoint)."""
        if self.initialization_phase == 'collecting_leaders':
            self.leader_gps_samples[leader_id].append((lat, lon))
            self.get_logger().info(
                f'Leader {leader_id} GPS sample '
                f'{len(self.leader_gps_samples[leader_id])}/{self.leader_init_samples}: '
                f'lat={lat:.6f}, lon={lon:.6f}')
            self._check_leader_initialization()
        elif self.initialization_phase == 'ready':
            x, y = self.gps_to_utm(lat, lon)
            if x is not None and y is not None and self.follower_utm_offset is not None:
                offset_x = x - self.follower_utm_offset[0] - self.boat_offset[0]
                offset_y = y - self.follower_utm_offset[1] - self.boat_offset[1]
                self.current_leader_positions[leader_id] = (offset_x, offset_y)
            else:
                self.get_logger().warn(
                    f'Follower UTM offset not initialized or UTM conversion failed for leader {leader_id}!')

    def gps1_callback(self, msg):
        self._handle_leader_gps(1, msg.latitude, msg.longitude)

    def gps2_callback(self, msg):
        self._handle_leader_gps(2, msg.latitude, msg.longitude)

    def gps1_geopoint_callback(self, msg):
        self._handle_leader_gps(1, msg.latitude, msg.longitude)

    def gps2_geopoint_callback(self, msg):
        self._handle_leader_gps(2, msg.latitude, msg.longitude)

    # ----------------------------------------------------------- follower GPS
    def follower_gps_callback(self, msg):
        """Follower GPS (NavSatFix): init samples + variable-dt prediction trigger."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._handle_follower_gps(msg.latitude, msg.longitude, t)

    def follower_gps_geopoint_callback(self, msg):
        """Follower GPS (GeoPoint): uses ROS clock for dt (no header stamp)."""
        t = self.get_clock().now().nanoseconds / 1e9
        self._handle_follower_gps(msg.latitude, msg.longitude, t)

    def _handle_follower_gps(self, lat, lon, t):
        if self.initialization_phase == 'collecting_follower':
            self.follower_gps_samples.append((lat, lon))
            self.get_logger().info(
                f'Follower GPS sample {len(self.follower_gps_samples)}/{self.follower_init_samples}: '
                f'lat={lat:.6f}, lon={lon:.6f}')
            if len(self.follower_gps_samples) >= self.follower_init_samples:
                self._initialize_ukf()
                self.last_predict_time = t
        elif self.initialization_phase == 'ready' and self.ukf is not None:
            if self.last_predict_time is not None:
                dt = t - self.last_predict_time
                if 0.0 < dt < 2.0:
                    with self.lock:
                        gx, gy = self.gps_to_utm(lat, lon)
                        if gx is not None and gy is not None:
                            self.last_truth_follower_utm = (gx, gy)
                        self.ukf.dt = float(dt)
                        self.ukf.Q = self.make_WNOA(self.process_noise, dt)
                        self.ukf.predict()
                        self._publish_outputs()
                elif dt >= 2.0:
                    self.get_logger().warn(f'Large dt gap: {dt:.3f}s, skipping prediction')
            self.last_predict_time = t
        else:
            gx, gy = self.gps_to_utm(lat, lon)
            if gx is not None and gy is not None:
                self.last_truth_follower_utm = (gx, gy)

    # ----------------------------------------------------------- distance
    def _horizontal_distance(self, slant):
        """Project slant range to horizontal using the current depth, if known."""
        if self.depth is None:
            return slant
        try:
            z_depth = max(self.depth - self.leader_depth, 0.0)
            return math.sqrt(max(slant ** 2 - z_depth ** 2, 0.0))
        except Exception as e:
            self.get_logger().error(f'Error computing horizontal distance: {e}')
            return slant

    def _handle_distance(self, leader_id, raw_distance):
        if self.initialization_phase != 'ready':
            return
        with self.lock:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_update_time < self.update_cooldown:
                return
            d = raw_distance * self.correction_factor
            self.last_distances[leader_id] = self._horizontal_distance(d) + self.range_bias
            if self.current_leader_positions[leader_id] is not None:
                z = np.array([
                    self.current_leader_positions[leader_id][0],
                    self.current_leader_positions[leader_id][1],
                    self.last_distances[leader_id],
                ], dtype=np.float32)
                self.get_logger().info(f'Updating UKF with leader {leader_id} measurement: {z}')
                self.ukf.update(z, leader_id - 1)  # UKF leader index is 0-based
                self.last_update_time = now
                self._publish_outputs()

    def dist1_callback(self, msg):
        self._handle_distance(1, msg.data)

    def dist2_callback(self, msg):
        self._handle_distance(2, msg.data)

    # ----------------------------------------------------------- heading
    def heading_callback(self, msg):
        """Store the latest follower heading; optionally run a UKF heading update.

        The heading is always recorded (used to place the follower modem relative
        to the GPS/baselink origin at initialization). The heading *update* step
        only runs when ``use_heading_updates`` is enabled.
        """
        eps = float(self.heading_perturbation)
        perturb = np.random.uniform(-eps, eps) if eps > 0 else 0.0
        self.follower_heading_deg = msg.data + perturb
        if not self.use_heading_updates:
            return
        self.current_heading_deg = self.follower_heading_deg
        if self.initialization_phase != 'ready' or self.ukf is None:
            return
        with self.lock:
            self.ukf.heading_update(self.current_heading_deg, self.heading_measurement_noise)

    # ----------------------------------------------------------- IMU / depth
    def imu_callback(self, msg):
        """IMU message triggers a constant-velocity prediction step."""
        if self.initialization_phase != 'ready' or self.ukf is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_predict_time is not None:
            dt = t - self.last_predict_time
            if dt > 0:
                with self.lock:
                    self.ukf.dt = float(dt)
                    self.ukf.Q = self.make_WNOA(self.process_noise, dt)
                    self.ukf.predict()
                    self._publish_outputs()
        self.last_predict_time = t

    def depth_callback(self, msg):
        self.depth = msg.data

    # ----------------------------------------------------------- init
    def _check_leader_initialization(self):
        if (len(self.leader_gps_samples[1]) >= self.leader_init_samples and
                len(self.leader_gps_samples[2]) >= self.leader_init_samples):
            self.initialization_phase = 'collecting_follower'
            self.get_logger().info('Leaders initialized, now collecting follower GPS samples...')

    def _initialize_ukf(self):
        """Initialize the UKF from collected GPS samples; store follower UTM offset."""
        self.get_logger().info('Initializing UKF...')
        leader1_samples = self.leader_gps_samples[1][-self.leader_init_samples:]
        leader2_samples = self.leader_gps_samples[2][-self.leader_init_samples:]
        if len(leader1_samples) < self.leader_init_samples or len(leader2_samples) < self.leader_init_samples:
            self.get_logger().error('Not enough leader GPS samples to initialize UKF')
            return
        if len(self.follower_gps_samples) < self.follower_init_samples:
            self.get_logger().error('Not enough follower GPS samples to initialize UKF')
            return

        avg_lat1 = sum(s[0] for s in leader1_samples) / len(leader1_samples)
        avg_lon1 = sum(s[1] for s in leader1_samples) / len(leader1_samples)
        avg_lat2 = sum(s[0] for s in leader2_samples) / len(leader2_samples)
        avg_lon2 = sum(s[1] for s in leader2_samples) / len(leader2_samples)
        avg_flat = sum(s[0] for s in self.follower_gps_samples) / len(self.follower_gps_samples)
        avg_flon = sum(s[1] for s in self.follower_gps_samples) / len(self.follower_gps_samples)

        leader1_x, leader1_y = self.gps_to_utm(avg_lat1, avg_lon1)
        leader2_x, leader2_y = self.gps_to_utm(avg_lat2, avg_lon2)
        follower_x, follower_y = self.gps_to_utm(avg_flat, avg_flon)
        if None in [leader1_x, leader1_y, leader2_x, leader2_y, follower_x, follower_y]:
            self.get_logger().error('Failed to convert GPS to UTM coordinates')
            return

        self.follower_utm_offset = np.array([follower_x, follower_y], dtype=np.float64)
        leader1_x_adj = leader1_x - self.boat_offset[0]
        leader1_y_adj = leader1_y - self.boat_offset[1]
        leader2_x_adj = leader2_x - self.boat_offset[0]
        leader2_y_adj = leader2_y - self.boat_offset[1]

        # Place the follower MODEM (the acoustic-range endpoint) relative to the
        # follower's GPS/baselink origin using the known compass heading. The
        # follower state tracks the modem, but the GPS origin is the baselink, so
        # without this the state would start ~modem_to_baselink_offset_x metres
        # off and have to drift to reconcile with the ranges. NED heading
        # (0=North, 90=East): forward (body +x) -> [E, N] = [sin h, cos h];
        # starboard (body +y) -> [cos h, -sin h].
        xf0 = yf0 = 0.0
        if self.follower_heading_deg is not None:
            h = math.radians(self.follower_heading_deg)
            ox = self.modem_to_baselink_offset_x
            oy = self.modem_to_baselink_offset_y
            xf0 = ox * math.sin(h) + oy * math.cos(h)
            yf0 = ox * math.cos(h) - oy * math.sin(h)
            self.get_logger().info(
                f'Init follower modem offset from heading {self.follower_heading_deg:.1f} deg: '
                f'({xf0:.2f}, {yf0:.2f}) m')
        else:
            self.get_logger().warn('No follower heading at init; modem placed at baselink origin')

        # State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, xf, yf, vfx, vfy]
        initial_state = np.array([
            leader1_x_adj - follower_x, leader1_y_adj - follower_y, 0.0, 0.0,
            leader2_x_adj - follower_x, leader2_y_adj - follower_y, 0.0, 0.0,
            xf0, yf0, 0.0, 0.0,
        ], dtype=np.float32)

        self.ukf = UKF(initial_state, Q=self.Q, R=self.R, dt=self.dt,
                       max_velocity=self.max_speed,
                       alpha=self.ukf_alpha, beta=self.ukf_beta, kappa=self.ukf_kappa)
        self.initialization_phase = 'ready'
        self.get_logger().info(f'UKF initialized with UTM offset: {self.follower_utm_offset}')
        self.get_logger().info('UKF ready for operation!')

    # ----------------------------------------------------------- publishing
    def _publish_navsat(self, publisher, lat, lon):
        msg = NavSatFix()
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.status.status = 0
        msg.status.service = 1
        publisher.publish(msg)

    def _publish_pose_with_cov(self, follower_x, follower_y, heading_deg):
        """Publish the follower pose with covariance for downstream subscribers.

        Position is in absolute UTM metres (easting=x, northing=y) in the 'utm'
        frame. The 6x6 pose covariance is filled from the filter's follower
        position block P[8:10, 8:10] (x, y); z and the orientation axes are
        marked unknown (large variance) since they are not estimated. Orientation
        encodes the estimated heading as yaw (ENU: yaw measured CCW from east).
        """
        if self.ukf is None:
            return
        P = self.ukf.get_covariance()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.utm_frame  # 'utm'
        msg.pose.pose.position.x = float(follower_x)
        msg.pose.pose.position.y = float(follower_y)
        msg.pose.pose.position.z = 0.0

        # Orientation from estimated heading. NED heading (0=N, CW) -> ENU yaw
        # (0=E, CCW): yaw_enu = pi/2 - heading_ned.
        if heading_deg is not None:
            yaw = math.pi / 2.0 - math.radians(heading_deg)
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            msg.pose.pose.orientation.w = 1.0  # identity

        # 6x6 row-major covariance: order [x, y, z, rot_x, rot_y, rot_z].
        UNKNOWN = 1e6  # large variance flags an unestimated DoF
        cov = [0.0] * 36
        cov[0] = float(P[8, 8])    # xx
        cov[1] = float(P[8, 9])    # xy
        cov[6] = float(P[9, 8])    # yx
        cov[7] = float(P[9, 9])    # yy
        cov[14] = UNKNOWN          # zz
        cov[21] = UNKNOWN          # rot_x
        cov[28] = UNKNOWN          # rot_y
        cov[35] = UNKNOWN          # rot_z (yaw not formally estimated here)
        msg.pose.covariance = cov

        self.pose_cov_pub.publish(msg)

    def _publish_setpoint(self, state, l1x, l1y, l2x, l2y, fx, fy):
        """Publish the desired follower position (GeoPoint) behind the leaders.

        Geometry: treat the segment leader1<->leader2 as the diameter of a circle.
        Its midpoint is the circle centre and its half-length is the radius. The
        setpoint is the apex of the semicircle on the side *behind* the leaders,
        i.e. centre + radius * n_hat, where n_hat is the unit normal to the
        diameter pointing backwards relative to the leaders' direction of travel.

        The "behind" side is chosen from the leaders' estimated orientations
        (direction of their estimated velocities), trusting leader 1 more
        (``setpoint_leader1_weight``). If neither leader's orientation is
        reliable (both nearly stationary), we fall back to the side the follower
        is currently estimated to be on, so the setpoint stays sensible.
        """
        p1 = np.array([l1x, l1y])
        p2 = np.array([l2x, l2y])
        centre = 0.5 * (p1 + p2)
        diameter = p2 - p1
        d_len = float(np.linalg.norm(diameter))
        if d_len < 1e-6:
            return  # leaders coincide; arc is undefined
        radius = 0.5 * d_len

        # Two unit normals to the diameter; pick the "behind" one below.
        d_hat = diameter / d_len
        n_hat = np.array([-d_hat[1], d_hat[0]])

        # Leaders' forward direction from estimated velocities (east, north),
        # blended with a bias toward leader 1.
        v1 = np.array([state[2], state[3]])
        v2 = np.array([state[6], state[7]])
        s1 = float(np.linalg.norm(v1))
        s2 = float(np.linalg.norm(v2))
        w1 = self.setpoint_leader1_weight
        w2 = 1.0 - w1
        forward = np.zeros(2)
        if s1 >= self.heading_min_speed:
            forward += w1 * (v1 / s1)
        if s2 >= self.heading_min_speed:
            forward += w2 * (v2 / s2)

        if float(np.linalg.norm(forward)) > 1e-6:
            # "Behind" = opposite the leaders' heading. Choose the normal whose
            # direction aligns with backward (n_hat . (-forward) > 0).
            if float(np.dot(n_hat, -forward)) < 0.0:
                n_hat = -n_hat
        else:
            # No reliable orientation: pick the side the follower is on.
            if float(np.dot(n_hat, np.array([fx, fy]) - centre)) < 0.0:
                n_hat = -n_hat

        setpoint = centre + radius * n_hat
        if abs(setpoint[0]) > 1e7 or abs(setpoint[1]) > 1e7:
            return
        sp_lat, sp_lon = self.utm_to_gps(float(setpoint[0]), float(setpoint[1]))
        if sp_lat is not None and sp_lon is not None:
            geo = GeoPoint()
            geo.latitude = sp_lat
            geo.longitude = sp_lon
            geo.altitude = 0.0
            self.setpoint_pub.publish(geo)

    def _estimate_heading_deg(self, state):
        """Heading (deg, NED) from the estimated follower velocity direction.

        Returns the last good estimate when the follower is slower than
        ``heading_min_speed`` (velocity direction is unreliable near zero speed),
        or None if no estimate is available yet.
        """
        vfx = float(state[10])
        vfy = float(state[11])
        if math.hypot(vfx, vfy) >= self.heading_min_speed:
            # NED convention: 0=North, 90=East -> atan2(east, north) = atan2(vfx, vfy)
            self.current_heading_deg = math.degrees(math.atan2(vfx, vfy))
        return self.current_heading_deg

    def _publish_outputs(self):
        """Publish UKF state, NavSatFix positions, pose, setpoint and delta."""
        if self.ukf is None or self.follower_utm_offset is None:
            self.get_logger().warn('Cannot publish: UKF or follower_utm_offset not initialized!')
            return
        state = self.ukf.get_state()

        state_msg = Float32MultiArray()
        state_msg.data = state.astype(np.float32).tolist()
        self.state_pub.publish(state_msg)

        # Follower modem position in absolute UTM.
        follower_modem_x = state[8] + self.follower_utm_offset[0]
        follower_modem_y = state[9] + self.follower_utm_offset[1]

        # Reverse the modem->baselink offset using the heading estimated by the
        # filter itself (direction of the estimated follower velocity, NED:
        # heading = atan2(vfx, vfy)). The velocity direction is only meaningful
        # when the follower is actually moving, so below a minimum speed we reuse
        # the last good heading estimate (and skip the offset on the very first
        # samples if none is available yet).
        heading_deg = self._estimate_heading_deg(state)
        if heading_deg is not None:
            heading_rad = math.radians(heading_deg)
            follower_x = follower_modem_x - self.modem_to_baselink_offset_x * math.sin(heading_rad)
            follower_y = follower_modem_y - self.modem_to_baselink_offset_x * math.cos(heading_rad)
        else:
            self.get_logger().warn(
                'No heading estimate yet (follower not moving), publishing modem position',
                throttle_duration_sec=5.0)
            follower_x = follower_modem_x
            follower_y = follower_modem_y

        if abs(follower_x) > 1e7 or abs(follower_y) > 1e7:
            self.get_logger().warn(
                f'Follower UTM values out of range: x={follower_x}, y={follower_y}. Skipping publish.')
            return

        follower_lat, follower_lon = self.utm_to_gps(follower_x, follower_y)
        if follower_lat is not None and follower_lon is not None:
            self._publish_navsat(self.navsatfix_pub, follower_lat, follower_lon)

        # Publish follower pose + covariance (absolute UTM metres, frame 'utm').
        self._publish_pose_with_cov(follower_x, follower_y, heading_deg)

        # Delta to ground truth.
        if self.last_truth_follower_utm is not None:
            truth_x, truth_y = self.last_truth_follower_utm
            delta = math.sqrt((follower_x - truth_x) ** 2 + (follower_y - truth_y) ** 2)
            delta_msg = Float32()
            delta_msg.data = float(delta)
            self.delta_pos_pub.publish(delta_msg)

        # Leader 1 absolute position.
        leader1_x = state[0] + self.follower_utm_offset[0] + self.boat_offset[0]
        leader1_y = state[1] + self.follower_utm_offset[1] + self.boat_offset[1]
        leader1_ok = abs(leader1_x) <= 1e7 and abs(leader1_y) <= 1e7
        if leader1_ok:
            leader1_lat, leader1_lon = self.utm_to_gps(leader1_x, leader1_y)
            if leader1_lat is not None and leader1_lon is not None:
                self._publish_navsat(self.navsatfix_pub_leader1, leader1_lat, leader1_lon)
        else:
            self.get_logger().warn(
                f'Leader1 UTM values out of range: x={leader1_x}, y={leader1_y}. Skipping publish.')

        # Leader 2 absolute position.
        leader2_x = state[4] + self.follower_utm_offset[0] + self.boat_offset[0]
        leader2_y = state[5] + self.follower_utm_offset[1] + self.boat_offset[1]
        leader2_ok = abs(leader2_x) <= 1e7 and abs(leader2_y) <= 1e7
        if leader2_ok:
            leader2_lat, leader2_lon = self.utm_to_gps(leader2_x, leader2_y)
            if leader2_lat is not None and leader2_lon is not None:
                self._publish_navsat(self.navsatfix_pub_leader2, leader2_lat, leader2_lon)
        else:
            self.get_logger().warn(
                f'Leader2 UTM values out of range: x={leader2_x}, y={leader2_y}. Skipping publish.')

        # Centroid of all three positions.
        if (follower_lat is not None and follower_lon is not None and leader1_ok and leader2_ok):
            centroid_x = (follower_x + leader1_x + leader2_x) / 3.0
            centroid_y = (follower_y + leader1_y + leader2_y) / 3.0
            centroid_lat, centroid_lon = self.utm_to_gps(centroid_x, centroid_y)
            if centroid_lat is not None and centroid_lon is not None:
                self._publish_navsat(self.navsatfix_pub_centroid, centroid_lat, centroid_lon)

        # Follower setpoint (apex of the semicircle behind the two leaders).
        if leader1_ok and leader2_ok:
            self._publish_setpoint(state, leader1_x, leader1_y, leader2_x, leader2_y,
                                   follower_x, follower_y)

    # ----------------------------------------------------------- process noise
    def make_WNOA(self, process_noise, dt):
        """Block-diagonal WNOA process noise (3 x 4x4 blocks for [x, y, vx, vy])."""
        dt = float(max(dt, 1e-3))  # guard against zero
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        block = (process_noise ** 2) * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ], dtype=np.float32)
        Q = np.zeros((12, 12), dtype=np.float32)
        for i in range(3):
            Q[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = block
        return Q


def main(args=None):
    rclpy.init(args=args)
    node = AcousticUKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
