#!/usr/bin/env python3
"""Unscented Kalman Filter (UKF) node for acoustic follower localization.

This node estimates the relative positions/velocities of however many leader
vehicles it hears, plus one follower, from:
  - leader GPS (NavSatFix / GeoPoint)  -> leader position pseudo-measurements
  - acoustic range (Float32)           -> leader<->follower distance measurements
  - follower heading (Float32)         -> follower velocity-direction measurement
  - IMU / follower GPS timestamps      -> drive the constant-velocity prediction

Leaders are not listed. The node watches ``leader_topic_root`` for the same
topics the OWTT follower opens on the first broadcast from a modem:

    <root>/m<modem_id>/smarc/latlon
    <root>/m<modem_id>/distance

State vector, metres relative to the follower's UTM origin. Each leader is
four elements ``[x, y, vx, vy]`` and the follower is the last four:

    [x1, y1, vx1, vy1, ..., xN, yN, vxN, vyN, xf, yf, vfx, vfy]

The filter starts once ``min_leaders`` modems have a fix. A modem heard after
that is appended.

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
  <root>/m<modem_id>/distance     std_msgs/Float32      acoustic slant range to that modem
  <root>/m<modem_id>/smarc/latlon sensor_msgs/NavSatFix that modem's GPS (or geographic_msgs/GeoPoint
                                                      if leader_gps_type='GeoPoint')
  /lolo/standard/navsatfix       sensor_msgs/NavSatFix follower GPS  (or GeoPoint if
                                                      follower_gps_type='GeoPoint');
                                                      also triggers a prediction step
  /lolo/smarc/latlon             geographic_msgs/GeoPoint  delta_pos truth when
                                                      delta_truth_topic is set. The
                                                      filter clock stays on the GPS topic.
  /lolo/standard/imu             sensor_msgs/Imu       timestamp source for prediction steps
  /lolo/smarc/depth              std_msgs/Float32      follower depth (slant->horizontal range)
  /lolo/smarc/heading            std_msgs/Float32      external heading; subscribed when
                                                      heading_topic is set

``<root>`` is ``leader_topic_root`` (default ``/lolo/owtt``). A subscription is
created the first time that topic shows up, and only if its type matches.

Publications
------------
  /follower/ukf/state              std_msgs/Float32MultiArray
                                     state vector, 4*(n_leaders+1) elements
  /follower/ukf/navsatfix          sensor_msgs/NavSatFix  follower lat/lon (baselink)
  /follower/ukf/navsatfix_<modem>  sensor_msgs/NavSatFix  that leader's lat/lon
  /follower/ukf/navsatfix_centroid sensor_msgs/NavSatFix  centroid of follower + leaders
  /follower/ukf/pose               geometry_msgs/PoseWithCovarianceStamped
                                     follower pose in absolute UTM metres (frame 'utm');
                                     position covariance is the follower block of P
  /follower/ukf/setpoint           geographic_msgs/GeoPoint  desired follower position:
                                     semicircle behind the two setpoint leaders
  /follower/ukf/delta_pos          std_msgs/Float32       distance between estimate and follower GPS truth

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


def parse_setpoint_modem_ids(value):
    """Modem ids for the setpoint diameter. Accepts '067,069', ['067','069'], or m067."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(';', ',').split(',')
    else:
        parts = list(value)
    ids = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if text[0] in ('m', 'M') and text[1:].isdigit():
            text = text[1:]
        if text.isdigit():
            text = f'{int(text):03d}'
        ids.append(text)
    return ids


def select_setpoint_pair(slots, state, pinned, min_speed):
    """Indices of the two leaders that form the setpoint diameter.

    ``pinned`` is the launch/yaml list, in leader-1 then leader-2 order.
    An empty list uses the two fastest leaders at or above ``min_speed``,
    ordered by modem id. If fewer than two are moving, the longest pair is
    used, also ordered by modem id. Missing pinned ids return None.
    """
    n = len(slots)
    if n < 2:
        return None
    if pinned:
        index = {modem: i for i, modem in enumerate(slots)}
        chosen = [index[modem] for modem in pinned if modem in index]
        if len(chosen) < 2:
            return None
        return chosen[0], chosen[1]

    speeds = []
    for i in range(n):
        speeds.append(math.hypot(float(state[i * 4 + 2]), float(state[i * 4 + 3])))
    moving = [i for i, speed in enumerate(speeds) if speed >= min_speed]
    if len(moving) >= 2:
        moving.sort(key=lambda i: speeds[i], reverse=True)
        pair = sorted(moving[:2], key=lambda i: slots[i])
        return pair[0], pair[1]

    best = None
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(
                float(state[i * 4]) - float(state[j * 4]),
                float(state[i * 4 + 1]) - float(state[j * 4 + 1]))
            if best is None or dist > best[0]:
                best = (dist, i, j)
    i, j = best[1], best[2]
    if slots[i] > slots[j]:
        i, j = j, i
    return i, j


def setpoint_from_pair(p1, p2, v1, v2, follower_xy, weight, min_speed):
    """Arc-midpoint behind the diameter p1--p2. Same circle as the two-leader setpoint."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    centre = 0.5 * (p1 + p2)
    diameter = p2 - p1
    length = float(np.linalg.norm(diameter))
    if length < 1e-6:
        return None
    radius = 0.5 * length
    d_hat = diameter / length
    n_hat = np.array([-d_hat[1], d_hat[0]])

    forward = np.zeros(2)
    for vel, w in ((v1, weight), (v2, 1.0 - weight)):
        vel = np.asarray(vel, dtype=np.float64)
        speed = float(np.linalg.norm(vel))
        if speed >= min_speed:
            forward += w * (vel / speed)
    if float(np.linalg.norm(forward)) > 1e-6:
        if float(np.dot(n_hat, -forward)) < 0.0:
            n_hat = -n_hat
    elif float(np.dot(n_hat, np.asarray(follower_xy, dtype=np.float64) - centre)) < 0.0:
        n_hat = -n_hat
    return centre + radius * n_hat


class UKF:
    """Unscented Kalman Filter for N leaders + 1 follower.

    Computations are done in float64 for numerical stability (the unscented
    transform relies on a Cholesky factorisation of the covariance, which is
    sensitive to round-off in float32).
    """

    def __init__(self, initial_state, Q, R, dt=0.0, max_velocity=500.0,
                 alpha=1.0, beta=2.0, kappa=0.0):
        initial_state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if initial_state.size < 8 or initial_state.size % 4 != 0:
            raise ValueError(
                f'UKF state length must be 4*(n_leaders+1) and at least 8, '
                f'got {initial_state.size}')
        self.n_leaders = initial_state.size // 4 - 1
        self.N = initial_state.size
        self.dt = float(dt)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.max_velocity = float(max_velocity)

        self.x = initial_state.reshape((self.N, 1))
        self.P = np.eye(self.N, dtype=np.float64) * 4.0

        # Van der Merwe scaled unscented-transform weights.
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self._set_weights()

    @property
    def follower_index(self):
        """Index of the follower's x component. Leaders occupy ``[0, f)``."""
        return 4 * self.n_leaders

    def _set_weights(self):
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
        for block in range(self.n_leaders + 1):
            for idx in (block * 4 + 2, block * 4 + 3):
                self.x[idx, 0] = np.clip(
                    self.x[idx, 0], -self.max_velocity, self.max_velocity)

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
        for block in range(self.n_leaders + 1):
            b = block * 4
            out[b] = s[b] + s[b + 2] * dt
            out[b + 1] = s[b + 1] + s[b + 3] * dt
        return out

    @staticmethod
    def _hx_leader(s, leader_id):
        """Measurement model: [leader_x, leader_y, range(leader, follower)]."""
        f = len(s) - 4  # follower is the last block
        lx = s[leader_id * 4]
        ly = s[leader_id * 4 + 1]
        dist = math.hypot(lx - s[f], ly - s[f + 1])
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
        f = self.follower_index
        lx = fx[leader_id * 4]
        ly = fx[leader_id * 4 + 1]
        xf = fx[f]
        yf = fx[f + 1]
        dist = math.hypot(lx - xf, ly - yf)
        if dist < 1e-6:
            dist = 1e-6

        # Measurement Jacobian H (3 x N) for h(x) = [lx, ly, ||leader-follower||].
        H = np.zeros((3, self.N), dtype=np.float64)
        H[0, leader_id * 4] = 1.0
        H[1, leader_id * 4 + 1] = 1.0
        H[2, leader_id * 4] = (lx - xf) / dist
        H[2, leader_id * 4 + 1] = (ly - yf) / dist
        H[2, f] = (xf - lx) / dist
        H[2, f + 1] = (yf - ly) / dist

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
        f = self.follower_index
        vfx = float(self.x[f + 2, 0])
        vfy = float(self.x[f + 3, 0])
        if math.hypot(vfx, vfy) < 1e-3:
            return self.x.flatten()  # direction is unobservable at near-zero speed

        z = math.radians(heading_deg)
        sigmas = self._sigma_points()
        Zsig = np.array([math.atan2(s[f + 2], s[f + 3]) for s in sigmas])

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

    def add_leader(self, x, y):
        """Append one leader in front of the follower block.

        Existing leader blocks stay put. The follower block shifts by four.
        The new leader starts at ``(x, y)`` with zero velocity and the same
        initial variance the filter uses at startup (4 m^2 on its diagonal).
        """
        old_f = self.follower_index
        old_n = self.N
        n2 = old_n + 4

        new_x = np.zeros((n2, 1), dtype=np.float64)
        new_x[:old_f] = self.x[:old_f]
        new_x[old_f, 0] = float(x)
        new_x[old_f + 1, 0] = float(y)
        new_x[old_f + 4:] = self.x[old_f:]

        new_P = np.zeros((n2, n2), dtype=np.float64)
        new_P[:old_f, :old_f] = self.P[:old_f, :old_f]
        new_P[:old_f, old_f + 4:] = self.P[:old_f, old_f:]
        new_P[old_f + 4:, :old_f] = self.P[old_f:, :old_f]
        new_P[old_f + 4:, old_f + 4:] = self.P[old_f:, old_f:]
        new_P[old_f:old_f + 4, old_f:old_f + 4] = np.eye(4, dtype=np.float64) * 4.0

        Q = np.asarray(self.Q, dtype=np.float64)
        new_Q = np.zeros((n2, n2), dtype=np.float64)
        if Q.shape == (old_n, old_n):
            new_Q[:old_f, :old_f] = Q[:old_f, :old_f]
            new_Q[:old_f, old_f + 4:] = Q[:old_f, old_f:]
            new_Q[old_f + 4:, :old_f] = Q[old_f:, :old_f]
            new_Q[old_f + 4:, old_f + 4:] = Q[old_f:, old_f:]
            if old_f >= 4:
                new_Q[old_f:old_f + 4, old_f:old_f + 4] = Q[:4, :4]
            else:
                new_Q[old_f:old_f + 4, old_f:old_f + 4] = np.eye(4)

        self.n_leaders += 1
        self.N = n2
        self.x = new_x
        self.P = new_P
        self.Q = new_Q
        self._set_weights()

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
        # use_sim_time is often pre-declared by launch; default live=false.
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

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

        # The setpoint circle always uses two leaders. setpoint_leader1_weight
        # trusts the first of that pair (the lower modem id, unless the launch
        # list names an order).
        self.setpoint_leader1_weight = float(config.get('setpoint_leader1_weight', 0.7))
        pinned = parse_setpoint_modem_ids(config.get('setpoint_modem_ids', []))
        self.declare_parameter('setpoint_modem_ids', '')
        override = self.get_parameter('setpoint_modem_ids').get_parameter_value().string_value
        if override.strip():
            pinned = parse_setpoint_modem_ids(override)
        self.setpoint_modem_ids = pinned
        self.min_leaders = max(1, int(config.get('min_leaders', 2)))

        # UKF unscented-transform tuning parameters.
        # alpha=1.0, kappa=0.0 -> lambda=0, which is numerically well-conditioned
        # for this state (small alpha makes the central weight blow up and
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
        # Per-leader NavSatFix topics append _<modem_id>. Centroid appends _centroid.
        self.navsatfix_topic = build_topic(
            config.get('navsatfix_topic', 'follower/ukf/navsatfix'))
        follower_gps_topic = build_topic(config.get('follower_gps_topic', 'smarc/latlon'))
        delta_pos_topic = build_topic(config.get('delta_pos_topic', 'ukf/delta_pos'))
        pose_cov_topic = build_topic(config.get('pose_cov_topic', 'follower/ukf/pose'))
        setpoint_topic = build_topic(config.get('setpoint_topic', 'follower/ukf/setpoint'))
        leader_gps_type = config.get('leader_gps_type', 'NavSatFix')
        self.leader_gps_type = leader_gps_type
        follower_gps_type = config.get('follower_gps_type', 'NavSatFix')
        root = config.get('leader_topic_root', '') or ''
        if not root:
            root = f'/{follower_ns}/owtt' if follower_ns else '/owtt'
        self.leader_topic_root = build_topic(root).rstrip('/')
        if self.leader_gps_type == 'GeoPoint':
            self._leader_gps_msg = GeoPoint
            self._leader_gps_type_name = 'geographic_msgs/msg/GeoPoint'
        else:
            self._leader_gps_msg = NavSatFix
            self._leader_gps_type_name = 'sensor_msgs/msg/NavSatFix'
        self._distance_type_name = 'std_msgs/msg/Float32'

        # Publishers
        self.state_pub = self.create_publisher(Float32MultiArray, publish_topic, 10)
        self.navsatfix_pub = self.create_publisher(NavSatFix, self.navsatfix_topic, 10)
        self.navsatfix_pub_centroid = self.create_publisher(
            NavSatFix, self.navsatfix_topic + '_centroid', 10)
        self.delta_pos_pub = self.create_publisher(Float32, delta_pos_topic, 10)
        self.pose_cov_pub = self.create_publisher(PoseWithCovarianceStamped, pose_cov_topic, 10)
        # Setpoint is a target coordinate -> geographic_msgs/GeoPoint (lat/lon/alt).
        self.setpoint_pub = self.create_publisher(GeoPoint, setpoint_topic, 10)

        # Thread safety
        self.lock = threading.Lock()

        # UTM reference zone (set from the first GPS message).
        self.utm_zone_number = None
        self.utm_zone_letter = None

        # One entry per modem once its topics appear. slot is None until the
        # filter has a state block for it.
        self.leaders = {}
        self.leader_slots = []  # modem ids in state order
        self._type_warned = set()
        self.follower_gps_samples = []
        self._follower_wait_logged = False
        self.initialization_phase = 'collecting'
        self.follower_utm_offset = None
        self.last_truth_follower_utm = None
        self._separate_delta_truth = False

        # UKF instance
        self.ukf = None

        if follower_gps_type == 'GeoPoint':
            self.create_subscription(
                GeoPoint, follower_gps_topic, self.follower_gps_geopoint_callback, 10)
        else:
            self.create_subscription(
                NavSatFix, follower_gps_topic, self.follower_gps_callback, 10)

        # delta_pos truth. Empty keeps the follower GPS topic. A separate topic
        # (LoLo: /lolo/smarc/latlon) is not used to clock the filter.
        delta_truth_topic = config.get('delta_truth_topic', '') or ''
        self._separate_delta_truth = bool(delta_truth_topic) and delta_truth_topic != follower_gps_topic
        if self._separate_delta_truth:
            delta_truth_topic = build_topic(delta_truth_topic)
            delta_truth_type = config.get('delta_truth_type', 'GeoPoint')
            if delta_truth_type == 'GeoPoint':
                self.create_subscription(
                    GeoPoint, delta_truth_topic, self._delta_truth_geopoint, 10)
            else:
                self.create_subscription(
                    NavSatFix, delta_truth_topic, self._delta_truth_navsat, 10)
            self.get_logger().info(
                f'delta_pos compares against {delta_truth_topic} ({delta_truth_type})')
        else:
            self.get_logger().info(
                f'delta_pos compares against {follower_gps_topic}')

        # Heading: optional external compass. Empty heading_topic skips the sub;
        # sticks typically leave this empty and rely on velocity-derived heading.
        self.follower_heading_deg = None
        heading_topic = config.get('heading_topic', '/lolo/smarc/heading')
        if heading_topic:
            heading_topic = build_topic(heading_topic)
            self.create_subscription(Float32, heading_topic, self.heading_callback, 10)
            self.get_logger().info(f'Subscribed to heading on {heading_topic}')

        imu_topic = config.get('imu_topic', 'core/imu')
        if imu_topic:
            imu_topic = build_topic(imu_topic)
            self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
            self.get_logger().info(f'Subscribed to IMU on {imu_topic}')
        else:
            self.get_logger().info('imu_topic empty; IMU predict disabled (GPS-driven predict only)')

        # R: leader x/y use measurement_noise^2, distance uses its own noise.
        # Q is built when the filter starts, once the leader count is known.
        self.R = np.eye(3, dtype=np.float64)
        self.R[0, 0] = measurement_noise ** 2
        self.R[1, 1] = measurement_noise ** 2
        self.R[2, 2] = distance_measurement_noise ** 2

        self.update_cooldown = config.get('update_cooldown', 0.0)
        self.last_update_time = self.get_clock().now().nanoseconds / 1e9 - self.update_cooldown

        self.leader_depth = config.get('leader_depth', 1.0)
        depth_topic_raw = config.get('depth_topic', 'smarc/depth')
        self.depth = None
        if depth_topic_raw:
            depth_topic = build_topic(depth_topic_raw)
            self.create_subscription(Float32, depth_topic, self.depth_callback, 10)

        # TF2 transform listener (kept for parity / future use).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.imu_frame = 'lolo/base_link'
        self.utm_frame = 'utm'

        self.last_predict_time = None  # for variable-dt prediction

        # use_sim_time is declared above; do not force True (breaks live sticks).
        use_sim = bool(self.get_parameter('use_sim_time').value)
        self.create_timer(0.5, self._scan_leaders)
        self.get_logger().info(
            f'AcousticUKFNode initialized (use_sim_time={use_sim}). '
            f'Leaders open under {self.leader_topic_root}/m<modem_id>/ '
            f'(min {self.min_leaders}). '
            f'Setpoint pair: {self.setpoint_modem_ids or "two fastest"}.')

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

    # ----------------------------------------------------------- discovery
    def _scan_leaders(self):
        """Subscribe to owtt/m<modem_id>/ topics as the follower creates them."""
        root = self.leader_topic_root + '/'
        seen = {}
        for name, types in self.get_topic_names_and_types():
            if not name.startswith(root):
                continue
            rest = name[len(root):]
            parts = rest.split('/')
            if len(parts) == 2 and parts[1] == 'distance':
                kind = 'distance'
            elif len(parts) == 3 and parts[1] == 'smarc' and parts[2] == 'latlon':
                kind = 'latlon'
            else:
                continue
            token = parts[0]
            # Follower publishes m067, not 067: a topic token cannot start with a digit.
            if len(token) != 4 or token[0] != 'm' or not token[1:].isdigit():
                continue
            modem = token[1:]
            seen.setdefault(modem, {})[kind] = (name, types)

        for modem, kinds in seen.items():
            leader = self.leaders.get(modem)
            if leader is None:
                leader = {
                    'samples': [],
                    'pos': None,
                    'dist': None,
                    'slot': None,
                    'navsat_pub': self.create_publisher(
                        NavSatFix, f'{self.navsatfix_topic}_{modem}', 10),
                }
                self.leaders[modem] = leader
                self.get_logger().info(
                    f'Leader {modem} -> {root}m{modem}/smarc/latlon, {root}m{modem}/distance')
            if 'distance' in kinds and 'dist_sub' not in leader:
                topic, types = kinds['distance']
                if self._distance_type_name in types:
                    leader['dist_sub'] = self.create_subscription(
                        Float32, topic,
                        lambda msg, m=modem: self._handle_distance(m, msg.data), 10)
                else:
                    self._warn_type(topic, types, self._distance_type_name)
            if 'latlon' in kinds and 'gps_sub' not in leader:
                topic, types = kinds['latlon']
                if self._leader_gps_type_name in types:
                    leader['gps_sub'] = self.create_subscription(
                        self._leader_gps_msg, topic,
                        lambda msg, m=modem: self._handle_leader_gps(
                            m, msg.latitude, msg.longitude), 10)
                else:
                    self._warn_type(topic, types, self._leader_gps_type_name)

    def _warn_type(self, topic, types, expected):
        if topic in self._type_warned:
            return
        self._type_warned.add(topic)
        self.get_logger().error(
            f'{topic} is {types}, expected {expected}; not subscribing')

    def _relative_xy(self, lat, lon):
        """Leader position in the follower-origin frame, or None."""
        if self.follower_utm_offset is None:
            return None
        x, y = self.gps_to_utm(lat, lon)
        if x is None or y is None:
            return None
        return (
            x - self.follower_utm_offset[0] - self.boat_offset[0],
            y - self.follower_utm_offset[1] - self.boat_offset[1],
        )

    # ----------------------------------------------------------- leader GPS
    def _handle_leader_gps(self, modem_id, lat, lon):
        """Shared logic for leader GPS (NavSatFix or GeoPoint)."""
        leader = self.leaders.get(modem_id)
        if leader is None:
            return
        leader['samples'].append((lat, lon))
        if self.initialization_phase != 'ready':
            self.get_logger().info(
                f'Leader {modem_id} GPS sample '
                f'{len(leader["samples"])}/{self.leader_init_samples}: '
                f'lat={lat:.6f}, lon={lon:.6f}')
            self._try_initialize()
            return
        rel = self._relative_xy(lat, lon)
        if rel is None:
            self.get_logger().warn(
                f'UTM conversion failed for leader {modem_id}')
            return
        leader['pos'] = rel
        if leader['slot'] is None:
            self._adopt_leader(modem_id)

    # ----------------------------------------------------------- follower GPS
    def _remember_truth(self, lat, lon):
        gx, gy = self.gps_to_utm(lat, lon)
        if gx is not None and gy is not None:
            self.last_truth_follower_utm = (gx, gy)

    def _delta_truth_navsat(self, msg):
        self._remember_truth(msg.latitude, msg.longitude)

    def _delta_truth_geopoint(self, msg):
        self._remember_truth(msg.latitude, msg.longitude)

    def follower_gps_callback(self, msg):
        """Follower GPS (NavSatFix): init samples + variable-dt prediction trigger."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._handle_follower_gps(msg.latitude, msg.longitude, t)

    def follower_gps_geopoint_callback(self, msg):
        """Follower GPS (GeoPoint): uses ROS clock for dt (no header stamp)."""
        t = self.get_clock().now().nanoseconds / 1e9
        self._handle_follower_gps(msg.latitude, msg.longitude, t)

    def _handle_follower_gps(self, lat, lon, t):
        if self.initialization_phase != 'ready':
            if len(self.follower_gps_samples) < self.follower_init_samples:
                self.follower_gps_samples.append((lat, lon))
                self.get_logger().info(
                    f'Follower GPS sample {len(self.follower_gps_samples)}/'
                    f'{self.follower_init_samples}: lat={lat:.6f}, lon={lon:.6f}')
            else:
                self.follower_gps_samples[-1] = (lat, lon)
                if not self._follower_wait_logged:
                    self.get_logger().info(
                        'Follower GPS collected; waiting for leader fixes '
                        'before the filter starts')
                    self._follower_wait_logged = True
            if not self._separate_delta_truth:
                self._remember_truth(lat, lon)
            if self._try_initialize():
                self.last_predict_time = t
        elif self.ukf is not None:
            if self.last_predict_time is not None:
                dt = t - self.last_predict_time
                if 0.0 < dt < 2.0:
                    with self.lock:
                        if not self._separate_delta_truth:
                            self._remember_truth(lat, lon)
                        self.ukf.dt = float(dt)
                        self.ukf.Q = self.make_WNOA(self.process_noise, dt)
                        self.ukf.predict()
                        self._publish_outputs()
                elif dt >= 2.0:
                    self.get_logger().warn(f'Large dt gap: {dt:.3f}s, skipping prediction')
            self.last_predict_time = t
        else:
            if not self._separate_delta_truth:
                self._remember_truth(lat, lon)

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

    def _handle_distance(self, modem_id, raw_distance):
        leader = self.leaders.get(modem_id)
        if leader is None:
            return
        d = raw_distance * self.correction_factor
        leader['dist'] = self._horizontal_distance(d) + self.range_bias
        if self.initialization_phase != 'ready' or leader['slot'] is None:
            return
        if leader['pos'] is None:
            return
        with self.lock:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_update_time < self.update_cooldown:
                return
            z = np.array([
                leader['pos'][0],
                leader['pos'][1],
                leader['dist'],
            ], dtype=np.float64)
            self.get_logger().info(
                f'Updating UKF with leader {modem_id} measurement: {z}')
            self.ukf.update(z, leader['slot'])
            self.last_update_time = now
            self._publish_outputs()

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
    def _ready_modems(self):
        return sorted(
            modem for modem, leader in self.leaders.items()
            if len(leader['samples']) >= self.leader_init_samples
            and leader['slot'] is None)

    def _try_initialize(self):
        if self.initialization_phase == 'ready' or self.ukf is not None:
            return False
        ready = self._ready_modems()
        if len(ready) < self.min_leaders:
            return False
        if len(self.follower_gps_samples) < self.follower_init_samples:
            return False
        return self._initialize_ukf(ready)

    def _mean_latlon(self, samples):
        window = samples[-self.leader_init_samples:]
        lat = sum(s[0] for s in window) / len(window)
        lon = sum(s[1] for s in window) / len(window)
        return lat, lon

    def _follower_modem_origin(self):
        """Follower modem position relative to the GPS/baselink origin."""
        if self.follower_heading_deg is None:
            self.get_logger().warn(
                'No follower heading at init; modem placed at baselink origin')
            return 0.0, 0.0
        h = math.radians(self.follower_heading_deg)
        ox = self.modem_to_baselink_offset_x
        oy = self.modem_to_baselink_offset_y
        # NED heading (0=North, 90=East): body +x -> [E, N] = [sin h, cos h].
        xf0 = ox * math.sin(h) + oy * math.cos(h)
        yf0 = ox * math.cos(h) - oy * math.sin(h)
        self.get_logger().info(
            f'Init follower modem offset from heading {self.follower_heading_deg:.1f} deg: '
            f'({xf0:.2f}, {yf0:.2f}) m')
        return xf0, yf0

    def _abort_init(self, modem_ids):
        for modem in modem_ids:
            self.leaders[modem]['slot'] = None
        self.leader_slots = [m for m in self.leader_slots if m not in modem_ids]
        self.follower_utm_offset = None

    def _initialize_ukf(self, modem_ids):
        """Initialize from every modem that already has a fix, sorted by id."""
        self.get_logger().info(f'Initializing UKF with leaders {modem_ids}...')
        samples = self.follower_gps_samples[-self.follower_init_samples:]
        avg_flat = sum(s[0] for s in samples) / len(samples)
        avg_flon = sum(s[1] for s in samples) / len(samples)
        follower_x, follower_y = self.gps_to_utm(avg_flat, avg_flon)
        if follower_x is None or follower_y is None:
            self.get_logger().error('Failed to convert follower GPS to UTM')
            return False
        self.follower_utm_offset = np.array([follower_x, follower_y], dtype=np.float64)

        state = []
        for modem in modem_ids:
            leader = self.leaders[modem]
            lat, lon = self._mean_latlon(leader['samples'])
            rel = self._relative_xy(lat, lon)
            if rel is None:
                self.get_logger().error(f'Failed to convert leader {modem} GPS to UTM')
                self._abort_init(modem_ids)
                return False
            leader['pos'] = rel
            leader['slot'] = len(state) // 4
            state.extend([rel[0], rel[1], 0.0, 0.0])
            self.leader_slots.append(modem)

        xf0, yf0 = self._follower_modem_origin()
        state.extend([xf0, yf0, 0.0, 0.0])
        self.ukf = UKF(
            np.array(state, dtype=np.float64),
            Q=self.make_WNOA(
                self.process_noise, max(self.dt, 0.1), n_blocks=len(modem_ids) + 1),
            R=self.R, dt=self.dt, max_velocity=self.max_speed,
            alpha=self.ukf_alpha, beta=self.ukf_beta, kappa=self.ukf_kappa)
        self.initialization_phase = 'ready'
        self.get_logger().info(
            f'UKF ready ({self.ukf.n_leaders} leaders), '
            f'UTM offset {self.follower_utm_offset}')
        return True

    def _adopt_leader(self, modem_id):
        """Give a modem heard after startup its own state block."""
        leader = self.leaders[modem_id]
        if leader['slot'] is not None or leader['pos'] is None or self.ukf is None:
            return
        with self.lock:
            if leader['slot'] is not None:
                return
            slot = self.ukf.n_leaders
            self.ukf.add_leader(leader['pos'][0], leader['pos'][1])
            self.ukf.Q = self.make_WNOA(self.process_noise, max(self.ukf.dt, 0.1))
            leader['slot'] = slot
            self.leader_slots.append(modem_id)
            self.get_logger().info(
                f'Leader {modem_id} joined the filter as slot {slot} '
                f'({self.ukf.n_leaders} leaders)')

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
        position block; z and the orientation axes are marked unknown (large
        variance) since they are not estimated. Orientation encodes the
        estimated heading as yaw (ENU: yaw measured CCW from east).
        """
        if self.ukf is None:
            return
        P = self.ukf.get_covariance()
        f = self.ukf.follower_index

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
        cov[0] = float(P[f, f])          # xx
        cov[1] = float(P[f, f + 1])      # xy
        cov[6] = float(P[f + 1, f])      # yx
        cov[7] = float(P[f + 1, f + 1])  # yy
        cov[14] = UNKNOWN          # zz
        cov[21] = UNKNOWN          # rot_x
        cov[28] = UNKNOWN          # rot_y
        cov[35] = UNKNOWN          # rot_z (yaw not formally estimated here)
        msg.pose.covariance = cov

        self.pose_cov_pub.publish(msg)

    def _publish_setpoint(self, state, leader_xy, fx, fy):
        """Publish the arc-midpoint behind two leaders.

        The diameter is ``setpoint_modem_ids`` when that list is set, otherwise
        the two fastest leaders. Every other modem still updates the filter and
        is left out of the circle. With two leaders and no list, the only pair
        is that diameter, which is the previous setpoint.
        """
        pair = select_setpoint_pair(
            self.leader_slots, state, self.setpoint_modem_ids, self.heading_min_speed)
        if pair is None:
            if self.setpoint_modem_ids:
                self.get_logger().warn(
                    f'Setpoint waiting for {self.setpoint_modem_ids}; '
                    f'in the filter: {self.leader_slots}',
                    throttle_duration_sec=10.0)
            return
        i, j = pair
        setpoint = setpoint_from_pair(
            leader_xy[i], leader_xy[j],
            (state[i * 4 + 2], state[i * 4 + 3]),
            (state[j * 4 + 2], state[j * 4 + 3]),
            (fx, fy), self.setpoint_leader1_weight, self.heading_min_speed)
        if setpoint is None:
            return
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
        f = self.ukf.follower_index
        vfx = float(state[f + 2])
        vfy = float(state[f + 3])
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
        f = self.ukf.follower_index
        follower_modem_x = state[f] + self.follower_utm_offset[0]
        follower_modem_y = state[f + 1] + self.follower_utm_offset[1]

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

        # Delta to the configured truth fix (smarc/latlon on LoLo).
        if self.last_truth_follower_utm is not None:
            truth_x, truth_y = self.last_truth_follower_utm
            delta = math.sqrt((follower_x - truth_x) ** 2 + (follower_y - truth_y) ** 2)
            delta_msg = Float32()
            delta_msg.data = float(delta)
            self.delta_pos_pub.publish(delta_msg)

        leader_xy = []
        for slot, modem in enumerate(self.leader_slots):
            lx = state[slot * 4] + self.follower_utm_offset[0] + self.boat_offset[0]
            ly = state[slot * 4 + 1] + self.follower_utm_offset[1] + self.boat_offset[1]
            if abs(lx) > 1e7 or abs(ly) > 1e7:
                self.get_logger().warn(
                    f'Leader {modem} UTM out of range: x={lx}, y={ly}. Skipping.')
                continue
            lat, lon = self.utm_to_gps(lx, ly)
            pub = self.leaders[modem]['navsat_pub']
            if lat is not None and lon is not None:
                self._publish_navsat(pub, lat, lon)
            leader_xy.append((lx, ly))

        if follower_lat is not None and follower_lon is not None and leader_xy:
            centroid_x = (follower_x + sum(p[0] for p in leader_xy)) / (len(leader_xy) + 1)
            centroid_y = (follower_y + sum(p[1] for p in leader_xy)) / (len(leader_xy) + 1)
            centroid_lat, centroid_lon = self.utm_to_gps(centroid_x, centroid_y)
            if centroid_lat is not None and centroid_lon is not None:
                self._publish_navsat(self.navsatfix_pub_centroid, centroid_lat, centroid_lon)

        if len(leader_xy) == len(self.leader_slots):
            self._publish_setpoint(state, leader_xy, follower_x, follower_y)

    # ----------------------------------------------------------- process noise
    def make_WNOA(self, process_noise, dt, n_blocks=None):
        """Block-diagonal WNOA process noise, one 4x4 block per vehicle."""
        if n_blocks is None:
            n_blocks = self.ukf.n_leaders + 1 if self.ukf is not None else 1
        dt = float(max(dt, 1e-3))  # guard against zero
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        block = (process_noise ** 2) * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ], dtype=np.float64)
        n = 4 * n_blocks
        Q = np.zeros((n, n), dtype=np.float64)
        for i in range(n_blocks):
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
