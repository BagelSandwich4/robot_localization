#!/usr/bin/env python3
"""ROS 2 particle filter node for localizing a robot against a known map."""

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import random
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler
from helper_functions import draw_random_sample

# Odometry motion model noise
ODOM_NOISE_SCALE = 0.1
ODOM_NOISE_FLOOR_XY = 0.01      # meters
ODOM_NOISE_FLOOR_THETA = 0.02   # radians

# Laser sensor model
LASER_SIGMA = 0.4               # std dev of sensor/map error (meters)
LASER_NUM_BEAMS = 40            # number of beams evaluated per particle
LASER_OUT_OF_MAP_DIST = 2.0     # distance assigned to hits outside the map (meters)
LASER_TEMPER = 0.5              # values below 1 flatten the weights

# Resampling
INJECT_FRACTION = 0.2           # fraction of particles replaced by fresh samples
LOCAL_INJECT_SPREAD = 0.3       # std dev of injected particles around the estimate (meters)

# Initialization around a given pose
INIT_POSE_SPREAD = 0.5          # std dev for x, y (meters) and theta (radians)


class Particle:
    """A hypothesis of the robot's pose in the map frame.

    Attributes:
        x: The x-coordinate of the hypothesis relative to the map frame.
        y: The y-coordinate of the hypothesis relative to the map frame.
        theta: The yaw of the hypothesis relative to the map frame.
        w: The particle weight. The class does not ensure that weights are
            normalized.
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """Construct a new Particle.

        Args:
            x: The x-coordinate in the map frame.
            y: The y-coordinate in the map frame.
            theta: The yaw in the map frame (radians).
            w: The particle weight.
        """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """Return the particle as a geometry_msgs/Pose message."""
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))


class ParticleFilter(Node):
    """ROS node that estimates the robot's pose with a particle filter.

    Attributes:
        base_frame: The robot base coordinate frame.
        map_frame: The map coordinate frame.
        odom_frame: The odometry coordinate frame.
        scan_topic: The laser scan topic to subscribe to.
        n_particles: The number of particles in the filter.
        d_thresh: Linear movement (meters) required before
            a filter update.
        a_thresh: Angular movement (radians) required before
            a filter update.
        n_eff_threshold: Resample when the normalized effective sample size
        resample_noise_xy: Std dev of x/y jitter added to resampled particles.
        resample_noise_theta: Std dev of yaw jitter added to resampled
            particles.
        odom_noise_scale: Odometry noise std dev as a fraction of the motion.
        odom_noise_floor_xy: Minimum x/y odometry noise std dev (meters).
        odom_noise_floor_theta: Minimum yaw odometry noise std dev (radians).
        particle_pub: Publisher for the particle cloud.
        last_scan_timestamp: Stamp of the latest scan, used to timestamp the
            map -> odom transform (important when playing bags).
        scan_to_process: The scan that run_loop should process next.
        particle_cloud: List of particles representing the pose distribution.
        current_odom_xy_theta: Odometry pose [x, y, theta] at the last filter
            update.
        odom_pose: Latest odometry pose (geometry_msgs/Pose), or None.
        robot_pose: Latest estimated pose in the map frame
            (geometry_msgs/Pose), or None.
        occupancy_field: Map query helper for distance to closest obstacle.
        transform_helper: Helper for tf2 transform operations.
        transform_update_timer: Timer that publishes the map -> odom
            transform.
    """

    def __init__(self):
        """Create publishers, subscribers, and start the filter loop."""
        super().__init__('pf')
        self.base_frame = 'base_footprint'
        self.map_frame = 'map'
        self.odom_frame = 'odom'
        self.scan_topic = 'scan'

        self.n_particles = 1000

        self.d_thresh = 0.2
        self.a_thresh = math.pi / 6

        self.n_eff_threshold = 0.4
        self.resample_noise_xy = 0.2
        self.resample_noise_theta = 0.02
        self.odom_noise_scale = ODOM_NOISE_SCALE
        self.odom_noise_floor_xy = ODOM_NOISE_FLOOR_XY
        self.odom_noise_floor_theta = ODOM_NOISE_FLOOR_THETA

        # Re-initialize the filter on a new approximate pose (e.g. from rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose',
                                 self.update_initial_pose, 10)

        self.particle_pub = self.create_publisher(ParticleCloud, 'particle_cloud',
                                                  qos_profile_sensor_data)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        self.last_scan_timestamp = None
        self.scan_to_process = None
        self.particle_cloud = []
        self.current_odom_xy_theta = []
        self.odom_pose = None
        self.robot_pose = None

        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # Run the filter in its own thread to avoid blocking the
        # single-threaded executor
        Thread(target=self.loop_wrapper).start()
        self.transform_update_timer = self.create_timer(0.05,
                                                        self.pub_latest_transform)

    def pub_latest_transform(self):
        """Publish the most recent map -> odom transform."""
        if self.last_scan_timestamp is None:
            return
        postdated_time = (Time.from_msg(self.last_scan_timestamp)
                          + Duration(seconds=0.1))
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame,
                                                              self.odom_frame,
                                                              postdated_time)

    def loop_wrapper(self):
        """Call run_loop repeatedly; runs in a separate thread."""
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """Process the pending scan, if any, through one filter iteration."""
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # Unable to get the robot pose matching the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # The scan is older than our oldest transform; drop it
                self.scan_to_process = None
            return

        (ranges, beam_angles) = self.transform_helper.convert_scan_to_polar_in_robot_frame(
            msg, self.base_frame)
        self.get_logger().debug(f'ranges[0]={ranges[0]}, beam_angles[0]={beam_angles[0]}')
        # Clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.get_logger().debug('x: {0}, y: {1}, yaw: {2}'.format(*new_odom_xy_theta))
        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:
            self.initialize_particle_cloud()
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            self.update_particles_with_odom()
            self.update_particles_with_laser(ranges, beam_angles)
            self.update_robot_pose()
            n_eff = self.compute_n_eff()
            self.get_logger().info(f'Useful particle density: {n_eff:.5f}')
            if n_eff < self.n_eff_threshold:
                self.resample_particles()
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        """Return True if the robot moved enough since the last update.

        Args:
            new_odom_xy_theta: The current odometry pose as (x, y, theta).

        Returns:
            True if the x, y, or yaw change exceeds its update threshold.
        """
        dx = new_odom_xy_theta[0] - self.current_odom_xy_theta[0]
        dy = new_odom_xy_theta[1] - self.current_odom_xy_theta[1]
        dtheta = self.transform_helper.angle_diff(new_odom_xy_theta[2],
                                                  self.current_odom_xy_theta[2])
        return (math.fabs(dx) > self.d_thresh
                or math.fabs(dy) > self.d_thresh
                or math.fabs(dtheta) > self.a_thresh)

    def update_robot_pose(self):
        """Update the robot pose estimate from the particles.

        The estimate is the weighted mean of the particles. Heading uses a
        weighted circular mean (via sin/cos), since angles cannot be averaged
        directly because of wraparound. Also refreshes the map -> odom
        transform if odometry is available.
        """
        self.normalize_particles()

        x = sum(p.w * p.x for p in self.particle_cloud)
        y = sum(p.w * p.y for p in self.particle_cloud)
        sin_sum = sum(p.w * math.sin(p.theta) for p in self.particle_cloud)
        cos_sum = sum(p.w * math.cos(p.theta) for p in self.particle_cloud)
        theta = math.atan2(sin_sum, cos_sum)

        self.robot_pose = Particle(x=x, y=y, theta=theta).as_pose()
        if self.odom_pose is not None:
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self):
        """Move each particle by the odometry change since the last update.

        The odometry delta is converted into the robot frame as a forward and
        lateral displacement, then applied relative to each particle's own
        heading, with Gaussian noise proportional to the motion:

            d_forward = dx * cos(theta_old) + dy * sin(theta_old)
            d_lateral = -dx * sin(theta_old) + dy * cos(theta_old)
            x += d_forward * cos(theta) - d_lateral * sin(theta)
            y += d_forward * sin(theta) + d_lateral * cos(theta)
            theta += dtheta
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        old_odom_xy_theta = self.current_odom_xy_theta
        dx = new_odom_xy_theta[0] - old_odom_xy_theta[0]
        dy = new_odom_xy_theta[1] - old_odom_xy_theta[1]
        dtheta = self.transform_helper.angle_diff(new_odom_xy_theta[2], old_odom_xy_theta[2])
        self.current_odom_xy_theta = new_odom_xy_theta

        old_theta = old_odom_xy_theta[2]
        base_forward = dx * math.cos(old_theta) + dy * math.sin(old_theta)
        base_lateral = -dx * math.sin(old_theta) + dy * math.cos(old_theta)

        for particle in self.particle_cloud:
            d_forward = base_forward + random.gauss(
                0, self.odom_noise_scale * abs(base_forward) + self.odom_noise_floor_xy)
            d_lateral = base_lateral + random.gauss(
                0, self.odom_noise_scale * abs(base_lateral) + self.odom_noise_floor_xy)
            cos_theta = math.cos(particle.theta)
            sin_theta = math.sin(particle.theta)
            particle.x += d_forward * cos_theta - d_lateral * sin_theta
            particle.y += d_forward * sin_theta + d_lateral * cos_theta
            particle.theta += dtheta + random.gauss(
                0, self.odom_noise_scale * abs(dtheta) + self.odom_noise_floor_theta)

    def resample_particles(self):
        """Resample the particle cloud according to the particle weights.

        Of the new cloud, (1 - INJECT_FRACTION) is drawn from the current
        particles with jitter. The rest is split evenly between particles
        near the current pose estimate and particles spread uniformly over
        the map, which helps recover from kidnapping and bad starts.
        """
        self.normalize_particles()
        n_inject = int(INJECT_FRACTION * self.n_particles)
        n_global = n_inject // 2
        n_local = n_inject - n_global

        weights = [p.w for p in self.particle_cloud]
        new_cloud = []
        for p in draw_random_sample(self.particle_cloud, weights, self.n_particles - n_inject):
            new_cloud.append(Particle(x=p.x + random.gauss(0, self.resample_noise_xy),
                                      y=p.y + random.gauss(0, self.resample_noise_xy),
                                      theta=p.theta + random.gauss(0, self.resample_noise_theta),
                                      w=1.0))

        est_x, est_y = self.robot_pose.position.x, self.robot_pose.position.y
        for _ in range(n_local):
            new_cloud.append(Particle(x=random.gauss(est_x, LOCAL_INJECT_SPREAD),
                                      y=random.gauss(est_y, LOCAL_INJECT_SPREAD),
                                      theta=random.uniform(-math.pi, math.pi),
                                      w=1.0))

        (x_min, x_max), (y_min, y_max) = self.occupancy_field.get_obstacle_bounding_box()
        for _ in range(n_global):
            new_cloud.append(Particle(x=random.uniform(x_min, x_max),
                                      y=random.uniform(y_min, y_max),
                                      theta=random.uniform(-math.pi, math.pi),
                                      w=1.0))

        self.particle_cloud = new_cloud
        self.normalize_particles()

    def compute_n_eff(self):
        """Return the normalized effective sample size of the particle cloud.

        Returns:
            N_eff / N, where N_eff = 1 / sum(w_i^2). The value is in
            [1/N, 1]; 1 means all weights are equal. Returns 0.0 if all
            weights are zero.
        """
        weights = np.array([p.w for p in self.particle_cloud])
        total = weights.sum()
        if total <= 0:
            return 0.0
        weights = weights / total
        n_eff = 1.0 / np.sum(weights ** 2)
        return n_eff / len(self.particle_cloud)

    def update_particles_with_laser(self, ranges, beam_angles):
        """Update the particle weights from a laser scan.

        For each particle, a subset of beams is projected into the map and the
        distance from each hit to the nearest obstacle is treated as error.
        Weights are computed in log space for numerical stability.

        Args:
            ranges: The distance readings to obstacles (meters).
            beam_angles: The angle of each reading in the robot frame
                (radians).
        """
        log_weights = []
        step = max(1, len(ranges) // LASER_NUM_BEAMS)
        for p in self.particle_cloud:
            sq_error = 0.0
            for i in range(0, len(ranges), step):
                if not math.isfinite(ranges[i]) or ranges[i] <= 0.0:
                    continue
                # Where the beam would hit if the robot were at this particle
                x_hit = p.x + ranges[i] * math.cos(p.theta + beam_angles[i])
                y_hit = p.y + ranges[i] * math.sin(p.theta + beam_angles[i])
                obstacle_dist = self.occupancy_field.get_closest_obstacle_distance(x_hit, y_hit)
                if math.isnan(obstacle_dist):
                    obstacle_dist = LASER_OUT_OF_MAP_DIST
                sq_error += obstacle_dist ** 2
            log_weights.append(math.log(max(p.w, 1e-10))
                               - LASER_TEMPER * sq_error / (2 * LASER_SIGMA ** 2))

        # Subtract the max log weight before exponentiating to avoid underflow
        max_log_weight = max(log_weights)
        for p, log_weight in zip(self.particle_cloud, log_weights):
            p.w = math.exp(log_weight - max_log_weight)
        self.normalize_particles()

    def update_initial_pose(self, msg):
        """Re-initialize the particle filter around a received pose estimate."""
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(xy_theta)

    def initialize_particle_cloud(self, xy_theta=None):
        """Initialize the particle cloud."""
        self.particle_cloud = []
        if xy_theta is None:
            (x_min, x_max), (y_min, y_max) = self.occupancy_field.get_obstacle_bounding_box()
            for _ in range(self.n_particles):
                self.particle_cloud.append(Particle(x=random.uniform(x_min, x_max),
                                                    y=random.uniform(y_min, y_max),
                                                    theta=random.uniform(-math.pi, math.pi),
                                                    w=1.0))
        else:
            for _ in range(self.n_particles):
                random_particle = Particle(x=random.gauss(xy_theta[0], INIT_POSE_SPREAD),
                                           y=random.gauss(xy_theta[1], INIT_POSE_SPREAD),
                                           theta=random.gauss(xy_theta[2], INIT_POSE_SPREAD),
                                           w=1.0)
                self.particle_cloud.append(random_particle)

        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """Normalize the particle weights so they sum to 1.0."""
        sum_w = sum(p.w for p in self.particle_cloud)
        if sum_w <= 0:
            for p in self.particle_cloud:
                p.w = 1.0 / len(self.particle_cloud)
            return

        for p in self.particle_cloud:
            p.w = p.w / sum_w

    def publish_particles(self, timestamp):
        """Publish the particle cloud for visualization (e.g. in rviz)."""
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        """Store an incoming scan if the previous one has been processed."""
        self.last_scan_timestamp = msg.header.stamp
        # Scans are dropped until run_loop finishes the previous one
        if self.scan_to_process is None:
            self.scan_to_process = msg


def main(args=None):
    """Run the particle filter node."""
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
