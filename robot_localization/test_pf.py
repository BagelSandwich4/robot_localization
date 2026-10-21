import math
from unittest.mock import MagicMock
from pf import ParticleFilter, Particle 


def make_pf(particles):
    pf = ParticleFilter.__new__(ParticleFilter)   # creates an instance without calling __init__
    pf.particle_cloud = particles
    pf.n_particles = len(particles)
    pf.keep_fraction = 0.1
    pf.resample_noise_xy = 0.02
    pf.resample_noise_theta = 0.02
    return pf


def test_normalize_sums_to_one():
    #tests that normalized weights add up to 1.0 with a small tolerance
    pf = make_pf([Particle(w=weight) for weight in (1.0, 2.0, 5.0)])
    pf.normalize_particles()
    assert abs(sum(p.w for p in pf.particle_cloud) - 1.0) < 0.05


def test_normalize_all_zero_weights():
    #tests that if all weights are zero, they will be set to equal values to prevent division by zero
    pf = make_pf([Particle(w=0.0) for _ in range(4)])
    pf.normalize_particles()
    assert all(abs(p.w - 0.25) < 0.05 for p in pf.particle_cloud)

def test_resample():
    good = Particle(x=5, y=5, w=0.99)
    pf = make_pf([good] + [Particle(x=0, y=0, w=0.0001) for _ in range(9)])
    pf.resample_particles()
    assert len(pf.particle_cloud) == 10
    assert good not in pf.particle_cloud                       # checks that the very good particle from the beggining was not just copied
    assert len({id(p) for p in pf.particle_cloud}) == 10       # checks that all paticles are unique
    near_good = sum(1 for p in pf.particle_cloud if abs(p.x - 5) < 0.5) # counts how many particles are close to the good particle 
    assert near_good >= 8                                      # makes sure that the resampling uses the good particle

def make_odom_pf(particles, old_odom, new_odom):
    pf = make_pf(particles) #makes a fake parricle filter with the function created before
    pf.transform_helper = MagicMock() # creates a mock obj so we do not need to use ROS to test odom updates
    pf.transform_helper.convert_pose_to_xy_and_theta.return_value = list(new_odom) # sets the return value of the mock to the new odom pose
    pf.transform_helper.angle_diff.side_effect = lambda a, b: a - b # subtracts the two angles to get the difference, so we do not need to use the ROS function in this test
    pf.odom_pose = object() # creates a dummy object to represent the odom pose, since we do not need to use it in this test
    pf.current_odom_xy_theta = list(old_odom) # sets the current odom pose to the old odom pose
    return pf


def test_forward_motion_follows_heading():
    # checks that the odometry update moves particles in the direction they are facing, not the direction the robot is facing
    # robot faces in positive x direction in odom and drives 1 m forward.
    # tests particle facing +y will move 1 m in +y, not +x.
    # tolerances need to be large because of noise in the odom update
    p = Particle(x=0, y=0, theta=math.pi / 2) # makes a partcile facing +y
    pf = make_odom_pf([p], old_odom=(0, 0, 0), new_odom=(5, 0, 0)) # robot moves 1 m forward in +x
    pf.update_particles_with_odom()
    assert abs(p.x - 0) < 0.1 # checks that x still is 0
    assert abs(p.y - 5) < 0.1 # checks that y is now 1 


def test_odom_rotation():
    #checks that the partcile will not move if the robot rotates in place
    p = Particle(x=3, y=4, theta=0.1) # makes a particle at 3,4 with a heading of 0.1 rad
    pf = make_odom_pf([p], old_odom=(0, 0, 0), new_odom=(0, 0, 0.5)) # robot rotates 0.5 rad
    pf.update_particles_with_odom()
    assert abs(p.x - 3) < 0.05 #chcek that x is still 3
    assert abs(p.y - 4) < 0.05 #chcek that y is still 4
    assert abs(p.theta - 0.6) < 0.05 #check the heading is 0.6 (0.1 + 0.5)


def test_odom_updates_reference_pose():
    #checks that the odom update also updates the refrence pose
    pf = make_odom_pf([Particle()], old_odom=(0, 0, 0), new_odom=(1, 2, 0.3)) #robot moves to 1,2 with a heading of 0.3 rad
    pf.update_particles_with_odom()
    assert pf.current_odom_xy_theta == [1, 2, 0.3] # checks that the new odom pose is where the robot moved