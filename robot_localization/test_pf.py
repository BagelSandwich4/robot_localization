import math
from unittest.mock import MagicMock
from pf import ParticleFilter, Particle 


def make_pf(particles):
    pf = ParticleFilter.__new__(ParticleFilter)   # creates an instance without calling __init__
    pf.particle_cloud = particles
    pf.n_particles = len(particles)
    return pf


def test_normalize_sums_to_one():
    #tests that normalized weights add up to 1.0 with a small tolerance
    pf = make_pf([Particle(w=weight) for weight in (1.0, 2.0, 5.0)])
    pf.normalize_particles()
    assert abs(sum(p.w for p in pf.particle_cloud) - 1.0) < 1e-9


def test_normalize_all_zero_weights():
    #tests that if all weights are zero, they will be set to equal values to prevent division by zero
    pf = make_pf([Particle(w=0.0) for _ in range(4)])
    pf.normalize_particles()
    assert all(abs(p.w - 0.25) < 1e-9 for p in pf.particle_cloud)

def test_resample():
    good = Particle(x=5, y=5, w=0.99)
    pf = make_pf([good] + [Particle(x=0, y=0, w=0.0001) for _ in range(9)])
    pf.resample_particles()
    assert len(pf.particle_cloud) == 10
    assert good not in pf.particle_cloud                       # checks that the very good particle from the beggining was not just copied
    assert len({id(p) for p in pf.particle_cloud}) == 10       # checks that all paticles are unique
    near_good = sum(1 for p in pf.particle_cloud if abs(p.x - 5) < 0.5) # counts how many particles are close to the good particle 
    assert near_good >= 8                                      # makes sure that the resampling uses the good particle