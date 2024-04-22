import numpy as np
import numba as nb
import matplotlib.pyplot as plt

class ParticleSystem:
	def __init__(self, num_particles: int, dim: int, dt: float, num_steps: int, L: np.ndarray):
	self.num_particles = num_particles
        self.dim = dim
        self.dt = dt
        self.num_steps = num_steps
        self.L = L
        
        self.pos = np.zeros((num_particles, dim))
        self.vel = np.zeros((num_particles, dim))
