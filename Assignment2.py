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

	def initialize_particles(self, initial_positions: np.ndarray):
        	self.pos = initial_positions

	@staticmethod
	@nb.njit(parallel=True, fastmath=True) 
	def update_lj(pos: np.ndarray, num_particles: int, dim: int, L: np.ndarray):
        	forces = np.zeros((num_particles, dim))
        	for i in nb.prange(num_particles):
            		for j in range(num_particles):
                	if i != j:
                    	rij = pos[j] - pos[i]
                    	r = np.linalg.norm(rij)
                    	r_unit = rij / r
                    	r7 = (1/r)**7
                    	r13 = (1/r)**13
                    	forces[i] += 24 * (-2*r13 + r7) * r_unit
        	return forces

    	def v(self, acc: np.ndarray):
        	return self.vel + (self.dt / 2) * acc
    
    	def r(self):
        	return self.pos + self.vel * self.dt
 
    	def reflect(self):
        	half = self.L / 2
        	for i in range(self.num_particles):
            		for d in range(self.dim):
                		if self.pos[i, d] >= half[d]:
                    			self.pos[i, d] = self.L[d] - self.pos[i, d]
                    			self.vel[i, d] = -self.vel[i, d]
                		elif self.pos[i, d] <= -half[d]:
                    			self.pos[i, d] = -half[d] - self.pos[i, d]
                    			self.vel[i, d] = -self.vel[i, d]
		return self.pos, self.vel
