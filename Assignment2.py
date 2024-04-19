import numpy as np

class Nbody:
	def __init__(self, particles, initp, initv, epsilon, sigma, argonmass):
		self.particles = particles
		self.positions = np.array(initp, dtype=np.float64)
		self.velocities = np.array(initv, dtype=np.float64)
		self.accelerations = np.zeros((particles, 3), dtype=np.float64)
		self.epsilon = epsilon
		self.sigma = sigma
		self.argonmass = argonmass
		self.tau = ((self.argonmass * self.sigma**2) / self.epsilon)**0.5

