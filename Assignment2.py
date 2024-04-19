import numpy as np

class Nbody:
	def __init__(self, particles, initp, initv, epsilon, sigma, argonmass):
		self.particles = particles
		self.positions = np.array(initp)
		self.velocities = np.array(initv)
		self.accelerations = np.zeros((particles, 3)
		self.epsilon = epsilon
		self.sigma = sigma
		self.argonmass = argonmass
		self.tau = ((self.argonmass * self.sigma**2) / self.epsilon)**0.5

