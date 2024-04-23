import numpy as np
import numba as nb
import matplotlib.pyplot as plt

kB = 1.3806452e-23
sigma = 0.3345e-9
epsilon = 125.7*kB
tau = np.sqrt(sigma**2 / epsilon)

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

	def calculate_pressure(self):
		L = self.L / sigma
		pos = self.pos / sigma
		vel = self.vel * (sigma/tau)
		area = L[1] * L[2]  
		crossing_momentum = np.sum(vel[:, 0][pos[:, 0] >= L[0] / 2])  
		t = self.dt / tau
		pressure_x = crossing_momentum / (area * t)

		return pressure_x

	def calculate_temperature(self):
		L = self.L / sigma
		pos = self.pos / sigma
		vel = self.vel * (sigma/tau)
		kinetic_energy = 0.5 * np.sum(vel ** 2)
		temperature = (2 / (3 * self.num_particles*kB)) * (kinetic_energy)

		return temperature

	def simulate(self):
		new_pos = np.zeros((self.num_steps, self.num_particles, self.dim))
		for i in range(self.num_steps):
			new_pos[i] = self.pos
			acc = self.update_lj(self.pos, self.num_particles, self.dim, self.L)
			self.vel = self.v(acc)
			self.pos = self.r()
			self.reflect()
		return new_pos

	def simulate2(self):
		pressures = []
		temperatures = []
		for i in range(self.num_steps):
			acc = self.update_lj(self.pos, self.num_particles, self.dim, self.L)
			self.vel = self.v(acc)
			self.pos = self.r()
			self.reflect()

			pressure_x = self.calculate_pressure()
			temperature = self.calculate_temperature()
			pressures.append(pressure_x)
			temperatures.append(temperature)

		avg_pressure = np.mean(pressures)
		avg_temperature = np.mean(temperatures)

		return avg_pressure, avg_temperature
