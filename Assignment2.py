import numpy as np
import numba as nb
import matplotlib.pyplot as plt

kB = 1.3806452e-23
sigma = 0.3345e-9
epsilon = 125.7*kB
tau = np.sqrt(sigma**2 / epsilon)

class ParticleSystem:

	def __init__(self, num_particles: int, dim: int, dt: float, num_steps: int, L: np.ndarray):

	"""
        Initialize the ParticleSystem object.

        Args:
            num_particles (int): Number of particles in the system.
            dim (int): Dimensionality of the system.
            dt (float): Timestep for the simulation.
            num_steps (int): Number of simulation steps to perform.
            L (np.ndarray): Array representing the size of the system in each dimension.
        """

		self.num_particles = num_particles
		self.dim = dim
		self.dt = dt
		self.num_steps = num_steps
		self.L = L

		self.pos = np.zeros((num_particles, dim))
		self.vel = np.zeros((num_particles, dim))

	def initialize_particles(self, initial_positions: np.ndarray):

        """
        Initialize the positions of the particles.

        Args:
            initial_positions (np.ndarray): Initial positions of the particles.
        """

		self.pos = initial_positions

	@staticmethod
	@nb.njit(parallel=True, fastmath=True)
	def update_lj(pos: np.ndarray, num_particles: int, dim: int, L: np.ndarray):

  	"""
        Update the forces on particles based on Lennard-Jones potential.

        Args:
            pos (np.ndarray): Array of particle positions.
            num_particles (int): Number of particles.
            dim (int): Dimensionality of the system.
            L (np.ndarray): Array representing the size of the system in each dimension.

        Returns:
            np.ndarray: Array of forces acting on particles.
        """

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

        """
        Calculate particle velocities.

        Args:
            acc (np.ndarray): Array of particle accelerations.

        Returns:
            np.ndarray: Updated velocities.
        """

		return self.vel + (self.dt / 2) * acc

	def r(self):

        """
        Update particle positions.

        Returns:
            np.ndarray: Updated positions.
        """

		return self.pos + self.vel * self.dt

	def reflect(self):

        """
        Reflect particles at system boundaries.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated positions and velocities.
        """

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

        """
        Calculate pressure in the system.

        Returns:
            float: Pressure in the system.
        """

		L = self.L / sigma
		pos = self.pos / sigma
		vel = self.vel * (sigma/tau)
		area = L[1] * L[2]  
		crossing_momentum = np.sum(vel[:, 0][np.logical_and(pos[:, 0] >= L[0] / 2, vel[:, 0] > 0)])
		t = self.dt / tau
		pressure_x = crossing_momentum / (area * t)

		return pressure_x

	def calculate_temperature(self):

        """
        Calculate temperature in the system.

        Returns:
            float: Temperature in the system.
        """

		L = self.L / sigma
		pos = self.pos / sigma
		vel = self.vel * (sigma/tau)
		kinetic_energy = 0.5 * np.sum(vel ** 2)
		temperature = (2 / (3 * self.num_particles*kB)) * (kinetic_energy)

		return temperature

	def simulate(self):

        """
        Perform the simulation.

        Returns:
            np.ndarray: Array of particle positions over simulation steps.
        """

		new_pos = np.zeros((self.num_steps, self.num_particles, self.dim))
		for i in range(self.num_steps):
			new_pos[i] = self.pos
			acc = self.update_lj(self.pos, self.num_particles, self.dim, self.L)
			self.vel = self.v(acc)
			self.pos = self.r()
			self.reflect()
		return new_pos

	def simulate2(self):

        """
        Perform the simulation and calculate average pressure and temperature.

        Returns:
            Tuple[float, float]: Average pressure and temperature.
        """

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
