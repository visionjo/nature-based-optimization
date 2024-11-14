from abc import ABC, abstractmethod
import numpy as np
class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms."""

    def __init__(self, dimensions: int, bounds: tuple, max_iterations: int):
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations

    @abstractmethod
    def optimize(self, objective_function: callable) -> tuple:
        """Run the optimization and return the best position and score."""
        pass

class PSO(OptimizationAlgorithm):
    """Particle Swarm Optimization implementation."""

    class Particle:
        """Represents a particle in the PSO algorithm."""

        def __init__(self, dimensions: int, bounds: tuple):
            self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
            self.velocity = np.random.uniform(-1, 1, dimensions)
            self.best_position = np.copy(self.position)
            self.best_score = float('inf')

        def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
            r1, r2 = np.random.rand(len(self.position)), np.random.rand(len(self.position))
            cognitive_component = c1 * r1 * (self.best_position - self.position)
            social_component = c2 * r2 * (global_best_position - self.position)
            self.velocity = w * self.velocity + cognitive_component + social_component

        def update_position(self, bounds: tuple):
            self.position += self.velocity
            self.position = np.clip(self.position, bounds[0], bounds[1])

    def __init__(self, num_particles: int, dimensions: int, bounds: tuple, max_iterations: int):
        super().__init__(dimensions, bounds, max_iterations)
        self.num_particles = num_particles
        self.swarm = [self.Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.global_best_score = float('inf')

    def optimize(self, objective_function: callable) -> tuple:
        w, c1, c2 = 0.5, 1.5, 1.5  # PSO parameters

        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                score = objective_function(particle.position)
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(particle.position)

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position(self.bounds)

            print(f"PSO Iteration {iteration + 1}, Best Score: {self.global_best_score}")

        return self.global_best_position, self.global_best_score

class GA(OptimizationAlgorithm):
    """Genetic Algorithm implementation."""

    def __init__(self, population_size: int, dimensions: int, bounds: tuple, max_iterations: int, mutation_rate: float = 0.01):
        super().__init__(dimensions, bounds, max_iterations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [np.random.uniform(bounds[0], bounds[1], dimensions) for _ in range(population_size)]

    def select_parent(self, fitness_values: np.ndarray) -> np.ndarray:
        probabilities = fitness_values / np.sum(fitness_values)
        return self.population[np.random.choice(len(self.population), p=probabilities)]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        crossover_point = np.random.randint(1, self.dimensions)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def mutate(self, individual: np.ndarray):
        for i in range(self.dimensions):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(self.bounds[0], self.bounds[1])

    def optimize(self, objective_function: callable) -> tuple:
        best_score = float('inf')
        best_solution = None

        for iteration in range(self.max_iterations):
            fitness_values = np.array([1 / (1 + objective_function(individual)) for individual in self.population])

            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.select_parent(fitness_values)
                parent2 = self.select_parent(fitness_values)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                self.mutate(child1)
                self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population
            for individual in self.population:
                score = objective_function(individual)
                if score < best_score:
                    best_score = score
                    best_solution = individual

            print(f"GA Iteration {iteration + 1}, Best Score: {best_score}")

        return best_solution, best_score

class DE(OptimizationAlgorithm):
    """Differential Evolution implementation."""

    def __init__(self, population_size: int, dimensions: int, bounds: tuple, max_iterations: int, mutation_factor: float = 0.8, crossover_prob: float = 0.7):
        super().__init__(dimensions, bounds, max_iterations)
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.population = [np.random.uniform(bounds[0], bounds[1], dimensions) for _ in range(population_size)]

    def optimize(self, objective_function: callable) -> tuple:
        best_score = float('inf')
        best_solution = None

        for iteration in range(self.max_iterations):
            new_population = []

            for i in range(self.population_size):
                # Mutation: select three random, distinct individuals
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                a, b, c = self.population[indices[0]], self.population[indices[1]], self.population[indices[2]]

                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover: mix mutant with target
                target = self.population[i]
                trial = np.where(np.random.rand(self.dimensions) < self.crossover_prob, mutant, target)

                # Selection: keep the better solution
                if objective_function(trial) < objective_function(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)

            self.population = new_population
            for individual in self.population:
                score = objective_function(individual)
                if score < best_score:
                    best_score = score
                    best_solution = individual

            print(f"DE Iteration {iteration + 1}, Best Score: {best_score}")

        return best_solution, best_score