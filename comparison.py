from ofunctions import sphere, rastrigin, ackley
from optimizers import PSO, GA, DE
import pandas as pd

def benchmark():
    dimensions = 2
    bounds = (-5, 5)
    max_iterations = 100

    # Define objective functions
    functions = {
        "Rastrigin": rastrigin,
        "Ackley": ackley,
        "Sphere": sphere
    }

    # Initialize a dictionary to store results for each function
    results = {name: [] for name in functions.keys()}

    for name, function in functions.items():
        print(f"\nBenchmarking on {name} Function:")

        # Run PSO
        pso = PSO(num_particles=30, dimensions=dimensions, bounds=bounds, max_iterations=max_iterations)
        pso_solution, pso_score = pso.optimize(function)
        results[name].append(("PSO", pso_score))

        # Run GA
        ga = GA(population_size=30, dimensions=dimensions, bounds=bounds, max_iterations=max_iterations)
        ga_solution, ga_score = ga.optimize(function)
        results[name].append(("GA", ga_score))


        # Run DE
        de = DE(population_size=30, dimensions=dimensions, bounds=bounds, max_iterations=max_iterations)
        de_solution, de_score = de.optimize(function)
        results[name].append(("DE", de_score))


    # Convert results to DataFrames and display
    for name in functions.keys():
        df = pd.DataFrame(results[name], columns=["Algorithm", "Best Score"])
        print(f"\nResults for {name} Function:")
        print(df)
        print("\n" + "-"*40 + "\n")

benchmark()