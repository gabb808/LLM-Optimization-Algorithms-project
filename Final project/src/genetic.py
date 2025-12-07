import random

class GeneticAlgorithm:
    def __init__(self, pop_size=10, mutation_rate=0.25):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        # Define the values that hyperparameters can take
        self.bounds = {
            'lr': (0.0001, 0.01), # learning rate aka the speed of learning
            'dropout': (0.0, 0.5) # dropout rate aka to prevent the model from overfitting (learning by heart)
        }
        self.choices = {
            'batch_size': [16, 32, 64, 128], # batch size: number of data samples processed simultaneously
            'n_embd': [32, 64, 128], # (Width): The model's "memorization capacity"
            'n_layer': [2, 4, 6],   # Model depth: The "reasoning capacity" (abstraction)
            }

    def init_population(self):
        # Create the initial population
        population = []
        for _ in range(self.pop_size):
            genome = {}
            # Continuous parameters
            for k, (min_v, max_v) in self.bounds.items():
                genome[k] = random.uniform(min_v, max_v)
            # Discrete parameters
            for k, options in self.choices.items():
                genome[k] = random.choice(options)
            population.append(genome)
        return population
    
    def mutate(self, genome):
        # Create mutations by slightly modifying parameters
        mutant = genome.copy()
        # 1. Mutation of continuous parameters (small adjustments)
        for k, (min_v, max_v) in self.bounds.items():
            if random.random() < self.mutation_rate:
                # Multiply by a factor between 0.8 and 1.2 (20% variation)
                change = random.uniform(0.8, 1.2)
                new_val = mutant[k] * change
                # Clip to stay within bounds
                mutant[k] = max(min_v, min(new_val, max_v))

        # 2. Mutation of discrete parameters (complete change)
        for k, options in self.choices.items():
            if random.random() < self.mutation_rate:
                mutant[k] = random.choice(options)
        
        return mutant
    

    
    def crossover(self, parent1, parent2):
        # Create a child by combining genes from two parents
        child = {}
        for key in list(self.bounds.keys()) + list(self.choices.keys()):
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child
    
    def evolve(self, population, scores):
        # Natural selection: who survives and who reproduces
        pop_with_scores = list(zip(population, scores))
        next_gen = []   
        # 1. Associate each individual with its score and sort them (the best is the one with the lowest loss)
        sorted_pop = sorted(pop_with_scores, key=lambda x: x[1]) # Sort by increasing Loss (Smaller = Better)
        next_gen.append(sorted_pop[0][0]) # The 1st
        next_gen.append(sorted_pop[1][0]) # The 2nd
        # 3. Reproduction: fill the rest of the population via crossover and mutation

      # 2. REPRODUCTION VIA TOURNAMENT
        while len(next_gen) < self.pop_size:
            # Tournament for Parent 1
            # Pick 3 random individuals, keep the best
            fighters1 = random.sample(pop_with_scores, 3)
            p1 = min(fighters1, key=lambda x: x[1])[0] # The one with the lowest loss wins

            # Tournament for Parent 2
            fighters2 = random.sample(pop_with_scores, 3)
            p2 = min(fighters2, key=lambda x: x[1])[0]

            # Birth
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            next_gen.append(child)

        return next_gen