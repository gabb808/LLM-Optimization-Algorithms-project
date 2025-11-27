import random

class GeneticAlgorithm:
    def __init__(self, pop_size=10, mutation_rate=0.25):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        # on definie les valeurs que peuvent prendre les hyperparametres 
        self.bounds = {
            'lr': (0.0001, 0.01),
            'dropout': (0.0, 0.5)
        }
        self.choices = {
            'batch_size': [16, 32, 64, 128],
            'n_embd': [32, 64, 128], # taille du modele
            'n_layer': [2, 4, 6],   # profondeur du modele
            }

    def init_population(self):
        # on cree la population initiale
        population = []
        for _ in range(self.pop_size):
            genome = {}
            # Paramètres continus
            for k, (min_v, max_v) in self.bounds.items():
                genome[k] = random.uniform(min_v, max_v)
            # Paramètres discrets
            for k, options in self.choices.items():
                genome[k] = random.choice(options)
            population.append(genome)
        return population
    
    def mutate(self, genome):
        # on cree des mutations en modifiant legerement les parametres
        mutant = genome.copy()
        # 1. Mutation des paramètres continus (petits ajustements)
        for k, (min_v, max_v) in self.bounds.items():
            if random.random() < self.mutation_rate:
                # On multiplie par un facteur entre 0.8 et 1.2 (Variation de 20%)
                change = random.uniform(0.8, 1.2)
                new_val = mutant[k] * change
                # On clip pour rester dans les bornes
                mutant[k] = max(min_v, min(new_val, max_v))

        # 2. Mutation des paramètres discrets (changement complet)
        for k, options in self.choices.items():
            if random.random() < self.mutation_rate:
                mutant[k] = random.choice(options)
        
        return mutant
    

    
    def crossover(self, parent1, parent2):
        # on cree un enfant en combinant les genes de deux parents
        child = {}
        for key in list(self.bounds.keys()) + list(self.choices.keys()):
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child
    
    def evolve(self, population, scores):
        # séléction naturelle: qui survit et qui se reproduit
        pop_with_scores = list(zip(population, scores))
        next_gen = []   
        # 1 on associe chaque individue a sons score et on les classe ( le meilleur est celui qui a la plus petite loss)
        sorted_pop = sorted(pop_with_scores, key=lambda x: x[1]) # Tri par Loss croissante (Plus petit = Mieux)
        next_gen.append(sorted_pop[0][0]) # Le 1er
        next_gen.append(sorted_pop[1][0]) # Le 2eme
        # 3 reproduction: on remplit le reste de la population par croisement et mutation

      # 2. REPRODUCTION PAR TOURNOI
        while len(next_gen) < self.pop_size:
            # Tournoi pour Parent 1
            # On prend 3 individus au hasard, on garde le meilleur
            fighters1 = random.sample(pop_with_scores, 3)
            p1 = min(fighters1, key=lambda x: x[1])[0] # Celui avec la plus petite loss gagne

            # Tournoi pour Parent 2
            fighters2 = random.sample(pop_with_scores, 3)
            p2 = min(fighters2, key=lambda x: x[1])[0]

            # Naissance
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            next_gen.append(child)

        return next_gen
    

