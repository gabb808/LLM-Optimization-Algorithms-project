import random
import numpy as np

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.05):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Espace d'actions (Doit correspondre à ce que evaluate_config attend)
        self.param_choices = [
            ('lr', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
            ('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            ('batch_size', [16, 32, 64, 128]),
            ('n_embd', [32, 64, 128]),
            ('n_layer', [2, 4, 6])
        ]
        
        # Q-Table : Dictionnaire {(state_index, action_index): value}
        self.q_table = {}

    def get_q(self, state_idx, action_idx):
        return self.q_table.get((state_idx, action_idx), 0.0)

    def choose_config(self):
        """L'agent construit une config étape par étape"""
        chosen_config = {}
        chosen_indices = []
        
        # Pour chaque paramètre (État 0, État 1, ...)
        for state_idx, (param_name, options) in enumerate(self.param_choices):
            
            # Epsilon-Greedy
            if random.random() < self.epsilon:
                action_idx = random.randint(0, len(options) - 1) # Exploration
            else:
                # Exploitation : Trouver l'action avec le max Q pour cet état
                q_values = [self.get_q(state_idx, a) for a in range(len(options))]
                if not q_values: # Sécurité si q_values est vide
                    max_q = 0
                else:
                    max_q = max(q_values)
                
                # Astuce : Si plusieurs max égaux, on choisit au hasard parmi eux
                best_actions = [i for i, q in enumerate(q_values) if q == max_q]
                if not best_actions:
                    action_idx = random.randint(0, len(options) - 1)
                else:
                    action_idx = random.choice(best_actions)
            
            chosen_config[param_name] = options[action_idx]
            chosen_indices.append(action_idx)
            
        return chosen_config, chosen_indices

    def learn(self, action_indices, reward):
        """Mise à jour de la Q-Table"""
        for state_idx, action_idx in enumerate(action_indices):
            old_q = self.get_q(state_idx, action_idx)
            # Formule Q-Learning simplifiée (Bandit contextuel séquentiel)
            new_q = old_q + self.lr * (reward - old_q)
            self.q_table[(state_idx, action_idx)] = new_q
            
        # Réduire l'exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)