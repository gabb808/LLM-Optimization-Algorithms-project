import numpy as np
from src.trainer import evaluate_config

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.98, min_epsilon=0.1):
        self.alpha = alpha        # Learning Rate: How fast I accept new information
        self.gamma = gamma        # Discount Factor: How much I care about future rewards vs immediate ones
        self.epsilon = epsilon    # Exploration Rate: Probability of trying something random
        self.epsilon_decay = epsilon_decay # How fast I stop exploring and start exploiting
        self.min_epsilon = min_epsilon     # I always want to keep a tiny bit of exploration
        
        # SEARCH SPACE
        # I'm defining the hyperparameters and their possible values right here.
        # The order matters because I treat this as a sequence (State 0 -> State 1 -> etc.)
        lr_options = np.geomspace(0.0001, 0.01, num=50).tolist()
        dropout_options = np.linspace(0.0, 0.5, num=21).tolist()
        
        self.param_specs = [
            ('lr', lr_options),
            ('dropout', dropout_options),
            ('batch_size', [16, 32, 64, 128]),
            ('n_embd', [32, 64, 128]),
            ('n_layer', [2, 4, 6])
        ]
        
        # Helper lists to access names and choices easily
        self.param_names = [p[0] for p in self.param_specs]
        self.param_choices = [p[1] for p in self.param_specs]
        
        max_choices = max(len(opts) for opts in self.param_choices)
        num_states = len(self.param_names) + 1 # +1 for the terminal state
        self.q_table = np.zeros((num_states, max_choices))

    def choose_action(self, state_idx):
        """Decide which option to pick for the current parameter."""
        possible_actions = len(self.param_choices[state_idx])
        
        # Epsilon-Greedy Logic
        if np.random.rand() < self.epsilon:
            return np.random.randint(possible_actions)
        else:
            # Pick the action with the highest Q-value for this state
            return np.argmax(self.q_table[state_idx, :possible_actions])

    def train(self, episodes):
        rewards_history = []
        best_loss = float('inf')
        best_config = None

        for episode in range(episodes):
            # Print progress every 5 episodes so I know it's working
            if (episode+1) % 5 == 0:
                print(f"--- Episode {episode+1}/{episodes} (Epsilon: {self.epsilon:.2f}) ---")
            
            # Reset for the new episode
            current_config = {}
            trajectory = [] # Keep track of my choices: (state, action, next_state)
            
            # I iterate through my parameters one by one (Sequential Decision Process)
            for state_idx in range(len(self.param_names)):
                
                # Make a choice
                action_idx = self.choose_action(state_idx)
                
                # Decode the choice (index -> actual value) and store it
                param_name = self.param_names[state_idx]
                chosen_value = self.param_choices[state_idx][action_idx]
                current_config[param_name] = chosen_value
                
                # Move to the next parameter
                next_state_idx = state_idx + 1
                
                # Record this transition so I can learn from it later
                trajectory.append((state_idx, action_idx, next_state_idx))

            # Now that I have a full set of hyperparameters, I train the model to get the loss
            loss = evaluate_config(current_config)
            
            # Convert Loss to Reward (Lower loss = Higher reward)
            # Adding 1e-8 to avoid division by zero
            final_reward = 10.0 / (loss + 1e-8)
            
            # Keep track of my personal best
            if loss < best_loss:
                best_loss = loss
                best_config = current_config.copy()
                print(f"   >>> NEW BEST FOUND! Loss: {best_loss:.4f}")

            # I walk backward from the result to the first decision to update Q-values
            for (s, a, s_next) in reversed(trajectory):
                
                # What is the potential of the NEXT state? (Max Q of next state)
                if s_next < len(self.param_names):
                    best_next_q = np.max(self.q_table[s_next, :len(self.param_choices[s_next])])
                else:
                    best_next_q = 0 # No future value after the last step
                
                # I only get the actual reward at the very end step
                current_r = final_reward if s_next == len(self.param_names) else 0
                
                # Bellman Equation Update:
                # New Q = Old Q + Alpha * (Reward + Gamma * Next_Best - Old Q)
                self.q_table[s, a] += self.alpha * (current_r + self.gamma * best_next_q - self.q_table[s, a])

            # Decay epsilon (reduce exploration over time)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_history.append(final_reward)

        return best_config, best_loss, rewards_history