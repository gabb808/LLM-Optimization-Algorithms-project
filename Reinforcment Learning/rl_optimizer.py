import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
# FIX: Use torch.multiprocessing instead of concurrent.futures
import torch.multiprocessing as mp
from model import TinyModel

# ==========================================
# 1. Standalone Worker Function 
# ==========================================
def train_worker(config):
    """
    Runs in a separate process. Trains the model and returns loss.
    """
    # Re-import needed libraries inside the worker to ensure context on Windows
    import torch
    from model import TinyModel
    
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        # Fallback if file read fails
        return 10.0 

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    
    # Limit data size for speed proxy
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Force CPU for workers to avoid CUDA multiprocessing complexity on Windows
    device = 'cpu' 
    block_size = 64
    training_iters = 100 

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (config['batch_size'],))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    try:
        model = TinyModel(
            vocab_size=vocab_size,
            n_embd=config['n_embd'],
            n_heads=4,
            n_layers=config['n_layer'],
            dropout=config['dropout']
        ).to(device)
        
        # Use a slightly higher LR for the proxy task to learn fast
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        model.train()

        for _ in range(training_iters):
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # Quick validation (avg of 5 batches)
            losses = torch.zeros(5)
            for k in range(5):
                X, Y = get_batch('val')
                _, loss = model(X, Y)
                losses[k] = loss.item()
            val_loss = losses.mean().item()
            
        return val_loss
        
    except Exception as e:
        # If training crashes (e.g. OOM or bad params), return high loss
        return 10.0 

# ==========================================
# 2. The Logic Environment
# ==========================================
class LogicEnv:
    def __init__(self):
        self.param_choices = {
            0: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],     
            1: [16, 32, 64],                       
            2: [2, 4, 6],                          
            3: [32, 64, 128],                      
            4: [0.0, 0.1, 0.2],                    
        }
        self.param_names = ['lr', 'batch_size', 'n_layer', 'n_embd', 'dropout']
        self.state = 0
        self.current_config = {}

    def reset(self):
        self.state = 0
        self.current_config = {}
        return self.state

    def step(self, action_index):
        param_name = self.param_names[self.state]
        if action_index >= len(self.param_choices[self.state]): action_index = 0
        
        selected_value = self.param_choices[self.state][action_index]
        self.current_config[param_name] = selected_value
        
        # Reward Shaping (Compute Cost Penalty)
        step_penalty = 0
        if param_name == 'n_layer': step_penalty = selected_value * 0.02 
        elif param_name == 'n_embd': step_penalty = selected_value * 0.001

        self.state += 1
        terminated = (self.state >= len(self.param_names))
        
        return self.state, -step_penalty, terminated, self.current_config

# ==========================================
# 3. The DQN Agent
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): return self.net(x)

class ParallelDQNAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.95 # Faster decay for demo
        self.min_epsilon = 0.1
        
        self.num_states = len(env.param_names) + 1
        self.max_actions = max(len(opts) for opts in env.param_choices.values())
        
        # Check for CUDA for the AGENT (Logic), workers stay CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.policy_net = QNetwork(self.num_states, self.max_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def get_state_tensor(self, state_idx):
        x = np.zeros(self.num_states)
        x[state_idx] = 1.0
        return torch.FloatTensor(x).unsqueeze(0).to(self.device)

    def choose_action(self, state_idx):
        possible_actions = len(self.env.param_choices[state_idx])
        if np.random.rand() < self.epsilon:
            return np.random.randint(possible_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(self.get_state_tensor(state_idx))
                # We slice to only valid actions for this state
                valid_q = q_values[0, :possible_actions]
                return torch.argmax(valid_q).item()

    def train_parallel(self, total_episodes=40, num_workers=4):
        rewards_history = []
        best_loss = float('inf')
        best_config = None
        
        batches = total_episodes // num_workers
        
        print(f"Starting Parallel Training: {total_episodes} episodes using {num_workers} workers.")

        # FIX: Use torch.multiprocessing.Pool instead of ProcessPoolExecutor
        # This handles PyTorch contexts on Windows correctly
        with mp.Pool(processes=num_workers) as pool:
            
            for batch in range(batches):
                print(f"\n--- Batch {batch+1}/{batches} (Epsilon: {self.epsilon:.2f}) ---")
                
                batch_trajectories = []
                batch_configs = []
                
                # 1. Generate Trajectories (Logic Step)
                for _ in range(num_workers):
                    state_idx = self.env.reset()
                    terminated = False
                    trajectory = []
                    
                    while not terminated:
                        action = self.choose_action(state_idx)
                        next_state_idx, reward, terminated, info = self.env.step(action)
                        trajectory.append((state_idx, action, reward, next_state_idx))
                        state_idx = next_state_idx
                    
                    batch_trajectories.append(trajectory)
                    batch_configs.append(info)

                # 2. Execute Training in Parallel (Heavy Step)
                # Map the train_worker function to the list of configs
                results = pool.map(train_worker, batch_configs)

                # 3. Update Q-Network with results
                for i, val_loss in enumerate(results):
                    print(f"   Config {i+1}: {batch_configs[i]} -> Loss: {val_loss:.4f}")
                    
                    accuracy_reward = 10.0 / (val_loss + 1e-6)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_config = batch_configs[i]
                        print(f"   >>> NEW BEST FOUND! Loss: {best_loss:.4f}")

                    trajectory = batch_trajectories[i]
                    # Last reward in trajectory is the Accuracy Reward
                    
                    # Update loop (Reverse order for Bellman update)
                    # For the terminal step, target is just the reward.
                    # For previous steps, target is reward + gamma * max(next_Q)
                    
                    # Calculate the final accumulated reward for this trajectory
                    # (Sum of step penalties + final accuracy reward)
                    cumulative_reward = accuracy_reward
                    
                    for (s_idx, a, r_step, s_next_idx) in reversed(trajectory):
                        state_tensor = self.get_state_tensor(s_idx)
                        
                        # Current prediction
                        pred_q = self.policy_net(state_tensor)[0, a]
                        
                        # Target calculation
                        with torch.no_grad():
                            if s_next_idx == len(self.env.param_names):
                                target = torch.tensor(cumulative_reward + r_step).to(self.device)
                            else:
                                next_state_tensor = self.get_state_tensor(s_next_idx)
                                next_qs = self.policy_net(next_state_tensor)
                                valid_actions = len(self.env.param_choices[s_next_idx])
                                max_next_q = torch.max(next_qs[0, :valid_actions])
                                target = r_step + (self.gamma * max_next_q)

                        # Optimization Step
                        loss = self.loss_fn(pred_q, target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                    rewards_history.append(accuracy_reward)

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return best_config, best_loss, rewards_history

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # CRITICAL FOR WINDOWS MULTIPROCESSING
    # This prevents the recursive spawning loop
    mp.set_start_method('spawn', force=True)

    import matplotlib.pyplot as plt

    WORKERS = 4 
    EPISODES = 200 # Kept small for testing, increase for final result

    env = LogicEnv()
    agent = ParallelDQNAgent(env)
    
    start_time = time.time()
    best_conf, best_val, hist = agent.train_parallel(total_episodes=EPISODES, num_workers=WORKERS)
    total_time = time.time() - start_time

    print("\n" + "="*40)
    print(f"Parallel Optimization Finished in {total_time:.2f}s")
    print(f"Best Loss: {best_val:.4f}")
    print(f"Best Config: {best_conf}")
    print("="*40)

    loss_hist = [10.0/(r+1e-6) for r in hist]
    best_so_far = np.minimum.accumulate(loss_hist)
    
    plt.figure(figsize=(10,6))
    plt.plot(loss_hist, alpha=0.4, label='Raw Episode Loss')
    plt.plot(best_so_far, color='blue', linewidth=2, label='Best So Far')
    plt.title(f"Parallel Q-Learning Optimization ({WORKERS} workers)")
    plt.xlabel("Total Episodes")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig('final_results.png')
    print("Graph saved.")