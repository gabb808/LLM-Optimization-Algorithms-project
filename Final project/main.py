from src.genetic import GeneticAlgorithm
from src.rl_agent import QLearningAgent
from src.trainer import evaluate_config
import matplotlib.pyplot as plt
import time
import numpy as np

# --- GLOBAL PARAMETERS ---
POP_SIZE = 20        
GENERATIONS = 10     
RL_EPISODES = POP_SIZE * GENERATIONS # = 200, to have the same budget

def run_genetic_vs_random():
    print(f"\n🧬 [MODE 1] Detailed Analysis: Genetic vs Random...")
    start = time.time()
    
    # 1. GENETIC ALGORITHM
    ga = GeneticAlgorithm(pop_size=POP_SIZE, mutation_rate=0.25)
    pop = ga.init_population()
    
    hist_ga_best = []
    hist_ga_avg = []
    global_ga_best = float('inf')
    
    print("  > Run GA...")
    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
        scores = []
        for i, ind in enumerate(pop):
            loss = evaluate_config(ind)
            scores.append(loss)
            # --- ADDED: Detailed display per individual ---
            print(f"    > Ind {i+1:02d}: Loss={loss:.4f} | LR={ind['lr']:.5f}")
        
        # Stats GA
        gen_best = min(scores)
        gen_avg = sum(scores) / len(scores)
        
        if gen_best < global_ga_best: global_ga_best = gen_best
        
        hist_ga_best.append(global_ga_best)
        hist_ga_avg.append(gen_avg)
        
        print(f"  🏆 Stats Gen {gen+1}: Global Record={global_ga_best:.4f} | Avg={gen_avg:.4f}")
        
        if gen < GENERATIONS - 1:
            pop = ga.evolve(pop, scores)

    # 2. RANDOM SEARCH (By Batch)
    print("\n  > Run Random...")
    ga_rnd = GeneticAlgorithm(pop_size=1)
    
    hist_rnd_best = []
    hist_rnd_avg = []
    global_rnd_best = float('inf')
    
    for gen in range(GENERATIONS):
        print(f"\n--- Random Batch {gen+1}/{GENERATIONS} ---")
        batch_scores = []
        for i in range(POP_SIZE):
            cfg = ga_rnd.init_population()[0]
            loss = evaluate_config(cfg)
            batch_scores.append(loss)
            # --- ADDED: Detailed display per random trial ---
            print(f"    > Random Trial {i+1:02d}: Loss={loss:.4f}")
            
        # Stats Random
        batch_min = min(batch_scores)
        batch_avg = sum(batch_scores) / len(batch_scores)
        
        if batch_min < global_rnd_best: global_rnd_best = batch_min
        
        hist_rnd_best.append(global_rnd_best)
        hist_rnd_avg.append(batch_avg)
        
        print(f"  🎲 Stats Batch {gen+1}: Record={global_rnd_best:.4f} | Batch Avg={batch_avg:.4f}")

    print(f"✅ Finished in {time.time()-start:.1f}s")
    
    # --- FULL GRAPH ---
    plt.figure(figsize=(12, 7))
    x = range(1, GENERATIONS+1)
    
    # GA
    plt.plot(x, hist_ga_best, 'b-o', linewidth=3, label='GA: Record (Best So Far)')
    plt.plot(x, hist_ga_avg, 'b--', alpha=0.5, label='GA: Pop Average (Learning)')
    
    # Random
    plt.plot(x, hist_rnd_best, 'r-x', linewidth=2, label='Random: Record (Luck)')
    plt.plot(x, hist_rnd_avg, 'r:', alpha=0.5, label='Random: Average (Stagnation)')
    
    plt.title("Detailed Analysis: Intelligence (GA) vs Random")
    plt.xlabel("Generations")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("result_mode1_ga_vs_random.png")
    plt.show()

def run_rl_detailed():
    print(f"\n🤖 [MODE 2] Detailed Analysis: Reinforcement Learning...")
    start = time.time()
    
    agent = QLearningAgent()
    
    hist_rl_best = []
    hist_rl_raw = [] # Raw score per episode
    global_best = float('inf')
    
    for episode in range(RL_EPISODES):
        config, indices = agent.choose_config()
        loss = evaluate_config(config)
        reward = 10 / loss  
        agent.learn(indices, reward)
        
        # Stats
        if loss < global_best: global_best = loss
        
        hist_rl_best.append(global_best)
        hist_rl_raw.append(loss)
        
        # Print every episode to show progress
        print(f"  > Episode {episode+1:03d}/{RL_EPISODES} - Loss: {loss:.4f} | Best: {global_best:.4f}")
            
    print(f"\n✅ Finished in {time.time()-start:.1f}s")

    # --- GRAPH RL ---
    plt.figure(figsize=(12, 7))
    
    # Raw Data (Light scatter to show exploration)
    plt.plot(hist_rl_raw, 'o', color='green', alpha=0.15, markersize=3, label='RL: Raw Episode Score (Exploration)')
    
    # Best So Far
    plt.plot(hist_rl_best, 'g-', linewidth=3, label='RL: Record (Best So Far)')
    
    plt.title("Reinforcement Learning Learning Dynamics")
    plt.xlabel("Episodes")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("result_mode2_rl_details.png")
    plt.show()

def run_final_battle():
    print(f"\n⚔️ [MODE 3] FINAL BATTLE: GA vs RL (Full Details)")
    start_battle = time.time()

    # --- 1. RUN GA ---
    print("1. Running Genetic Algorithm...")
    ga = GeneticAlgorithm(pop_size=POP_SIZE, mutation_rate=0.25)
    pop = ga.init_population()
    
    ga_best = []
    ga_avg = []
    glob_ga = float('inf')
    
    for gen in range(GENERATIONS):
        print(f"\n--- [GA] Generation {gen+1}/{GENERATIONS} ---")
        scores = []
        for i, ind in enumerate(pop):
            loss = evaluate_config(ind)
            scores.append(loss)
            # --- ADDED: Detailed display per individual (GA) ---
            print(f"    > [GA] Ind {i+1:02d}: Loss={loss:.4f} | LR={ind['lr']:.5f}")

        curr_min = min(scores)
        curr_avg = sum(scores)/len(scores)
        if curr_min < glob_ga: glob_ga = curr_min
        
        ga_best.append(glob_ga)
        ga_avg.append(curr_avg)
        
        print(f"  🏆 [GA] Stats Gen {gen+1}: Record={glob_ga:.4f} | Avg={curr_avg:.4f}")

        if gen < GENERATIONS - 1:
            pop = ga.evolve(pop, scores)
            
    # --- 2. RUN RL (Aggregated by "Generation" for comparison) ---
    print("\n2. Running Reinforcement Learning...")
    agent = QLearningAgent()
    
    rl_best = []
    rl_avg_batch = [] # Average of N trials to simulate a "generation"
    glob_rl = float('inf')
    
    # Split the episodes into blocks (like GA)
    for gen in range(GENERATIONS):
        print(f"\n--- [RL] Batch {gen+1}/{GENERATIONS} ---")
        batch_losses = []
        for i in range(POP_SIZE):
            cfg, idx = agent.choose_config()
            loss = evaluate_config(cfg)
            reward = 10/loss
            agent.learn(idx, reward)
            
            if loss < glob_rl: glob_rl = loss
            batch_losses.append(loss)
            
            # --- ADDED: Detailed display per trial (RL) ---
            print(f"    > [RL] Trial {i+1:02d}: Loss={loss:.4f} | Best={glob_rl:.4f}")
            
        # Save stats at end of "RL Batch"
        rl_best.append(glob_rl)
        rl_avg_batch.append(sum(batch_losses)/len(batch_losses))
        print(f"  🤖 [RL] Batch {gen+1} done. Record: {glob_rl:.4f}")

    print(f"\n✅ Battle Finished in {time.time()-start_battle:.1f}s")

    # --- 3. ULTIMATE GRAPH ---
    print("📊 Generating Master Graph...")
    plt.figure(figsize=(14, 8))
    x = range(1, GENERATIONS+1)
    
    # GA
    plt.plot(x, ga_best, 'b-o', linewidth=3, label='GA: Best So Far')
    plt.plot(x, ga_avg, 'b--', alpha=0.4, label='GA: Population Avg')
    
    # RL
    plt.plot(x, rl_best, 'g-s', linewidth=3, label='RL: Best So Far')
    plt.plot(x, rl_avg_batch, 'g:', alpha=0.4, label='RL: Batch Avg (Policy trend)')
    
    plt.title(f"FINAL COMPARISON: Evolutionary (GA) vs Sequential (RL)\nEqual Budget: {RL_EPISODES} evaluations", fontsize=14)
    plt.xlabel("Generations / Batches (Time)", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig("result_mode3_final_battle.png", dpi=300)
    plt.show()

def main():
    while True:
        print("\n" + "="*40)
        print("   AI PROJECT MENU - DATA SCIENCE")
        print("="*40)
        print("1. Analysis GA vs Random (Best & Avg)")
        print("2. Analysis RL (Best & Raw)")
        print("3. Final Comparison (GA vs RL)")
        print("4. Quit")
        
        c = input("\n👉 Choice : ")
        if c=='1': run_genetic_vs_random()
        elif c=='2': run_rl_detailed()
        elif c=='3': run_final_battle()
        elif c=='4': break
        else: print("?")

if __name__ == "__main__":
    main()