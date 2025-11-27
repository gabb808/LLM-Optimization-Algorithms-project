from src.genetic import GeneticAlgorithm
from src.trainer import evaluate_config
import matplotlib.pyplot as plt
import time
import numpy as np

def main():
    print("="*60)
    print("🚀 PROJET MASTER AI : NAS (Neural Architecture Search) & GA")
    print("="*60)
    
    # --- PARAMÈTRES "EXTRÊMES" ---
    POP_SIZE = 20        # Assez grand pour la diversité
    GENERATIONS = 10     # Assez long pour la convergence
    
    # -----------------------------------------------------
    # PHASE 1 : ALGORITHME GÉNÉTIQUE (MODE PRO)
    # -----------------------------------------------------
    print(f"\n🧬 [PHASE 1] Lancement Optimisation Évolutionnaire...")
    start_time = time.time()
    
    ga = GeneticAlgorithm(pop_size=POP_SIZE, mutation_rate=0.25)
    population = ga.init_population()
    
    history_genetic_best = []
    history_genetic_avg = [] 
    
    # Variable pour stocker le RECORD ABSOLU (Best So Far)
    # Cela empêche la courbe de remonter si une génération est moins chanceuse
    global_best_loss = float('inf')
    
    for gen in range(GENERATIONS):
        print(f"\n--- Génération {gen+1}/{GENERATIONS} ---")
        scores = []
        
        for i, ind in enumerate(population):
            loss = evaluate_config(ind)
            scores.append(loss)
            # Affichage "Tech"
            print(f"  > Ind {i+1:02d}: Loss={loss:.4f} | LR={ind['lr']:.5f} | Layers={ind['n_layer']} | Embd={ind['n_embd']}")
        
        # Calculs statistiques
        current_gen_best = min(scores)
        avg_loss = sum(scores) / len(scores)
        
        # --- MODIFICATION ICI : MISE A JOUR DU RECORD GLOBAL ---
        if current_gen_best < global_best_loss:
            global_best_loss = current_gen_best
            
        # On ajoute le RECORD GLOBAL au graphique (et pas juste le meilleur du moment)
        history_genetic_best.append(global_best_loss)
        history_genetic_avg.append(avg_loss)
        
        print(f"  🏆 Stats Gen {gen+1}: Record Global={global_best_loss:.4f} | Avg Gen={avg_loss:.4f}")
        
        if gen < GENERATIONS - 1:
            population = ga.evolve(population, scores)

    print(f"⏱️ Temps Total Génétique : {time.time()-start_time:.1f}s")

    # -----------------------------------------------------
    # PHASE 2 : RANDOM SEARCH (BASELINE ROBUSTE)
    # -----------------------------------------------------
    print(f"\n🎲 [PHASE 2] Lancement Random Search (Benchmark)...")
    total_evals = POP_SIZE * GENERATIONS
    history_random = []
    best_rnd = float('inf')
    
    # On utilise le GA juste pour générer des configs random valides
    ga_rnd = GeneticAlgorithm(pop_size=1)
    
    for i in range(total_evals):
        cfg = ga_rnd.init_population()[0]
        loss = evaluate_config(cfg)
        
        if loss < best_rnd:
            best_rnd = loss
            
        # On log tous les POP_SIZE essais pour aligner les graphes
        if (i+1) % POP_SIZE == 0:
            history_random.append(best_rnd)
            print(f"  > Random Batch {(i+1)//POP_SIZE}/{GENERATIONS}: Record={best_rnd:.4f}")

    # -----------------------------------------------------
    # PHASE 3 : VISUALISATION "PAPER QUALITY"
    # -----------------------------------------------------
    print("\n📊 Génération du graphique Haute Qualité...")
    
    plt.figure(figsize=(12, 7))
    x_axis = range(1, GENERATIONS+1)
    
    # Courbe Baseline (Hasard)
    plt.plot(x_axis, history_random, label='Recherche Aléatoire (Baseline)', 
             color='red', linestyle='--', linewidth=2, marker='x')
             
    # Courbe Best So Far (Génétique) - Celle-ci sera maintenant monotone (en escalier)
    plt.plot(x_axis, history_genetic_best, label='Algorithme Génétique (Best So Far)', 
             color='blue', linewidth=3, marker='o')
             
    # Courbe Moyenne (montre la santé de la population)
    plt.plot(x_axis, history_genetic_avg, label='Algorithme Génétique (Moyenne Pop)', 
             color='green', linewidth=1, alpha=0.5, linestyle=':')

    plt.title(f"Neural Architecture Search : GA vs Random\nPop={POP_SIZE}, Gens={GENERATIONS}, MaxIter=200", fontsize=14)
    plt.xlabel("Générations (Temps d'évolution)", fontsize=12)
    plt.ylabel("Validation Loss (Log Scale)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    output_filename = "resultat_pro_monotone.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Terminé ! Graphique sauvegardé : {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()

