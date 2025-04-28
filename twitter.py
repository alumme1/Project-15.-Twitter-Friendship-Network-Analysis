import csv
import re
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
import powerlaw
import numpy as np
from itertools import combinations
import random

# Load graph from CSV
def load_graph(csv_file='data.csv'):
    # collect user IDs
    id_set = set()
    with open(csv_file, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_set.add(row['id'])
    # build edges among known users
    edges = []
    with open(csv_file, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['id']
            friends = re.findall(r"\d+", row['friends'])
            for tgt in friends:
                if tgt in id_set:
                    edges.append((src, tgt))
    G = nx.DiGraph()
    G.add_nodes_from(id_set)
    G.add_edges_from(edges)
    return G

# 1) Compute and display metrics
def compute_metrics(G):
    G_und = G.to_undirected()
    avg_clust = nx.average_clustering(G_und)
    if nx.is_connected(G_und):
        sub = G_und
    else:
        comp_nodes = max(nx.connected_components(G_und), key=len)
        sub = G_und.subgraph(comp_nodes)
    diam = nx.diameter(sub)
    avg_path = nx.average_shortest_path_length(sub)
    metrics = {
        'Total Nodes': G.number_of_nodes(),
        'Total Edges': G.number_of_edges(),
        'Diameter': diam,
        'Average Clustering': avg_clust,
        'Average Path Length': avg_path,
        'Strongly Connected Components': nx.number_strongly_connected_components(G),
        'Weakly Connected Components': nx.number_weakly_connected_components(G)
    }
    print(tabulate(metrics.items(), headers=['Metric','Value'], tablefmt='github'))

# 2) Plot degree and closeness centrality
def plot_centrality_distributions(G, bins=50):
    deg_cent = nx.degree_centrality(G)
    clo_cent = nx.closeness_centrality(G)
    plt.figure()
    plt.hist(list(deg_cent.values()), bins=bins, log=True)
    plt.title('Degree Centrality Distribution (log y)')
    plt.xlabel('Degree Centrality'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()
    plt.figure()
    plt.hist(list(clo_cent.values()), bins=bins, log=True)
    plt.title('Closeness Centrality Distribution (log y)')
    plt.xlabel('Closeness Centrality'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()
    degrees = [d for _, d in G.degree()]
    min_deg = max(1, min(degrees)); max_deg = max(degrees)
    log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bins)
    plt.figure()
    plt.hist(degrees, bins=log_bins)
    plt.xscale('log'); plt.yscale('log')
    plt.title('Raw Degree Distribution (log-log)')
    plt.xlabel('Degree'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()

# 3) Fit power law to distributions

def fit_powerlaw_with_ci(data, label):
    try:
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
    except Exception as e:
        print(f"Skipping {label} fit: {e}"); return
    alpha, sigma, xmin = fit.alpha, fit.sigma, fit.xmin
    plt.figure(); fit.plot_ccdf(label='Empirical'); fit.power_law.plot_ccdf(label=f'Fit (alpha={alpha:.2f})')
    if not np.isnan(sigma) and sigma > 0:
        lo, hi = alpha - 1.64*sigma, alpha + 1.64*sigma
        xs = np.linspace(xmin, max(data), 100)
        ccdf = lambda x,a: (x/xmin)**(-(a-1))
        plt.plot(xs, ccdf(xs, lo), 'k--', label='90% CI'); plt.plot(xs, ccdf(xs, hi), 'k--')
    plt.title(f'{label} CCDF & Power-Law Fit'); plt.legend(); plt.tight_layout(); plt.show()
    try:
        R, p = fit.distribution_compare('power_law', 'lognormal')
        print(f"{label}: alpha={alpha:.2f}, sigma={sigma:.2f}, R={R:.2f}, p={p:.4f}")
        print("-> power law is plausible." if p > 0.1 else "-> power law is not better than lognormal.")
    except:
        print(f"Could not compare {label} distributions.")

# 4) Structural equivalence by raw degree

def structural_equivalence_by_degree(G, max_pairs=10000):
    G_und = G.to_undirected()
    buckets = {}
    for node, deg in G_und.degree():
        if deg > 0:
            buckets.setdefault(deg, []).append(node)
    se_scores = {}
    for deg, nodes in sorted(buckets.items()):
        n = len(nodes)
        if n < 2:
            se_scores[deg] = 0.0
            continue
        total_pairs = n * (n - 1) // 2
        if total_pairs <= max_pairs:
            pairs = list(combinations(nodes, 2))
        else:
            pairs = set()
            while len(pairs) < max_pairs:
                u, v = random.sample(nodes, 2)
                pairs.add(tuple(sorted((u, v))))
            pairs = list(pairs)
        sims = []
        for u, v in pairs:
            Nu = set(G_und.neighbors(u))
            Nv = set(G_und.neighbors(v))
            union = Nu | Nv
            if union:
                sims.append(len(Nu & Nv) / len(union))
        se_scores[deg] = sum(sims) / len(sims) if sims else 0.0
    return se_scores

if __name__ == '__main__':
    G = load_graph('data.csv')
    compute_metrics(G)
    plot_centrality_distributions(G)
    deg_vals = [d for _, d in G.degree()]
    clo_vals = [v for v in nx.closeness_centrality(G).values() if v > 0]
    fit_powerlaw_with_ci(deg_vals, 'Raw Degree')
    fit_powerlaw_with_ci(clo_vals, 'Nonzero Closeness')
    se = structural_equivalence_by_degree(G)
    print("\n4) Structural Equivalence (avg Jaccard per raw degree):")
    for deg, score in se.items():
        print(f"  degree={deg}: avg_Jaccard={score:.4f}")
