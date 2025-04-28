import pandas as pd
import json
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import approximation as approx
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
import powerlaw

# ─── Part 1: Global Graph Metrics ──────────────────────────────────────────────

# Step 1: Load & clean CSV
df = pd.read_csv('data.csv', sep=',(?=\S)', engine='python')

def delete_quotes(x):
    return x[1:-1] if isinstance(x, str) and len(x) > 1 else x

for col in ["id", "screenName", "avatar", "lang", "tweetId"]:
    df[col] = df[col].apply(delete_quotes)

df["friends"] = df["friends"].apply(lambda x: json.loads(x) if pd.notna(x) else [])

# Step 2: Build directed graph
G = nx.DiGraph()
root_users = set(df['id'].astype(str))
for _, row in df.iterrows():
    u = str(row['id'])
    G.add_node(u)
    for f in row['friends']:
        f = str(f)
        if f in root_users:
            G.add_edge(u, f)

# Step 3: Undirected version & connected components
G_und = G.to_undirected()
comps = list(nx.connected_components(G_und))
num_components = len(comps)
lcc_nodes = max(comps, key=len)
LCC = G_und.subgraph(lcc_nodes).copy()

# Step 4: Approximate diameter (2-sweep) on the LCC
diameter = approx.diameter(LCC, seed=42)

# Step 5: Approximate average local clustering (Monte Carlo) on full graph
approx_avg_clust = average_clustering(G_und, trials=1000, seed=42)

# Step 6: Approximate average shortest-path length on the LCC via sampling
k = min(100, len(lcc_nodes))
random.seed(42)
sources = random.sample(list(lcc_nodes), k)
total_d, total_pairs = 0, 0
for s in sources:
    dist = nx.single_source_shortest_path_length(LCC, s)
    total_d += sum(dist.values())
    total_pairs += (len(dist) - 1)
approx_avg_path = total_d / total_pairs

# Print Part 1 results
print("\n=== Part 1: Global Network Metrics ===")
print(f"{'Metric':<35} Value")
print(f"{'-'*50}")
print(f"{'Connected components':<35} {num_components}")
print(f"{'Approx. diameter (LCC)':<35} {diameter}")
print(f"{'Approx. avg. path length (LCC)':<35} {approx_avg_path:.4f}")
print(f"{'Approx. avg. clustering coef.':<35} {approx_avg_clust:.4f}")


# ─── Part 2: Centrality Distributions ──────────────────────────────────────────

# 2a. Exact degree centrality on the LCC
n = LCC.number_of_nodes()
deg_vals = [deg / (n - 1) for _, deg in LCC.degree()]

# 2b. Landmark-based approximate closeness centrality on the LCC
def approx_closeness_landmarks(G, k=30, seed=42):
    random.seed(seed)
    landmarks = random.sample(list(G.nodes()), k)
    dist_sum = {v: 0 for v in G.nodes()}
    for lm in landmarks:
        lengths = nx.single_source_shortest_path_length(G, lm)
        for v, d in lengths.items():
            dist_sum[v] += d
    N = G.number_of_nodes()
    closeness = {}
    for v, total in dist_sum.items():
        closeness[v] = (N - 1) * (k / total) if total > 0 else 0.0
    return closeness

cl_dict = approx_closeness_landmarks(LCC, k=30, seed=42)
close_vals = list(cl_dict.values())

# Plot Part 2
plt.figure(figsize=(8,5))
plt.hist(deg_vals, bins=50, edgecolor='black')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.title('Degree Centrality Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(close_vals, bins=50, edgecolor='black')
plt.xlabel('Approx. Closeness Centrality')
plt.ylabel('Frequency')
plt.title('Closeness Centrality Distribution (Landmarks)')
plt.tight_layout()
plt.show()


# ─── Part 3: Power-Law Fit with 90% CI ─────────────────────────────────────────

def plot_powerlaw_ci(data, label, ax, bs=200, seed=42):
    """Fit a power-law, bootstrap alpha for 90% CI, and plot CCDF + fit + CI."""
    np.random.seed(seed)
    fit = powerlaw.Fit(data, discrete=False, verbose=False)
    alpha = fit.power_law.alpha
    x_min = fit.power_law.xmin

    filtered = np.array([x for x in data if x >= x_min])
    n = len(filtered)
    alpha_samples = []
    for _ in range(bs):
        samp = np.random.choice(filtered, size=n, replace=True)
        alpha_samples.append(1 + n / np.sum(np.log(samp / x_min)))
    alpha_low, alpha_up = np.percentile(alpha_samples, [5, 95])

    # Empirical CCDF
    fit.plot_ccdf(ax=ax, label=f"{label} empirical", marker='o', markersize=4)

    # Theoretical CCDF curves
    xs = np.linspace(x_min, max(data), 200)
    emp_x, emp_ccdf = fit.ccdf()
    emp0 = next(y for x, y in zip(emp_x, emp_ccdf) if x >= x_min)

    ccdf_hat = emp0 * (xs / x_min) ** (-(alpha - 1))
    ccdf_low = emp0 * (xs / x_min) ** (-(alpha_low - 1))
    ccdf_up  = emp0 * (xs / x_min) ** (-(alpha_up  - 1))

    ax.plot(xs, ccdf_hat, color='r', linestyle='--', label=f"{label} fit alpha={alpha:.2f}")
    ax.plot(xs, ccdf_low, color='gray', linestyle=':', label="90% CI")
    ax.plot(xs, ccdf_up,  color='gray', linestyle=':')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(label)
    ax.set_ylabel('CCDF')
    ax.legend()
    return alpha, alpha_low, alpha_up

# Perform fits & plotting
fig, ax = plt.subplots(figsize=(8,6))
alpha_deg, alpha_deg_low, alpha_deg_high = plot_powerlaw_ci(deg_vals,   'Degree centrality',    ax)
alpha_clo, alpha_clo_low, alpha_clo_high = plot_powerlaw_ci(close_vals, 'Closeness centrality', ax)
plt.title('Power-Law Fit + 90% Confidence Bounds')
plt.tight_layout()
plt.show()

# Print Part 3 results without non-ASCII characters
print("\n=== Part 3: Power-Law Exponents & 90% CI ===")
print(f"Degree centrality:    alpha = {alpha_deg:.2f}, 90% CI = [{alpha_deg_low:.2f}, {alpha_deg_high:.2f}]")
print(f"Closeness centrality: alpha = {alpha_clo:.2f}, 90% CI = [{alpha_clo_low:.2f}, {alpha_clo_high:.2f}]")
