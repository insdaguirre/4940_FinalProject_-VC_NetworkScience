"""
company_board_overlap.py
========================
Builds a company–to–company "board–overlap" graph from a
VC–portfolio DataFrame and studies whether a firm's network
position predicts its success_probability.

This version is adapted to work with the VC data repository and
integrates with the existing network analysis.
"""

import os
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import multiprocessing as mp
from functools import partial

# Configuration
INFILE = "Combined VC Data - Sheet1.csv"
OUTDIR = "outputs"
PLOTDIR = os.path.join(OUTDIR, "plots")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PLOTDIR, exist_ok=True)

def clean_success_probability(value):
    """Clean success probability values"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove % and convert to float
        value = value.strip('%')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return float(value)

def create_bipartite_network(df):
    """Create a bipartite network from the VC-Company data"""
    B = nx.Graph()
    
    # Add nodes with their types
    vcs = df['vc_name'].unique()
    companies = df['company_name'].unique()
    
    # Add nodes with attributes
    for _, row in df.iterrows():
        vc = row['vc_name']
        co = row['company_name']
        
        B.add_node(vc, bipartite="vc")
        B.add_node(co, bipartite="co",
                  success=clean_success_probability(row['success_probability']),
                  employees=row['employees'],
                  city=row['hq_city'],
                  verticals=row['verticals'],
                  total_raised=float(row['total_raised']),
                  last_financing_size=float(row['last_financing_size']))
        
        # Edge weight based on last financing size
        B.add_edge(vc, co, weight=float(row['last_financing_size']))
    
    return B

def analyze_company_network(G):
    """Analyze the company projection network"""
    # Get company nodes
    company_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == "co"}
    
    # Create company projection with weighted edges
    company_G = nx.bipartite.weighted_projected_graph(G, company_nodes)
    
    # Compute centrality measures
    degree_w = dict(company_G.degree(weight="weight"))
    deg_cent = nx.degree_centrality(company_G)
    bet_cent = nx.betweenness_centrality(company_G, weight="weight", normalized=True)
    eig_cent = nx.eigenvector_centrality(company_G, weight="weight", max_iter=1000)
    clust_coef = nx.clustering(company_G, weight="weight")
    
    # Create centrality metrics DataFrame
    cent_df = pd.DataFrame({
        "company_name": list(company_G.nodes()),
        "deg_weighted": pd.Series(degree_w),
        "deg_centrality": pd.Series(deg_cent),
        "betweenness": pd.Series(bet_cent),
        "eigenvector": pd.Series(eig_cent),
        "clustering": pd.Series(clust_coef)
    })
    
    # Add company attributes
    company_attrs = {n: d for n, d in G.nodes(data=True) if d["bipartite"] == "co"}
    attrs_df = pd.DataFrame.from_dict(company_attrs, orient='index')
    attrs_df.index.name = 'company_name'
    attrs_df.reset_index(inplace=True)
    
    cent_df = cent_df.merge(attrs_df[['company_name', 'success', 'total_raised', 'employees']],
                           on='company_name', how='left')
    
    return company_G, cent_df

def visualize_network(G, output_file='company_network.png'):
    """Create and save a network visualization"""
    plt.figure(figsize=(20, 20))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color='blue',
                          node_size=200, alpha=0.5)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Add labels for nodes
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Company Network (Based on Common VC Investors)", fontsize=16)
    plt.axis('off')
    plt.savefig(os.path.join(PLOTDIR, output_file), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_correlations(cent_df):
    """Analyze correlations between network metrics and success"""
    # Drop rows with NaN values in success
    cent_df = cent_df.dropna(subset=['success'])
    
    metrics = ['deg_weighted', 'deg_centrality', 'betweenness', 
              'eigenvector', 'clustering']
    
    results = []
    for metric in metrics:
        # Drop rows with NaN values in the current metric
        valid_data = cent_df[[metric, 'success']].dropna()
        if len(valid_data) > 0:
            corr, pval = pearsonr(valid_data[metric], valid_data['success'])
            results.append({
                'metric': metric,
                'correlation': corr,
                'p_value': pval
            })
        else:
            results.append({
                'metric': metric,
                'correlation': np.nan,
                'p_value': np.nan
            })
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = cent_df[metrics + ['success']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Network Metrics and Success')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTDIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    return pd.DataFrame(results)

def main():
    # Read and clean data
    print("Loading and cleaning data...")
    df = pd.read_csv(INFILE)
    
    # Clean the data
    df['last_financing_size'] = df['last_financing_size'].astype(float)
    df['success_probability'] = df['success_probability'].apply(clean_success_probability)
    df['verticals'] = df['verticals'].astype(str)
    
    # Create bipartite network
    print("Creating bipartite network...")
    B = create_bipartite_network(df)
    
    # Analyze company network
    print("Analyzing company network...")
    company_G, cent_df = analyze_company_network(B)
    
    # Save network for Gephi
    print("Saving network for Gephi...")
    nx.write_gexf(company_G, os.path.join(OUTDIR, "company_graph.gexf"))
    
    # Save centrality metrics
    print("Saving centrality metrics...")
    cent_df.to_csv(os.path.join(OUTDIR, "centrality_metrics.csv"), index=False)
    
    # Visualize network
    print("Creating network visualization...")
    visualize_network(company_G)
    
    # Analyze correlations
    print("Analyzing correlations...")
    corr_results = analyze_correlations(cent_df)
    
    # Print results
    print("\nCorrelations between network metrics and success:")
    print(corr_results.to_string(index=False))
    
    print("\nNetwork Statistics:")
    print(f"Number of companies: {company_G.number_of_nodes()}")
    print(f"Number of edges: {company_G.number_of_edges()}")
    print(f"Average clustering coefficient: {nx.average_clustering(company_G, weight='weight'):.3f}")
    print(f"Network density: {nx.density(company_G):.3f}")
    
    # Print data quality metrics
    print("\nData Quality Metrics:")
    print(f"Number of companies with success data: {cent_df['success'].notna().sum()}")
    print(f"Average success probability: {cent_df['success'].mean():.2f}%")
    print(f"Success probability range: {cent_df['success'].min():.2f}% - {cent_df['success'].max():.2f}%")

if __name__ == "__main__":
    main()
