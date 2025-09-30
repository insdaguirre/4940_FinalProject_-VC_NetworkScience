import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

def create_bipartite_network(df):
    """Create a bipartite network from the VC-Company data"""
    G = nx.Graph()
    
    # Add nodes with their types
    vcs = df['vc_name'].unique()
    companies = df['company_name'].unique()
    
    G.add_nodes_from(vcs, bipartite=0)  # VC nodes
    G.add_nodes_from(companies, bipartite=1)  # Company nodes
    
    # Add edges with weights
    for _, row in df.iterrows():
        G.add_edge(row['vc_name'], row['company_name'], 
                  weight=float(row['last_financing_size']),
                  verticals=str(row['verticals']),  # Ensure verticals is a string
                  success_probability=float(row['success_probability']),
                  last_financing_date=row['last_financing_date'])
    
    return G

def visualize_network(G, output_file='network_visualization.png'):
    """Create and save a network visualization"""
    plt.figure(figsize=(20, 20))
    
    # Separate VC and company nodes
    vc_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    company_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, nodelist=vc_nodes, node_color='red', 
                          node_size=500, alpha=0.8, label='VCs')
    nx.draw_networkx_nodes(G, pos, nodelist=company_nodes, node_color='blue',
                          node_size=200, alpha=0.5, label='Companies')
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Add labels for VC nodes only
    vc_labels = {node: node for node in vc_nodes}
    nx.draw_networkx_labels(G, pos, labels=vc_labels, font_size=8)
    
    plt.title("VC Investment Network", fontsize=16)
    plt.legend(fontsize=12)
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_time_series(df):
    """Analyze investment patterns over time"""
    # Convert date string to datetime
    df['last_financing_date'] = pd.to_datetime(df['last_financing_date'])
    
    # Create time series analysis
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    
    # Plot 1: Investment amounts over time by top VCs
    ax1 = fig.add_subplot(gs[0])
    top_vcs = df['vc_name'].value_counts().head(5).index
    for vc in top_vcs:
        vc_data = df[df['vc_name'] == vc]
        ax1.scatter(vc_data['last_financing_date'], vc_data['last_financing_size'],
                   label=vc, alpha=0.6)
    
    ax1.set_ylabel('Investment Size (M$)')
    ax1.set_title('Investment Patterns Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of deals over time (monthly)
    ax2 = fig.add_subplot(gs[1])
    monthly_deals = df.resample('M', on='last_financing_date').size()
    ax2.plot(monthly_deals.index, monthly_deals.values, color='black')
    ax2.set_ylabel('Number of Deals')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_co_investments(G):
    """Analyze co-investment patterns between VCs"""
    vc_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    co_investments = np.zeros((len(vc_nodes), len(vc_nodes)))
    
    # Create VC index mapping
    vc_to_idx = {vc: idx for idx, vc in enumerate(vc_nodes)}
    
    # Count co-investments
    for company in [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]:
        investors = list(G.neighbors(company))
        for i in range(len(investors)):
            for j in range(i+1, len(investors)):
                idx1, idx2 = vc_to_idx[investors[i]], vc_to_idx[investors[j]]
                co_investments[idx1][idx2] += 1
                co_investments[idx2][idx1] += 1
    
    # Visualize co-investments
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_investments, xticklabels=vc_nodes, yticklabels=vc_nodes,
                cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('VC Co-Investment Patterns')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('co_investment_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return co_investments, vc_nodes

def analyze_industry_success(G):
    """Analyze success rates by industry vertical"""
    vertical_metrics = defaultdict(lambda: {'count': 0, 'success_prob': [], 'investment_size': []})
    
    # Collect data for each vertical
    for _, company in G.edges():
        edge_data = G.edges[_, company]
        verticals = edge_data['verticals'].split(', ')
        for vertical in verticals:
            if vertical and vertical.lower() != 'nan':
                vertical_metrics[vertical]['count'] += 1
                vertical_metrics[vertical]['success_prob'].append(edge_data['success_probability'])
                vertical_metrics[vertical]['investment_size'].append(edge_data['weight'])
    
    # Calculate average metrics
    vertical_summary = {}
    for vertical, metrics in vertical_metrics.items():
        if metrics['count'] >= 5:  # Only include verticals with sufficient data
            vertical_summary[vertical] = {
                'count': metrics['count'],
                'avg_success': np.mean(metrics['success_prob']),
                'total_investment': np.sum(metrics['investment_size']),
                'avg_investment': np.mean(metrics['investment_size'])
            }
    
    # Visualize industry success rates
    verticals = list(vertical_summary.keys())
    success_rates = [v['avg_success'] for v in vertical_summary.values()]
    investments = [v['total_investment'] for v in vertical_summary.values()]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Success rates by vertical
    bars = ax1.bar(verticals, success_rates)
    ax1.set_xticklabels(verticals, rotation=45, ha='right')
    ax1.set_ylabel('Average Success Probability (%)')
    ax1.set_title('Success Rates by Industry Vertical')
    
    # Total investment by vertical
    bars = ax2.bar(verticals, investments)
    ax2.set_xticklabels(verticals, rotation=45, ha='right')
    ax2.set_ylabel('Total Investment (M$)')
    ax2.set_title('Total Investment by Industry Vertical')
    
    plt.tight_layout()
    plt.savefig('industry_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return vertical_summary

def analyze_vc_connections(G, top_n=10):
    """Analyze which VCs are most connected to attractive companies"""
    vc_metrics = {}
    for vc in [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]:
        # Get connected companies
        companies = list(G.neighbors(vc))
        
        # Calculate average success probability
        avg_success = np.mean([G[vc][company]['success_probability'] for company in companies])
        
        # Calculate total investment
        total_investment = sum(G[vc][company]['weight'] for company in companies)
        
        vc_metrics[vc] = {
            'degree': len(companies),
            'avg_success': avg_success,
            'total_investment': total_investment
        }
    
    return vc_metrics

def analyze_vertical_concentration(G):
    """Analyze vertical concentration in VC portfolios"""
    # Use regular dictionaries instead of defaultdict
    vc_verticals = {}
    
    for vc in [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]:
        vc_verticals[vc] = {}
        for company in G.neighbors(vc):
            verticals = G[vc][company]['verticals'].split(', ')
            for vertical in verticals:
                if vertical:  # Only count non-empty verticals
                    vc_verticals[vc][vertical] = vc_verticals[vc].get(vertical, 0) + 1
    
    return vc_verticals

def calculate_vc_exclusivity(G):
    """Calculate how exclusive each VC is (average company degree)"""
    vc_exclusivity = {}
    for vc in [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]:
        companies = list(G.neighbors(vc))
        avg_company_degree = np.mean([G.degree(company) for company in companies])
        vc_exclusivity[vc] = avg_company_degree
    
    return vc_exclusivity

def parallel_analysis(df_chunk):
    """Function to run in parallel for each chunk of data"""
    G = create_bipartite_network(df_chunk)
    vc_metrics = analyze_vc_connections(G)
    vertical_concentration = analyze_vertical_concentration(G)
    vc_exclusivity = calculate_vc_exclusivity(G)
    return vc_metrics, vertical_concentration, vc_exclusivity

def combine_vertical_concentrations(all_vertical_concentration, vertical_concentration):
    """Helper function to combine vertical concentration results"""
    for vc, verticals in vertical_concentration.items():
        if vc not in all_vertical_concentration:
            all_vertical_concentration[vc] = {}
        for vertical, count in verticals.items():
            all_vertical_concentration[vc][vertical] = all_vertical_concentration[vc].get(vertical, 0) + count
    return all_vertical_concentration

def combine_vc_metrics(all_vc_metrics, vc_metrics):
    """Helper function to combine VC metrics results"""
    for vc, metrics in vc_metrics.items():
        if vc not in all_vc_metrics:
            all_vc_metrics[vc] = {
                'degree': 0,
                'avg_success': 0,
                'total_investment': 0,
                'count': 0
            }
        all_vc_metrics[vc]['degree'] += metrics['degree']
        all_vc_metrics[vc]['total_investment'] += metrics['total_investment']
        all_vc_metrics[vc]['avg_success'] = (all_vc_metrics[vc]['avg_success'] * all_vc_metrics[vc]['count'] + 
                                           metrics['avg_success'] * metrics['degree']) / (all_vc_metrics[vc]['count'] + metrics['degree'])
        all_vc_metrics[vc]['count'] += metrics['degree']
    return all_vc_metrics

def main():
    # Set style for visualizations
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Read the data
    df = pd.read_csv("Combined VC Data - Sheet1.csv")
    
    # Clean the data
    df['last_financing_size'] = df['last_financing_size'].astype(float)
    if df['success_probability'].dtype == 'object':
        df['success_probability'] = df['success_probability'].str.strip('%').astype(float)
    else:
        df['success_probability'] = df['success_probability'].astype(float)
    df['verticals'] = df['verticals'].astype(str)
    
    # Create full network for visualizations
    print("Creating network visualization...")
    G = create_bipartite_network(df)
    visualize_network(G)
    
    print("Analyzing time series patterns...")
    analyze_time_series(df)
    
    print("Analyzing co-investment patterns...")
    co_investments, vc_nodes = analyze_co_investments(G)
    
    print("Analyzing industry success rates...")
    vertical_summary = analyze_industry_success(G)
    
    # Split data for parallel processing of basic metrics
    num_cores = 6
    df_chunks = np.array_split(df, num_cores)
    
    print("Running parallel analysis of basic metrics...")
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(parallel_analysis, df_chunks)
    
    # Combine results
    all_vc_metrics = {}
    all_vertical_concentration = {}
    all_vc_exclusivity = {}
    
    for vc_metrics, vertical_concentration, vc_exclusivity in results:
        all_vc_metrics = combine_vc_metrics(all_vc_metrics, vc_metrics)
        all_vertical_concentration = combine_vertical_concentrations(all_vertical_concentration, vertical_concentration)
        all_vc_exclusivity.update(vc_exclusivity)
    
    # Print results
    print("\nTop VCs by number of connections:")
    sorted_vcs = sorted(all_vc_metrics.items(), key=lambda x: x[1]['degree'], reverse=True)
    for vc, metrics in sorted_vcs[:10]:
        print(f"{vc}: {metrics['degree']} connections, Avg Success: {metrics['avg_success']:.2f}%, Total Investment: ${metrics['total_investment']:.2f}M")
    
    print("\nVertical concentration in VC portfolios:")
    for vc, verticals in all_vertical_concentration.items():
        print(f"\n{vc}:")
        sorted_verticals = sorted(verticals.items(), key=lambda x: x[1], reverse=True)
        for vertical, count in sorted_verticals:
            print(f"  {vertical}: {count} companies")
    
    print("\nVC Exclusivity (average company degree):")
    sorted_exclusivity = sorted(all_vc_exclusivity.items(), key=lambda x: x[1])
    for vc, exclusivity in sorted_exclusivity[:10]:
        print(f"{vc}: {exclusivity:.2f}")
    
    print("\nIndustry Success Rates:")
    sorted_verticals = sorted(vertical_summary.items(), key=lambda x: x[1]['avg_success'], reverse=True)
    for vertical, metrics in sorted_verticals:
        print(f"{vertical}:")
        print(f"  Average Success Rate: {metrics['avg_success']:.2f}%")
        print(f"  Total Investment: ${metrics['total_investment']:.2f}M")
        print(f"  Average Investment: ${metrics['avg_investment']:.2f}M")
        print(f"  Number of Companies: {metrics['count']}")

if __name__ == "__main__":
    main() 