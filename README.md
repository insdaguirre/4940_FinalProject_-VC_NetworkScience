# VC Investment Network Analysis

Final Project for INFO 4940: Network Science (Cornell CIS)

This repository analyzes venture capital (VC) investment networks using a bipartite graph of VCs and companies. It builds network metrics, visualizations, and portfolio analytics to understand which VCs are most connected, where they co-invest, and which industry verticals concentrate success and capital.

## Motivation

This project was completed as the final project for INFO 4940: Network Science. Our goals were to:
- Model the VC-company ecosystem as a bipartite graph
- Quantify VC connectivity, portfolio concentration, and co-investment structure
- Explore temporal investment patterns and industry dynamics
- Produce compelling visualizations and summary metrics

 ## Results (Summary)

Highlights from the analysis include:
- VCs with the highest connectivity (degree) to portfolio companies, reflecting breadth of investing.
- Co-investment structure showing clusters of VCs that frequently invest together.
- Vertical concentration patterns and relative success rates by industry.
- Temporal dynamics in deal flow and check sizes.
- Report: [INFO 4940 Final Report (PDF)](INFO%204940%20Final%20Report.pdf)
- Final slide deck: [INFO 4940 Final (PDF)](INFO%204940%20Final.pdf)

## Repository Structure

- `vc_network_analysis.py`: Main analysis script. Builds the network, computes metrics, and outputs plots.
- `Network_Analysis_VC.ipynb`: Notebook with exploratory analysis and figures.
- `outputs/`: Saved artifacts such as `company_graph.gexf`, centrality CSVs, and plots.
- Data file(s): Expected CSV input (see below). You can update the path/filename in the script as needed.

## How It Worked

The pipeline in `vc_network_analysis.py`:
1. Load and clean data: parse dates, coerce numeric fields, normalize text.
2. Build a bipartite graph (VCs ↔ companies) with edge attributes:
   - `weight` (last financing size), `verticals`, `success_probability`, `last_financing_date`.
3. Compute and visualize:
   - Network visualization of the bipartite graph (`network_visualization.png`).
   - Time-series analysis of deal counts and check sizes (`time_series_analysis.png`).
   - VC co-investment heatmap (`co_investment_patterns.png`).
   - Industry vertical success and capital concentration (`industry_success_rates.png`).
4. Parallel metrics: the dataset is chunked and analyzed in parallel; results are correctly combined across chunks so VC-level metrics reflect all connections.
5. Console summaries: top VCs by degree (connections), average success probability, and total investment.

## Data Requirements

Provide a CSV file including at least the following columns:
- `vc_name`: VC firm name
- `company_name`: Portfolio company name
- `last_financing_size`: Most recent financing size (numeric, in millions)
- `verticals`: Comma-separated vertical tags
- `success_probability`: Success probability (percent or numeric)
- `last_financing_date`: ISO-like date string (e.g., 2023-05-01)

Default expected filename in the script: `Combined VC Data - Sheet1.csv`.
