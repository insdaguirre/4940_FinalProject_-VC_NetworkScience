# VC Investment Network Analysis

Final Project for INFO 4940: Network Science (Cornell University)

This repository analyzes venture capital (VC) investment networks using a bipartite graph of VCs and companies. It builds network metrics, visualizations, and portfolio analytics to understand which VCs are most connected, where they co-invest, and which industry verticals concentrate success and capital.

## Motivation

This project was completed as the final project for INFO 4940: Network Science. Our goals were to:
- Model the VC-company ecosystem as a bipartite graph
- Quantify VC connectivity, portfolio concentration, and co-investment structure
- Explore temporal investment patterns and industry dynamics
- Produce compelling visualizations and summary metrics

## Repository Structure

- `vc_network_analysis.py`: Main analysis script. Builds the network, computes metrics, and outputs plots.
- `README.md`: This file.
- Data file(s): Expected CSV input (see below). You can update the path/filename in the script as needed.

## Data Requirements

Provide a CSV file including at least the following columns:
- `vc_name`: VC firm name
- `company_name`: Portfolio company name
- `last_financing_size`: Most recent financing size (numeric, in millions)
- `verticals`: Comma-separated vertical tags
- `success_probability`: Success probability (percent or numeric)
- `last_financing_date`: ISO-like date string (e.g., 2023-05-01)

Default expected filename in the script: `Combined VC Data - Sheet1.csv`.

## How It Works

At a high level, the workflow in `vc_network_analysis.py`:
1. Loads the CSV and cleans key fields (types, dates).
2. Builds a bipartite graph using NetworkX with VCs on one side and companies on the other.
3. Computes:
   - VC connectivity and portfolio stats (`analyze_vc_connections`).
   - Co-investment heatmap across VCs (`analyze_co_investments`).
   - Vertical concentration per VC (`analyze_vertical_concentration`).
   - Time-series trends of deals and check sizes (`analyze_time_series`).
4. Uses parallel processing to speed up per-chunk metrics and correctly combines results across chunks.
5. Saves multiple figures (e.g., `network_visualization.png`, `co_investment_patterns.png`, `time_series_analysis.png`, `industry_success_rates.png`).

### Outputs

- `network_visualization.png`: Bipartite layout of VCs and companies
- `co_investment_patterns.png`: VC-by-VC co-investment heatmap
- `time_series_analysis.png`: Monthly deal volume and scatter of check sizes over time
- `industry_success_rates.png`: Success and capital by vertical
- Console output: Top VCs by number of connections, average success, total investment

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, you can install the main libs directly:

```bash
pip install pandas networkx matplotlib numpy seaborn
```

## Usage

1. Place your CSV (e.g., `Combined VC Data - Sheet1.csv`) in the repo root.
2. If needed, update the filename inside `vc_network_analysis.py`.
3. Run the analysis:

```bash
python vc_network_analysis.py
```

Artifacts and images are written in the repo root.

## Report and Slides

- Report (link placeholder): [Add your report link here]
- Final slide deck (link placeholder): [Add your slide deck link here]

## Repository Link

This project is linked to GitHub: `https://github.com/insdaguirre/4940_FinalProject_-VC_NetworkScience`.

If you cloned from elsewhere, set the remote:

```bash
git remote add origin https://github.com/инствоaguirre/4940_FinalProject_-VC_NetworkScience.git
```

## License

Educational project. Add a license if you plan to distribute or reuse.
