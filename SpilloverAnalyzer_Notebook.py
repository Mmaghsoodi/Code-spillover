# Spillover Analysis with LASSO-VAR and Network Visualization
# This script provides a comprehensive implementation of spillover analysis using LASSO-VAR models and network visualization.

# =============================================================================
# 1. INSTALLATION AND IMPORTS
# =============================================================================

# Install required packages if not already installed
# !pip install pandas numpy scikit-learn networkx matplotlib seaborn

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All packages imported successfully!")

# =============================================================================
# 2. SPILLOVERANALYZER CLASS
# =============================================================================

class SpilloverAnalyzer:
    """
    Comprehensive spillover analysis class combining VAR modeling, connectedness measures, 
    and network visualization.
    """
    
    def __init__(self, data_path, sector_name, node_names_dict, caps_csv_path,
                 alpha_lasso=0.5, spillover_threshold=0.01, steps=10, output_path="./results"):
        """
        Initialize the SpilloverAnalyzer.
        
        Parameters:
            data_path (str): Path to time series data CSV
            sector_name (str): Name of the sector for analysis
            node_names_dict (dict): Mapping of node indices to display names
            caps_csv_path (str): Path to capitalization data CSV
            alpha_lasso (float): LASSO regularization parameter
            spillover_threshold (float): Threshold for network connections
            steps (int): Number of periods for impulse response
            output_path (str): Directory to save results
        """
        self.data_path = data_path
        self.sector_name = sector_name
        self.node_names_dict = node_names_dict
        self.caps_csv_path = caps_csv_path
        self.alpha_lasso = alpha_lasso
        self.spillover_threshold = spillover_threshold
        self.steps = steps
        self.output_path = output_path
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare data for analysis."""
        try:
            self.data = pd.read_csv(self.data_path)
            self.caps_data = pd.read_csv(self.caps_csv_path)
            print(f"Data loaded successfully: {self.data.shape}")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
    
    def filter_by_date_window(self, df, specific_date, window_length):
        """Filter data by date window."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        specific_date = pd.to_datetime(specific_date)
        
        mask = df['date'] <= specific_date
        result_df = (
            df.loc[mask]
            .sort_values('date', ascending=False)
            .head(window_length)
            .drop(columns=['date'])
            .reset_index(drop=True)
        )
        
        return result_df, specific_date
    
    def lasso_var_with_impulse_response(self, data, alpha=0.5, steps=10):
        """Estimate LASSO-VAR model and compute impulse responses."""
        # Prepare lagged data
        current_matrix = data.iloc[1:].to_numpy()
        lagged_matrix = data.iloc[:-1].to_numpy()
        
        N = data.shape[1]
        coef_matrix = np.zeros((N, N))
        residuals_list = []
        
        # Fit LASSO for each variable
        for i in range(N):
            y = current_matrix[:, i]
            X = lagged_matrix
            
            lasso = Lasso(alpha=alpha, fit_intercept=False)
            lasso.fit(X, y)
            
            coef_matrix[i, :] = lasso.coef_
            residuals_list.append(y - lasso.predict(X))
        
        # Compute residual covariance
        residuals = np.array(residuals_list).T
        residual_cov_matrix = np.cov(residuals, rowvar=False)
        
        # Compute impulse responses
        impulse_responses = np.zeros((steps, N, N))
        impulse_responses[0] = np.eye(N)
        
        for t in range(1, steps):
            impulse_responses[t] = coef_matrix @ impulse_responses[t - 1]
        
        return coef_matrix, residual_cov_matrix, impulse_responses
    
    def compute_connectedness_measures(self, list_coef, sigma, sector):
        """Compute connectedness measures from VAR coefficients."""
        N = sigma.shape[0]
        theta_matrix = np.zeros((N, N))
        
        # Compute pairwise connectedness
        for i in range(N):
            for j in range(N):
                sigma_inv = 1 / np.sqrt(sigma[j, j]) if sigma[j, j] != 0 else 0
                
                e_i = np.zeros(N)
                e_i[i] = 1
                e_j = np.zeros(N)
                e_j[j] = 1
                
                num = 0
                denum = 0
                
                for h in range(len(list_coef)):
                    coef_h = list_coef[h]
                    num_temp = (e_i.T @ coef_h @ sigma @ e_j) ** 2
                    denum_temp = (e_i.T @ coef_h @ sigma @ coef_h.T @ e_i)
                    
                    num += num_temp
                    denum += denum_temp
                
                theta_matrix[i, j] = sigma_inv * num / denum if denum != 0 else 0
        
        # Normalize
        row_sums = theta_matrix.sum(axis=1, keepdims=True)
        theta_hat_matrix = theta_matrix / row_sums
        
        # Compute directional measures
        to_others = theta_hat_matrix.sum(axis=0) - np.diag(theta_hat_matrix)
        from_others = theta_hat_matrix.sum(axis=1) - np.diag(theta_hat_matrix)
        net_connectedness = to_others - from_others
        
        # Create tables
        dir_con_table = pd.DataFrame({
            'to_others': to_others,
            'from_others': from_others,
            'Net_connectedness': net_connectedness
        })
        dir_con_table.index = [f"{sector}_{i}" for i in range(11, 11 + N)]
        
        net_matrix = theta_hat_matrix.T - theta_hat_matrix
        net_table = pd.DataFrame(net_matrix)
        net_table.index = [f"{sector}_{i}" for i in range(11, 11 + N)]
        net_table.columns = [f"{sector}_{j}" for j in range(11, 11 + N)]
        
        total_con = from_others.mean()
        
        connectedness_table = pd.DataFrame(theta_hat_matrix)
        connectedness_table.index = [f"{sector}_{i}" for i in range(11, 11 + N)]
        connectedness_table.columns = [f"{sector}_{j}" for j in range(11, 11 + N)]
        
        return dir_con_table, net_table, total_con, connectedness_table
    
    def analyze_network_topology(self, df, threshold):
        """Apply threshold and compute network metrics."""
        matrix = df.to_numpy()
        mask = np.abs(matrix) >= threshold
        modified_matrix = matrix * mask
        
        G = nx.from_numpy_array(modified_matrix, create_using=nx.DiGraph)
        density = nx.density(G)
        
        global_efficiency = None
        try:
            G_undirected = G.to_undirected()
            global_efficiency = nx.global_efficiency(G_undirected)
        except Exception as e:
            print(f"Warning: Could not calculate global efficiency: {e}")
        
        return modified_matrix, density, global_efficiency
    
    def normalize_size_column(self, df):
        """Normalize size column for visualization."""
        normalized_df = df.copy()
        normalized_df['size'] = np.sqrt(df['size'])
        normalized_df['size'] = normalized_df['size'] / normalized_df['size'].max()
        return normalized_df
    
    def plot_network_spillover(self, matrix, size_table, title, node_names=None,
                              plot_fig=False, figsize=(10, 8), node_scale=150,
                              arrow_width=2, intensity_factor=1, save_path=None):
        """Create network visualization."""
        normalized_sizes = self.normalize_size_column(size_table)
        
        if not np.allclose(matrix.values + matrix.values.T, 0, atol=1e-10):
            raise ValueError("Matrix must be skew-symmetric")
        
        G = nx.DiGraph()
        
        for node, row in normalized_sizes.iterrows():
            display_name = node_names.get(node, node) if node_names else node
            G.add_node(node, size=row['size'], net=row['net'], display_name=display_name)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j and matrix.iloc[i, j] > 0:
                    G.add_edge(
                        matrix.index[i],
                        matrix.columns[j],
                        weight=abs(matrix.iloc[i, j]) * intensity_factor
                    )
        
        node_sizes = [G.nodes[node]['size'] * node_scale for node in G.nodes]
        node_colors = ['red' if G.nodes[node]['net'] > 0 else 'green' for node in G.nodes]
        node_labels = {node: G.nodes[node]['display_name'] for node in G.nodes}
        
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_colors = [(0, 0, 0, weight/max_weight) for weight in edge_weights]
        
        pos = nx.circular_layout(G)
        label_pos = nx.circular_layout(G, scale=1.15)
        
        plt.figure(figsize=figsize)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              edgecolors='black', linewidths=1)
        nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=10,
                               font_color='black', font_weight='bold')
        nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color=edge_colors,
                              width=arrow_width, arrowstyle='-|>', arrowsize=15)
        
        plt.title(f"Directional Spillover Network: {title}", fontsize=14, fontweight='bold', pad=20)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Net Transmitter'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                      markersize=10, label='Net Receiver')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        if plot_fig:
            save_file = save_path or os.path.join(self.output_path, f"{title}_spillover_network.png")
            plt.savefig(save_file, dpi=300, format='png', bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def run_analysis(self, date_at=None, window_length=None, plot_fig=True):
        """
        Run complete spillover analysis.
        
        Parameters:
            date_at (str): Cutoff date for analysis
            window_length (int): Number of observations to use
            plot_fig (bool): Whether to save plots
        
        Returns:
            dict: Analysis results
        """
        print(f"Starting spillover analysis for {self.sector_name}...")
        
        # Set defaults
        if date_at is None:
            date_at = self.data['date'].max()
        if window_length is None:
            window_length = len(self.data)
        
        # Step 1: Filter data
        filtered_data, date_calculated_on = self.filter_by_date_window(
            self.data, specific_date=date_at, window_length=window_length
        )
        
        # Step 2: Estimate VAR model
        coef_matrix, residual_cov_matrix, impulse_responses = self.lasso_var_with_impulse_response(
            filtered_data, alpha=self.alpha_lasso, steps=self.steps
        )
        
        # Step 3: Compute connectedness measures
        directional_table, net_table, total_connectedness, connectedness_table = self.compute_connectedness_measures(
            [coef_matrix], residual_cov_matrix, self.sector_name
        )
        
        # Step 4: Apply threshold and compute network metrics
        thresholded_matrix, network_density, global_efficiency = self.analyze_network_topology(
            connectedness_table, self.spillover_threshold
        )
        
        # Step 5: Create visualization
        self.plot_network_spillover(
            matrix=net_table,
            size_table=directional_table,
            title=self.sector_name,
            node_names=self.node_names_dict,
            plot_fig=plot_fig,
            save_path=os.path.join(self.output_path, f"{self.sector_name}_network.png")
        )
        
        # Step 6: Save results
        results = {
            'date_calculated_on': date_calculated_on,
            'connectedness_table': connectedness_table,
            'directional_table': directional_table,
            'net_table': net_table,
            'total_connectedness': total_connectedness,
            'network_density': network_density,
            'global_efficiency': global_efficiency,
            'thresholded_matrix': thresholded_matrix
        }
        
        # Save to Excel
        excel_path = os.path.join(self.output_path, f"{self.sector_name}_results.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            connectedness_table.to_excel(writer, sheet_name='Connectedness_Matrix')
            directional_table.to_excel(writer, sheet_name='Directional_Table')
            net_table.to_excel(writer, sheet_name='Net_Table')
            
            # Summary statistics
            summary_df = pd.DataFrame({
                'Metric': ['Total Connectedness', 'Network Density', 'Global Efficiency'],
                'Value': [total_connectedness, network_density, global_efficiency]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Analysis completed! Results saved to {self.output_path}")
        return results

print("SpilloverAnalyzer class defined successfully!")

# =============================================================================
# 3. SAMPLE DATA GENERATION
# =============================================================================

def generate_sample_data(n_days=500, n_sectors=5):
    """Generate sample financial time series data."""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate correlated time series
    returns = np.random.multivariate_normal(
        mean=[0.001, 0.0008, 0.0012, 0.0009, 0.0011],
        cov=[[0.02, 0.01, 0.008, 0.012, 0.009],
              [0.01, 0.025, 0.015, 0.011, 0.013],
              [0.008, 0.015, 0.03, 0.014, 0.016],
              [0.012, 0.011, 0.014, 0.035, 0.018],
              [0.009, 0.013, 0.016, 0.018, 0.04]],
        size=n_days
    )
    
    # Create DataFrame
    data = pd.DataFrame(returns, columns=['sector_11', 'sector_12', 'sector_13', 'sector_14', 'sector_15'])
    data['date'] = dates
    
    # Generate capitalization data
    caps_data = pd.DataFrame({
        'sector': ['sector_11', 'sector_12', 'sector_13', 'sector_14', 'sector_15'],
        'market_cap': [1000, 800, 1200, 900, 1100],
        'volume': [500, 400, 600, 450, 550]
    })
    
    return data, caps_data

# Generate sample data
sample_data, sample_caps = generate_sample_data()

# Save to CSV files
sample_data.to_csv('sample_financial_data.csv', index=False)
sample_caps.to_csv('sample_capitalization_data.csv', index=False)

print("Sample data generated and saved!")
print(f"Financial data shape: {sample_data.shape}")
print(f"Capitalization data shape: {sample_caps.shape}")

# Display first few rows
print("\nSample Financial Data:")
print(sample_data.head())

print("\nSample Capitalization Data:")
print(sample_caps)

# =============================================================================
# 4. USAGE EXAMPLES
# =============================================================================

# Define parameters
data_path = "sample_financial_data.csv"
caps_csv_path = "sample_capitalization_data.csv"
sector_name = "Financial_Sector"

# Node names mapping
node_names_dict = {
    'sector_11': 'Banking',
    'sector_12': 'Insurance',
    'sector_13': 'Real Estate',
    'sector_14': 'Technology',
    'sector_15': 'Energy'
}

print("Parameters defined successfully!")

# Create analyzer instance
analyzer = SpilloverAnalyzer(
    data_path=data_path,
    sector_name=sector_name,
    node_names_dict=node_names_dict,
    caps_csv_path=caps_csv_path,
    alpha_lasso=0.5,
    spillover_threshold=0.01,
    steps=10,
    output_path="./spillover_results"
)

print("Analyzer instance created successfully!")

# Run analysis
results = analyzer.run_analysis(
    date_at="2024-01-01",
    window_length=252,  # One year of trading days
    plot_fig=True
)

print("Analysis completed successfully!")

# Display results
print("=== SPILLOVER ANALYSIS RESULTS ===")
print(f"Date calculated on: {results['date_calculated_on']}")
print(f"Total Connectedness: {results['total_connectedness']:.4f}")
print(f"Network Density: {results['network_density']:.4f}")
print(f"Global Efficiency: {results['global_efficiency']:.4f}")

print("\n=== DIRECTIONAL CONNECTEDNESS TABLE ===")
print(results['directional_table'])

print("\n=== NET SPILLOVER TABLE ===")
print(results['net_table'])

# =============================================================================
# 5. RESULTS INTERPRETATION
# =============================================================================

# Create summary plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Connectedness heatmap
sns.heatmap(results['connectedness_table'], annot=True, fmt='.3f', 
            cmap='YlOrRd', ax=axes[0,0])
axes[0,0].set_title('Connectedness Matrix', fontweight='bold')

# 2. Directional connectedness bar plot
directional_data = results['directional_table']
x = range(len(directional_data))
width = 0.35

axes[0,1].bar([i - width/2 for i in x], directional_data['to_others'], 
               width, label='To Others', color='skyblue')
axes[0,1].bar([i + width/2 for i in x], directional_data['from_others'], 
               width, label='From Others', color='lightcoral')
axes[0,1].set_xlabel('Sectors')
axes[0,1].set_ylabel('Connectedness')
axes[0,1].set_title('Directional Connectedness', fontweight='bold')
axes[0,1].legend()
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels([node_names_dict.get(idx, idx) for idx in directional_data.index], rotation=45)

# 3. Net connectedness
axes[1,0].bar(range(len(directional_data)), directional_data['Net_connectedness'], 
               color=['red' if x > 0 else 'green' for x in directional_data['Net_connectedness']])
axes[1,0].set_xlabel('Sectors')
axes[1,0].set_ylabel('Net Connectedness')
axes[1,0].set_title('Net Connectedness (Red=Transmitter, Green=Receiver)', fontweight='bold')
axes[1,0].set_xticks(range(len(directional_data)))
axes[1,0].set_xticklabels([node_names_dict.get(idx, idx) for idx in directional_data.index], rotation=45)
axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 4. Summary metrics
metrics = ['Total Connectedness', 'Network Density', 'Global Efficiency']
values = [results['total_connectedness'], results['network_density'], results['global_efficiency']]
colors = ['gold', 'lightblue', 'lightgreen']

axes[1,1].bar(metrics, values, color=colors)
axes[1,1].set_ylabel('Value')
axes[1,1].set_title('Network Summary Metrics', fontweight='bold')
for i, v in enumerate(values):
    axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# =============================================================================
# 6. ADVANCED USAGE EXAMPLES
# =============================================================================

# Example 1: Different LASSO parameters
analyzer_high_alpha = SpilloverAnalyzer(
    data_path=data_path,
    sector_name="Financial_High_Alpha",
    node_names_dict=node_names_dict,
    caps_csv_path=caps_csv_path,
    alpha_lasso=1.0,  # Higher regularization
    spillover_threshold=0.01,
    steps=10,
    output_path="./spillover_results_high_alpha"
)

results_high_alpha = analyzer_high_alpha.run_analysis(plot_fig=True)

print(f"High Alpha Total Connectedness: {results_high_alpha['total_connectedness']:.4f}")
print(f"Original Total Connectedness: {results['total_connectedness']:.4f}")

# Example 2: Different threshold levels
thresholds = [0.005, 0.01, 0.02, 0.05]
threshold_results = {}

for threshold in thresholds:
    analyzer_thresh = SpilloverAnalyzer(
        data_path=data_path,
        sector_name=f"Financial_Thresh_{threshold}",
        node_names_dict=node_names_dict,
        caps_csv_path=caps_csv_path,
        alpha_lasso=0.5,
        spillover_threshold=threshold,
        steps=10,
        output_path=f"./spillover_results_thresh_{threshold}"
    )
    
    results_thresh = analyzer_thresh.run_analysis(plot_fig=False)
    threshold_results[threshold] = results_thresh

# Plot threshold comparison
plt.figure(figsize=(10, 6))
thresholds_list = list(threshold_results.keys())
densities = [threshold_results[t]['network_density'] for t in thresholds_list]
efficiencies = [threshold_results[t]['global_efficiency'] for t in thresholds_list]

plt.subplot(1, 2, 1)
plt.plot(thresholds_list, densities, 'o-', color='blue', linewidth=2, markersize=8)
plt.xlabel('Threshold')
plt.ylabel('Network Density')
plt.title('Network Density vs Threshold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(thresholds_list, efficiencies, 'o-', color='red', linewidth=2, markersize=8)
plt.xlabel('Threshold')
plt.ylabel('Global Efficiency')
plt.title('Global Efficiency vs Threshold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 7. SUMMARY AND NEXT STEPS
# =============================================================================

print("""
=== SUMMARY ===

This script provides a comprehensive implementation of spillover analysis using:

Key Features:
1. LASSO-VAR Modeling: Sparse vector autoregression for efficient parameter estimation
2. Connectedness Measures: Diebold-Yilmaz spillover measures
3. Network Analysis: Graph-based analysis of spillover relationships
4. Visualization: Interactive network plots and summary statistics
5. Export Capabilities: Excel and image export functionality

Applications:
- Financial market spillover analysis
- Economic sector interconnectedness
- Risk transmission modeling
- Systemic risk assessment

Next Steps:
1. Real Data Integration: Replace sample data with actual financial time series
2. Parameter Tuning: Optimize LASSO alpha and threshold parameters
3. Time-varying Analysis: Implement rolling window analysis
4. Additional Metrics: Add more network centrality measures
5. Interactive Dashboards: Create web-based visualization tools

Files Generated:
- sample_financial_data.csv: Sample time series data
- sample_capitalization_data.csv: Market capitalization data
- ./spillover_results/: Analysis results and visualizations

The analysis provides insights into how shocks propagate through financial networks and helps identify systemically important sectors.
""")