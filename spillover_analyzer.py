import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


class SpilloverAnalyzer:
    
    def __init__(self, data_path, sector_name, node_names_dict, caps_csv_path,
                 alpha_lasso=0.5, spillover_threshold=0.01, steps=10, output_path="./results"):
        
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
            # Load main data with date and stock_names columns
            self.data = pd.read_csv(self.data_path)
            
            # Load caps data with Code and size columns
            self.caps_data = pd.read_csv(self.caps_csv_path)
            
            # Validate data structure
            if 'date' not in self.data.columns:
                raise ValueError("Data must contain 'date' column")
            
            # Check if stock_names column exists, if not, assume all columns except 'date' are stock names
            if 'stock_names' not in self.data.columns:
                stock_columns = [col for col in self.data.columns if col != 'date']
                self.data = self.data[['date'] + stock_columns]
                # Rename columns to use stock names
                self.data.columns = ['date'] + [f'stock_{i}' for i in range(len(stock_columns))]
            
            # Validate caps data structure
            if 'Code' not in self.caps_data.columns or 'size' not in self.caps_data.columns:
                raise ValueError("Caps CSV must contain 'Code' and 'size' columns")
            
            print(f"Data loaded successfully: {self.data.shape}")
            print(f"Caps data loaded successfully: {self.caps_data.shape}")
            
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
        
        # Create tables with proper stock names using node_names_dict
        stock_names = list(self.node_names_dict.keys())
        if len(stock_names) != N:
            # If node_names_dict doesn't match data dimensions, use generic names
            stock_names = [f"{sector}_{i}" for i in range(N)]
        
        dir_con_table = pd.DataFrame({
            'to_others': to_others,
            'from_others': from_others,
            'Net_connectedness': net_connectedness
        })
        dir_con_table.index = stock_names
        
        net_matrix = theta_hat_matrix.T - theta_hat_matrix
        net_table = pd.DataFrame(net_matrix)
        net_table.index = stock_names
        net_table.columns = stock_names
        
        total_con = from_others.mean()
        
        connectedness_table = pd.DataFrame(theta_hat_matrix)
        connectedness_table.index = stock_names
        connectedness_table.columns = stock_names
        
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
    
    def create_size_table(self, directional_table):
        """Create size table using caps data and node_names_dict."""
        # Create a mapping from stock names to codes
        stock_to_code = {v: k for k, v in self.node_names_dict.items()}
        
        # Create size table based on directional table
        size_table = directional_table.copy()
        
        # Add size information from caps_data
        for stock_name in size_table.index:
            if stock_name in self.node_names_dict:
                code = self.node_names_dict[stock_name]
                # Find corresponding size in caps_data
                size_info = self.caps_data[self.caps_data['Code'] == code]
                if not size_info.empty:
                    size_table.loc[stock_name, 'size'] = size_info['size'].iloc[0]
                else:
                    size_table.loc[stock_name, 'size'] = 1.0  # Default size
            else:
                size_table.loc[stock_name, 'size'] = 1.0  # Default size
        
        return size_table
    
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
            G.add_node(node, size=row['size'], net=row['Net_connectedness'], display_name=display_name)
        
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
            self.data['date'] = pd.to_datetime(self.data['date'])
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
        
        # Step 5: Create size table using caps data
        size_table = self.create_size_table(directional_table)
        
        # Step 6: Create visualization
        self.plot_network_spillover(
            matrix=net_table,
            size_table=size_table,
            title=self.sector_name,
            node_names=self.node_names_dict,
            plot_fig=plot_fig,
            save_path=os.path.join(self.output_path, f"{self.sector_name}_network.png")
        )
        
        # Step 7: Save results
        results = {
            'date_calculated_on': date_calculated_on,
            'connectedness_table': connectedness_table,
            'directional_table': directional_table,
            'net_table': net_table,
            'size_table': size_table,
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
            size_table.to_excel(writer, sheet_name='Size_Table')
            
            # Summary statistics
            summary_df = pd.DataFrame({
                'Metric': ['Total Connectedness', 'Network Density', 'Global Efficiency'],
                'Value': [total_connectedness, network_density, global_efficiency]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Analysis completed! Results saved to {self.output_path}")
        return results


# Example usage function
def create_example_data():
    """Create example data files for testing."""
    # Create example data with date and stock_names
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stock_data = {
        'date': dates,
        'stock_AAPL': np.random.randn(100).cumsum(),
        'stock_MSFT': np.random.randn(100).cumsum(),
        'stock_GOOGL': np.random.randn(100).cumsum(),
        'stock_AMZN': np.random.randn(100).cumsum(),
        'stock_TSLA': np.random.randn(100).cumsum()
    }
    data_df = pd.DataFrame(stock_data)
    data_df.to_csv('example_data.csv', index=False)
    
    # Create example caps data with Code and size
    caps_data = {
        'Code': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'size': [1000, 800, 600, 400, 200]
    }
    caps_df = pd.DataFrame(caps_data)
    caps_df.to_csv('example_caps.csv', index=False)
    
    # Create node_names_dict
    node_names_dict = {
        'stock_AAPL': 'AAPL',
        'stock_MSFT': 'MSFT', 
        'stock_GOOGL': 'GOOGL',
        'stock_AMZN': 'AMZN',
        'stock_TSLA': 'TSLA'
    }
    
    return node_names_dict


if __name__ == "__main__":
    # Create example data
    node_names_dict = create_example_data()
    
    # Initialize analyzer
    analyzer = SpilloverAnalyzer(
        data_path='example_data.csv',
        sector_name='Technology',
        node_names_dict=node_names_dict,
        caps_csv_path='example_caps.csv',
        alpha_lasso=0.5,
        spillover_threshold=0.01,
        steps=10,
        output_path="./results"
    )
    
    # Run analysis
    results = analyzer.run_analysis(plot_fig=True)
    print("SpilloverAnalyzer class defined and tested successfully!")


print("SpilloverAnalyzer class defined successfully!")