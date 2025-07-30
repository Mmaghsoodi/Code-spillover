# Spillover Analysis with LASSO-VAR and Network Visualization

This repository provides a comprehensive implementation of spillover analysis using LASSO-VAR models and network visualization. The analysis includes data preprocessing, VAR modeling, connectedness measures, network topology analysis, and interactive visualizations.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Examples](#examples)
7. [Outputs](#outputs)
8. [Troubleshooting](#troubleshooting)

## 🎯 Overview

The SpilloverAnalyzer class implements a complete pipeline for analyzing spillover effects in financial networks:

- **LASSO-VAR Modeling**: Sparse vector autoregression for efficient parameter estimation
- **Connectedness Measures**: Diebold-Yilmaz spillover measures
- **Network Analysis**: Graph-based analysis of spillover relationships
- **Visualization**: Interactive network plots and summary statistics
- **Export Capabilities**: Excel and image export functionality

## ✨ Features

### Core Functionality
- **Data Filtering**: Date window filtering with customizable parameters
- **LASSO-VAR Estimation**: Regularized vector autoregression modeling
- **Impulse Response Analysis**: Dynamic response computation
- **Connectedness Calculation**: Pairwise and directional spillover measures
- **Network Topology**: Density and efficiency metrics
- **Visualization**: Network graphs with customizable styling

### Advanced Features
- **Parameter Tuning**: Adjustable LASSO alpha and threshold parameters
- **Multiple Analysis**: Batch processing with different parameters
- **Export Options**: Excel files with multiple sheets
- **Custom Node Names**: Mapping for better visualization labels
- **Error Handling**: Robust error checking and validation

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Required packages (see requirements.txt)

### Quick Setup
```bash
# Clone or download the repository
git clone <repository-url>
cd spillover-analysis

# Install required packages
pip install pandas numpy scikit-learn networkx matplotlib seaborn

# Run the analysis
python SpilloverAnalyzer_Notebook.py
```

### Required Packages
```python
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
networkx>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📖 Usage

### Basic Usage

```python
# Import the class
from SpilloverAnalyzer_Notebook import SpilloverAnalyzer

# Define parameters
data_path = "your_financial_data.csv"
caps_csv_path = "your_capitalization_data.csv"
sector_name = "Financial_Sector"

# Node names mapping
node_names_dict = {
    'sector_11': 'Banking',
    'sector_12': 'Insurance',
    'sector_13': 'Real Estate',
    'sector_14': 'Technology',
    'sector_15': 'Energy'
}

# Create analyzer instance
analyzer = SpilloverAnalyzer(
    data_path=data_path,
    sector_name=sector_name,
    node_names_dict=node_names_dict,
    caps_csv_path=caps_csv_path,
    alpha_lasso=0.5,
    spillover_threshold=0.01,
    steps=10,
    output_path="./results"
)

# Run analysis
results = analyzer.run_analysis(
    date_at="2024-01-01",
    window_length=252,
    plot_fig=True
)
```

### Data Format Requirements

#### Financial Data CSV
```csv
date,sector_11,sector_12,sector_13,sector_14,sector_15
2020-01-01,0.001,0.002,0.003,0.001,0.002
2020-01-02,0.002,0.001,0.002,0.003,0.001
...
```

#### Capitalization Data CSV
```csv
sector,market_cap,volume
sector_11,1000,500
sector_12,800,400
sector_13,1200,600
sector_14,900,450
sector_15,1100,550
```

## 📁 File Structure

```
spillover-analysis/
├── SpilloverAnalyzer_Notebook.py    # Main analysis script
├── README.md                        # This file
├── requirements.txt                  # Package dependencies
├── sample_financial_data.csv        # Sample financial data
├── sample_capitalization_data.csv   # Sample capitalization data
└── spillover_results/               # Output directory
    ├── Financial_Sector_results.xlsx
    ├── Financial_Sector_network.png
    └── ...
```

## 📊 Examples

### Example 1: Basic Analysis
```python
# Run basic spillover analysis
results = analyzer.run_analysis()

# Access results
print(f"Total Connectedness: {results['total_connectedness']:.4f}")
print(f"Network Density: {results['network_density']:.4f}")
print(f"Global Efficiency: {results['global_efficiency']:.4f}")
```

### Example 2: Parameter Comparison
```python
# Compare different LASSO parameters
alphas = [0.1, 0.5, 1.0]
results_comparison = {}

for alpha in alphas:
    analyzer_temp = SpilloverAnalyzer(
        data_path=data_path,
        sector_name=f"Financial_Alpha_{alpha}",
        node_names_dict=node_names_dict,
        caps_csv_path=caps_csv_path,
        alpha_lasso=alpha,
        output_path=f"./results_alpha_{alpha}"
    )
    results_comparison[alpha] = analyzer_temp.run_analysis()
```

### Example 3: Threshold Analysis
```python
# Analyze different threshold levels
thresholds = [0.005, 0.01, 0.02, 0.05]
threshold_results = {}

for threshold in thresholds:
    analyzer_thresh = SpilloverAnalyzer(
        data_path=data_path,
        sector_name=f"Financial_Thresh_{threshold}",
        node_names_dict=node_names_dict,
        caps_csv_path=caps_csv_path,
        spillover_threshold=threshold,
        output_path=f"./results_thresh_{threshold}"
    )
    threshold_results[threshold] = analyzer_thresh.run_analysis()
```

## 📈 Outputs

### Excel Files
- **Connectedness_Matrix**: Full connectedness matrix
- **Directional_Table**: Directional connectedness measures
- **Net_Table**: Net pairwise spillover matrix
- **Summary**: Key metrics summary

### Visualizations
- **Network Graph**: Interactive spillover network
- **Heatmaps**: Connectedness matrix visualization
- **Bar Charts**: Directional and net connectedness
- **Summary Plots**: Key metrics comparison

### Key Metrics
- **Total Connectedness**: Overall spillover measure
- **Network Density**: Proportion of connections
- **Global Efficiency**: Information flow efficiency
- **Directional Measures**: To/from others for each sector

## 🔧 Troubleshooting

### Common Issues

#### 1. Data Loading Errors
```python
# Ensure data format is correct
# Check CSV file structure
# Verify column names match expected format
```

#### 2. Memory Issues
```python
# Reduce window_length for large datasets
# Use smaller number of steps
# Consider data sampling for very large datasets
```

#### 3. Visualization Issues
```python
# Check matplotlib backend
# Ensure sufficient memory for large networks
# Verify node names dictionary format
```

### Performance Tips

1. **Data Size**: For large datasets, consider:
   - Reducing `window_length`
   - Using smaller `steps` parameter
   - Sampling data if necessary

2. **Parameter Tuning**:
   - Start with `alpha_lasso=0.5`
   - Adjust `spillover_threshold` based on data characteristics
   - Use `steps=10` for most applications

3. **Memory Management**:
   - Close plots after saving: `plt.close()`
   - Use `plot_fig=False` for batch processing
   - Clear variables when processing multiple analyses

## 📚 Applications

### Financial Analysis
- Market spillover analysis
- Sector interconnectedness
- Risk transmission modeling
- Systemic risk assessment

### Economic Research
- Cross-country spillover effects
- Regional economic integration
- Policy transmission analysis
- Crisis propagation modeling

### Academic Research
- Time series econometrics
- Network science applications
- Financial econometrics
- Risk management studies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue on GitHub
4. Contact the maintainers

## 🔄 Version History

- **v1.0.0**: Initial release with basic functionality
- **v1.1.0**: Added advanced visualization features
- **v1.2.0**: Improved error handling and performance
- **v1.3.0**: Added batch processing capabilities

---

**Note**: This implementation is based on the Diebold-Yilmaz connectedness framework and LASSO-VAR methodology. For academic use, please cite the original papers.