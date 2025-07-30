# Spillover Analyzer

A comprehensive tool for analyzing financial spillover effects using LASSO-VAR models and network analysis.

## Overview

The `SpilloverAnalyzer` class has been updated to handle the following data structure:

### Data Structure Requirements

1. **Main Data File**: Contains columns:
   - `date`: Date column for time series analysis
   - `stock_names`: Stock name columns (or all columns except 'date' are treated as stock data)

2. **Caps CSV File**: Contains columns:
   - `Code`: Stock codes/tickers
   - `size`: Market capitalization or size metric

3. **Node Names Dictionary**: Maps stock names to codes
   - Format: `{"stock_name": "CODE"}`

## Key Changes Made

### 1. Enhanced Data Loading (`_load_data`)
- Validates presence of required columns (`date`, `Code`, `size`)
- Handles flexible stock column naming
- Provides clear error messages for missing columns

### 2. Improved Connectedness Measures (`compute_connectedness_measures`)
- Uses `node_names_dict` to properly label stock names
- Creates consistent naming across all output tables
- Handles dimension mismatches gracefully

### 3. New Size Table Creation (`create_size_table`)
- Maps stock names to codes using `node_names_dict`
- Retrieves size information from caps data
- Provides default sizes for missing mappings

### 4. Enhanced Visualization (`plot_network_spillover`)
- Uses proper stock names from `node_names_dict`
- Incorporates size information for node scaling
- Improved network visualization with proper labeling

### 5. Comprehensive Results Export
- Saves all tables to Excel with multiple sheets
- Includes size table in results
- Provides summary statistics

## Usage Example

```python
from spillover_analyzer import SpilloverAnalyzer

# Define node names mapping
node_names_dict = {
    'stock_AAPL': 'AAPL',
    'stock_MSFT': 'MSFT',
    'stock_GOOGL': 'GOOGL',
    'stock_AMZN': 'AMZN',
    'stock_TSLA': 'TSLA'
}

# Initialize analyzer
analyzer = SpilloverAnalyzer(
    data_path='your_data.csv',
    sector_name='Technology',
    node_names_dict=node_names_dict,
    caps_csv_path='your_caps.csv',
    alpha_lasso=0.5,
    spillover_threshold=0.01,
    steps=10,
    output_path="./results"
)

# Run analysis
results = analyzer.run_analysis(plot_fig=True)
```

## Data Format Examples

### Main Data File (your_data.csv)
```csv
date,stock_AAPL,stock_MSFT,stock_GOOGL,stock_AMZN,stock_TSLA
2023-01-01,150.25,280.50,90.75,320.00,180.25
2023-01-02,151.30,281.20,91.10,321.50,181.00
...
```

### Caps Data File (your_caps.csv)
```csv
Code,size
AAPL,1000
MSFT,800
GOOGL,600
AMZN,400
TSLA,200
```

## Output Files

The analyzer generates:
1. **Network visualization**: PNG file showing spillover network
2. **Excel results**: Multi-sheet Excel file with:
   - Connectedness Matrix
   - Directional Table
   - Net Table
   - Size Table
   - Summary Statistics

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Features

- **LASSO-VAR Modeling**: Robust variable selection for spillover analysis
- **Network Analysis**: Graph-based spillover network visualization
- **Size Integration**: Incorporates market cap/size information
- **Flexible Data Handling**: Supports various data formats
- **Comprehensive Output**: Multiple analysis perspectives and visualizations

## Error Handling

The updated code includes robust error handling for:
- Missing required columns
- Data format mismatches
- Dimension inconsistencies
- File loading errors

## Testing

Run the example to test the functionality:
```bash
python spillover_analyzer.py
```

This will create example data files and run a complete analysis demonstration.