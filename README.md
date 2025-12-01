
# TravelTide: Data Analysis and User Segmentation Project

## Overview
TravelTide is a data analysis project focused on understanding user behavior, segmenting users, and assigning perks based on their interactions with travel services. This project includes data processing, exploratory data analysis (EDA), clustering, and visualization to derive actionable insights.

## Project Structure
```
/
├── core/                  # Core Python modules for data processing and analysis
│   ├── __init__.py
│   ├── advance_metrics.py
│   ├── eda.py
│   ├── load_data.py
│   ├── perk_assignment.py
│   ├── segment_analyse.py
│   ├── utils.py
│   └── visualization.py
├── data/                  # Data storage
│   ├── processed/          # Processed data files
│   │   ├── feature_metrics/
│   │   │   ├── final_user_table.csv
│   │   │   └── user_base.csv
│   │   ├── kmean/
│   │   │   └── user_segment.csv
│   │   ├── non_ml/
│   │   │   ├── analyse/
│   │   │   │   ├── business_impact_summary.csv
│   │   │   │   ├── feature_importance.csv
│   │   │   │   └── statistical_tests.csv
│   │   │   └── customer_segment.csv
│   │   ├── pca/
│   │   │   └── user_pca.csv
│   │   ├── session_base.csv
│   │   └── sessions_cleaned.csv
│   └── raw/                # Raw data files
│       ├── flights.csv
│       ├── hotels.csv
│       ├── sessions.csv
│       └── users.csv
├── notebooks/              # Jupyter notebooks for analysis
│   ├── analysis_plots/
│   ├── cluster_analyse.ipynb
│   ├── cluster_and_perks_assign.ipynb
│   ├── eda.ipynb
│   ├── kmean_cluster.ipynb
│   ├── load_data.ipynb
│   ├── manual_perk_assignment.ipynb
│   ├── pca_processing.ipynb
│   ├── processing.ipynb
│   ├── segment_analyse.ipynb
│   ├── user_features.ipynb
│   └── user_metrics.ipynb
├── reports/               # Reports and visualizations
│   ├── eda/
│   │   ├── results/
│   │   └── viz/
│   │       ├── altersverteilung_der_nutzer.png
│   │       ├── booking_discounts.png
│   │       ├── dashboard/
│   │       ├── demographic_summary.png
│   │       ├── outliers/
│   │       ├── page_clicks_outlier_comparison_4panel.png
│   │       ├── session_distributions.png
│   │       ├── session_duration_outlier_comparison_4panel.png
│   │       ├── session_relationships.png
│   │       └── verteilung_der_nutzer_geburtsjahre.png
│   └── viz/
├── sql/                   # SQL scripts
│   └── session_base.sql
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Key Features
- **Data Processing:** Cleaning and transforming raw data into structured formats.
- **Exploratory Data Analysis (EDA):** Understanding user behavior and identifying patterns.
- **User Segmentation:** Clustering users based on their interactions and features.
- **Perk Assignment:** Assigning perks to users based on their segments.
- **Visualization:** Creating insightful visualizations for reports and presentations.

## Core Modules
- **`load_data.py`:** Functions for loading and preprocessing data.
- **`eda.py`:** Exploratory data analysis tools.
- **`advance_metrics.py`:** Advanced metrics and feature engineering.
- **`segment_analyse.py`:** User segmentation and analysis.
- **`perk_assignment.py`:** Logic for assigning perks to users.
- **`visualization.py`:** Tools for creating visualizations.

## Data
- **Raw Data:** Contains raw CSV files for flights, hotels, sessions, and users.
- **Processed Data:** Includes cleaned and transformed data, user segments, and PCA results.

## Notebooks
- **`eda.ipynb`:** Exploratory data analysis.
- **`kmean_cluster.ipynb`:** K-means clustering for user segmentation.
- **`pca_processing.ipynb`:** Principal Component Analysis (PCA) for dimensionality reduction.
- **`segment_analyse.ipynb`:** Detailed analysis of user segments.
- **`manual_perk_assignment.ipynb`:** Manual assignment of perks to users.

## Reports
- **Visualizations:** Contains various plots and dashboards summarizing the analysis.

## SQL
- **`session_base.sql`:** SQL script for session data.

## Requirements
To run this project, install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation:**
   - Place raw data files in the `data/raw/` directory.
   - Run the data processing scripts to generate processed data.

2. **Analysis:**
   - Use the Jupyter notebooks in the `notebooks/` directory to perform EDA, clustering, and perk assignment.

3. **Visualization:**
   - Generate visualizations using the provided scripts and notebooks.

4. **Reports:**
   - Access the reports and visualizations in the `reports/` directory.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please contact [Guy Kaptue](mailto:guykaptue24@gmail.com).
