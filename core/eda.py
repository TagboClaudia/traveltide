# core/eda.py
import pandas as pd  # type: ignore
import numpy as np  # type: ignore  # noqa: F401
from datetime import datetime  # type: ignore  # noqa: F401
from IPython.display import display  # type: ignore
from scipy.stats import zscore #type: ignore

def missing_values_summary(df: pd.DataFrame):
    """
    ### Funktion: missing_values_summary
    **Purpose:**
    Zeigt eine Tabelle mit der Anzahl und dem Prozentsatz fehlender Werte (NaN) pro Spalte in einem DataFrame.
    **Arguments:**
    - `df` *(pd.DataFrame)* - Eingabe-DataFrame zur Analyse.
    **Returns:**
    - Gibt nichts zurück, zeigt aber eine formatierte Tabelle mit den Spalten:
      - `column`: Name der Spalte
      - `missing_count`: Anzahl fehlender Werte
      - `missing_percent`: Anteil fehlender Werte in Prozent
    **Example:**
    ```python
    missing_values_summary(my_dataframe)
    ```
    """
    # Anzahl fehlender Werte berechnen
    missing_count = df.isnull().sum()
    # Prozentualer Anteil fehlender Werte
    missing_percent = (missing_count / len(df)) * 100
    # Nur Spalten mit mindestens einem fehlenden Wert anzeigen
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing_count,
        'missing_percent': missing_percent.round(2)
    })
    # Nur relevante Zeilen anzeigen
    missing_df = missing_df[missing_df['missing_count'] > 0].reset_index(drop=True)
    display(missing_df)


def missing_values_summary(df: pd.DataFrame):
    """
    ### Function: missing_values_summary

    **Purpose:**  
    Generates a summary table showing the number and percentage of missing (NaN) values for each column in a DataFrame.

    **Parameters:**  
    - `df` *(pd.DataFrame)*: The input DataFrame to analyze for missing data.

    **Returns:**  
    - None – The function displays a formatted table containing:
        - `column`: Name of the column
        - `missing_count`: Number of missing values in the column
        - `missing_percent`: Percentage of missing values relative to the total number of rows

    **Example:**  
    ```python
    missing_values_summary(my_dataframe)
    ```
    """
    # Anzahl fehlender Werte berechnen
    missing_count = df.isnull().sum()
    # Prozentualer Anteil fehlender Werte
    missing_percent = (missing_count / len(df)) * 100
    # Nur Spalten mit mindestens einem fehlenden Wert anzeigen
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing_count,
        'missing_percent': missing_percent.round(2)
    })
    # Nur relevante Zeilen anzeigen
    # missing_df = missing_df[missing_df['missing_count'] > 0].reset_index(drop=True)
    display(missing_df)
