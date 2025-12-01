# type: ignore
import os
import sys
import pandas as pd # type: ignore
from IPython.display import display # type: ignore
# Add core module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.utils import init_connection, execute_query, execute_sql_file
# Initialize DB connection
init_connection()

# Base directories
base_path = os.path.dirname(__file__)
raw_data_path = os.path.join(base_path, '..', 'data', 'raw')
processed_data_path = os.path.join(base_path, '..', 'data', 'processed')
sql_path = os.path.join(base_path, '..', 'sql')
reports_path = os.path.join(base_path, '..', 'report')
feature_path = os.path.join(processed_data_path, 'feature_metrics')
segment_path = os.path.join(processed_data_path, 'segmentation')
pca_path = os.path.join(processed_data_path,"pca")
kmean_path = os.path.join (processed_data_path, 'kmean')
non_ml_path = os.path.join(processed_data_path, 'non_ml')

# mit diese fucntion kann man den path bekommen
def get_path(data_type: str, table_name: str) -> tuple[str, str]:
    if data_type == "raw":
        path = os.path.join(raw_data_path, f"{table_name}.csv")
    elif data_type == "processed":
        path = os.path.join(processed_data_path, f"{table_name}.csv")
    elif data_type == "sql":
        path = os.path.join(sql_path, f"{table_name}.sql")
    elif data_type == 'feature':
        path = os.path.join(feature_path, f"{table_name}.csv")
    elif data_type == 'segment':
        path = os.path.join(segment_path, f"{table_name}.csv")
    elif data_type == 'pca':
        path= os.path.join(pca_path, f"{table_name}.csv")
    elif data_type == 'kmean':
        path= os.path.join(kmean_path, f"{table_name}.csv")
    elif data_type == 'non_ml':
        path= os.path.join(non_ml_path, f"{table_name}.csv")
    else:
        raise ValueError(f"❌ Ungültiger Datentyp: '{data_type}'. Erlaubt sind 'raw', 'processed', 'sql'.")
    return path, data_type
  
# mit diese function kann man die tabelle laden
def load_table(data_type: str, table_name: str, show_table_display: bool = False) -> pd.DataFrame:
  file_path, resolved_type = get_path(data_type, table_name)
  if resolved_type == "sql" and os.path.exists(file_path):
    print(f" Lade Tabelle '{table_name}' aus SQL-Datei: {file_path}")
    df = execute_sql_file(file_path)
    print(f" SQL-Abfrage erfolgreich. Zeilen: {len(df)}")
    new_csv_path = os.path.join(processed_data_path, f"{table_name}.csv")
    df.to_csv(new_csv_path, index=False)
    print(f" {new_csv_path}")
  elif resolved_type in ["raw", "processed","feature", "segment","pca","kmean", "non_ml"] and os.path.exists(file_path):
    print(f" Lade Tabelle '{table_name}' aus CSV: {file_path}")
    df = pd.read_csv(file_path)
    print(f" CSV geladen. Zeilen: {len(df)}")
  else:
    print(f" Lade Tabelle '{table_name}' direkt aus der Datenbank...")
    df = execute_query(f"SELECT * FROM {table_name};")
    print(f" Datenbankabfrage erfolgreich. Zeilen: {len(df)}")
    if not df.empty and resolved_type in ["raw", "processed"]:
      df.to_csv(file_path, index=False)
      print(f" Gespeichert unter: {file_path}")
    elif df.empty:
      print(f" Keine Daten gefunden für Tabelle '{table_name}'")
  if not df.empty and show_table_display:
    display(df.sample(min(100, len(df))))
  return df

# mit diese function kann man eine benutzerdefinierte abfrage laden, wenn man ein bestimmte abfrage machen möchte.
def load_custom_query(query: str) -> pd.DataFrame:
  print(" Führe benutzerdefinierte SQL-Abfrage aus...")
  df = execute_query(query)
  if not df.empty:
    display(df.sample(min(100, len(df))))
  else:
    print(" Keine Ergebnisse für diese Abfrage.")
  return df

if __name__ == "__main__":
    # Beispiel: Lade die 'flights' Tabelle und zeige eine Stichprobe
    flights = load_table(data_type="raw", table_name="flights", show_table_display=False)
    users = load_table(data_type="processed", table_name="users", show_table_display=False)
    sessions = load_table(data_type="processed", table_name="sessions", show_table_display=False)
    hotels = load_table(data_type="raw", table_name="hotels", show_table_display=False)




