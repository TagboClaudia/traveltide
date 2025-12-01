# type: ignore‚
import os
import pandas as pd # type: ignore
import sqlalchemy as sa # type: ignore
from sqlalchemy.exc import SQLAlchemyError # type: ignore
from sqlalchemy.engine import URL # type: ignore
from dotenv import load_dotenv # type: ignore

# ============================================================
# :zahnrad: Utility Functions (General-Purpose)
# ============================================================
# Globale Verbindungsobjekte
load_dotenv()
_engine = None
_connection = None
def init_connection(db_url: str =None):
    """
    Initialisiert die Verbindung zur PostgreSQL-Datenbank.
    Args:
        db_url (str): Verbindungs-URL zur PostgreSQL-Datenbank.
    """
    global _engine, _connection
    try:
        if not db_url:
           db_url = URL.create(
                drivername="postgresql",
                username=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                query={"sslmode": os.getenv("DB_SSLMODE", "require")}
            )
        _engine = sa.create_engine(db_url, pool_pre_ping=True)
        _connection = _engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        print(":weißes_häkchen: Connected to PostgreSQL database.")
    except SQLAlchemyError as e:
        print(f":x: Connection failed: {e}")
        _engine = None
        _connection = None
def execute_query(query: str) -> pd.DataFrame:
    """
    Führt eine SQL-Abfrage aus und gibt das Ergebnis als DataFrame zurück.
    """
    if not _connection:
        raise ConnectionError(":warnung: No active database connection.")
    try:
        df = pd.read_sql(sa.text(query), _connection)
        print(f":weißes_häkchen: Query executed. {len(df)} rows retrieved.")
        return df
    except SQLAlchemyError as e:
        print(f":x: Query failed: {e}")
        return pd.DataFrame()
def execute_sql_file(sql_file_path: str) -> pd.DataFrame:
    """
    Führt eine SQL-Datei aus und gibt das Ergebnis als DataFrame zurück.
    """
    if not os.path.exists(sql_file_path):
        raise FileNotFoundError(f":warnung: SQL file not found: {sql_file_path}")
    with open(sql_file_path, "r") as file:
        sql_query = file.read()
    print(f":blatt_oben: Executing SQL from file: {sql_file_path}")
    return execute_query(sql_query)
def to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    ### Purpose
    Converts given columns in a DataFrame to datetime format safely.
    ### Arguments
    - `df`: Input DataFrame.
    - `columns`: List of column names to convert.
    ### Returns
    - DataFrame with converted datetime columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
def group_summary(df: pd.DataFrame, group_cols: list[str], metrics: dict, sort_by=None, top_n=None) -> pd.DataFrame:
    """
    ### Purpose
    Generalized groupby summary generator (can be reused for hotels, users, flights, etc.)
    ### Arguments
    - `df`: DataFrame to aggregate.
    - `group_cols`: Columns to group by.
    - `metrics`: Dict of aggregations, e.g. {'trip_id': 'count', 'price': 'mean'}
    - `sort_by`: Optional column name to sort descending.
    - `top_n`: Optional integer for limiting to top results.
    ### Returns
    - Aggregated and optionally sorted/filtered DataFrame.
    """
    result = df.groupby(group_cols).agg(metrics).reset_index()
    if sort_by and sort_by in result.columns:
        result = result.sort_values(by=sort_by, ascending=False)
    if top_n:
        result = result.head(top_n)
    return result.round(2)
def calculate_duration(df: pd.DataFrame, start_col: str, end_col: str, new_col: str = "duration_days") -> pd.DataFrame:
    """
    ### Purpose
    Calculates the duration in days between two datetime columns.
    ### Arguments
    - `df`: Input DataFrame.
    - `start_col`: Start datetime column name.
    - `end_col`: End datetime column name.
    - `new_col`: Name for the calculated duration column.
    ### Returns
    - DataFrame with new duration column.
    """
    df = to_datetime(df, [start_col, end_col])
    df[new_col] = (df[end_col] - df[start_col]).dt.days
    return df