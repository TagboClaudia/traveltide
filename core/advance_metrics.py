# core/advance_metrics.py
import os
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.cluster import KMeans, DBSCAN # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore  # noqa: F401
from IPython.display import display # type: ignore # type: ignore  # noqa: F401
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.io as pio # type: ignore  # noqa: F401
# Assume df_user is your dataframe
# --- Feature Preparation ---
def prepare_features(df):
    df = df.copy()
    # Derived ratios
    df['flight_ratio'] = np.where(df['num_trips'] > 0,
                                  df['num_flights'] / df['num_trips'], 0)
    df['hotel_ratio'] = np.where(df['num_trips'] > 0,
                                 df['num_hotels'] / df['num_trips'], 0)
    df['cancelation_rate'] = np.where(df['num_trips'] > 0,
                                      df['num_canceled_trips'] / df['num_trips'], 0)
    df['empty_session_ratio'] = np.where(df['num_sessions'] > 0,
                                         df['num_empty_sessions'] / df['num_sessions'], 0)
    # Additional features
    df['total_money_spent'] = df['avg_money_spent_flight'] + df['avg_money_spent_hotel']
    df['engagement_score'] = df['num_sessions'] * df['avg_session_clicks']
    df['booking_conversion'] = np.where(df['num_sessions'] > 0,
                                        df['num_trips'] / df['num_sessions'], 0)
    # Normalize skewed features
    skewed = ['avg_money_spent_hotel','avg_money_spent_flight',
              'num_trips','num_flights','num_hotels']
    for col in skewed:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df
# --- Feature Sets ---
BEHAVIOR_FEATURES = [
    'avg_bags','avg_money_spent_hotel','avg_money_spent_flight',
    'num_trips','num_flights','num_hotels',
    'cancelation_rate','empty_session_ratio','booking_conversion'
]
MINIMAL_FEATURES = [
    'avg_bags','total_money_spent','num_trips',
    'cancelation_rate','empty_session_ratio'
]
COMPREHENSIVE_FEATURES = [
    'num_clicks','avg_session_clicks','num_empty_sessions',
    'num_canceled_trips','num_sessions','avg_session_duration',
    'num_trips','num_destinations','num_flights','num_hotels',
    'avg_money_spent_flight','avg_money_spent_hotel','avg_km_flown',
    'avg_bags','flight_ratio','hotel_ratio',
    'cancelation_rate','empty_session_ratio'
]
# --- Evaluation ---
def evaluate_feature_sets(df, feature_sets_dict):
    results = {}
    for name, features in feature_sets_dict.items():
        print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
        X = df[features].fillna(0).copy()
        # Feature re-weighting
        weights = {
            'avg_bags': 1.2,
            'cancelation_rate': 1.2,
            'empty_session_ratio': 1.2,
            'num_trips': 0.8,
            'num_flights': 0.8,
            'num_hotels': 0.8
        }
        for f, w in weights.items():
            if f in X.columns:
                X[f] = X[f] * w
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        silhouette_scores, db_scores = [], []
        k_range = range(5, 9)  # try 5–8 clusters
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            silhouette_scores.append(sil_score)
            db_scores.append(db_score)
            print(f"K={k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
        optimal_k = k_range[np.argmax(silhouette_scores)]
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        print(f"\nDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        results[name] = {
            'features': features,
            'optimal_k': optimal_k,
            'best_silhouette': max(silhouette_scores),
            'dbscan_clusters': n_clusters,
            'dbscan_noise': n_noise
        }
    return results
def analyze_feature_importance_pca(df, features):
    """
    Use PCA to understand which features contribute most to variance
    """
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    # Create feature importance dataframe
    importance_df = pd.DataFrame(
        np.abs(pca.components_[:3]),  # First 3 PCs
        columns=features,
        index=[f'PC{i+1}' for i in range(3)]
    )
    print("\nFeature importance by Principal Component:")
    print(importance_df.round(3))
    print(f"\nVariance explained by first 3 PCs: {pca.explained_variance_ratio_[:3].sum():.2%}")
    return importance_df
def correlation_analysis(df, features):
    """
    Identify highly correlated features that might be redundant
    """
    corr_matrix = df[features].corr().abs()
    # Find highly correlated pairs (>0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    if high_corr_pairs:
        print("\nHighly correlated features (>0.8):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("\nNo highly correlated features (>0.8) found")
    return high_corr_pairs
def plot_pca_component_heatmap(
    component_matrix: pd.DataFrame,
    title: str = "PCA Component Loadings",
    figsize: tuple = (12, 8),
    cmap: str = "coolwarm",
    save_path: str = None
):
    """
    Plots a professional heatmap of absolute PCA component loadings.
    Args:
        component_matrix (pd.DataFrame): PCA components matrix with features as index
                                         and principal components as columns.
        title (str): Title of the heatmap.
        figsize (tuple): Figure size (width, height).
        cmap (str): Color map for the heatmap.
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
    Returns:
        matplotlib.figure.Figure: The heatmap figure.
    """
    # Compute absolute values
    abs_matrix = component_matrix.abs()
    plt.figure(figsize=figsize)
    sns.heatmap(
        abs_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Absolute Loading"},
        square=False
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.ylabel("Features", fontsize=12)
    plt.xlabel("Principal Components", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(save_path, "absolute_pca_heatmap.png")
    plt.savefig(save_path, dpi=300)
    print(f":weißes_häkchen: Heatmap saved to: {save_path}")
    plt.show()
def plot_2d(df:pd.DataFrame, save_path: str, optimal_n_clusters: int=3,  cluster_name:str='cluster') -> None:
    """
    Plot clusters using the first two PCA components (2D scatter plot).
    Parameters
    ----------
    user_pca : pd.DataFrame
        PCA-transformed data with a 'group' column containing cluster labels.
    optimal_n_clusters : int
        Number of clusters used in K-Means.
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df.iloc[:, 0],  # PC1
        df.iloc[:, 1],  # PC2
        c=df[cluster_name],
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    plt.title(f'K-Means Clustering mit {optimal_n_clusters} Clustern (2D PCA)')
    plt.xlabel('PCA Komponente 1')
    plt.ylabel('PCA Komponente 2')
    plt.grid(True)
    # Legend for cluster groups
    plt.legend(*scatter.legend_elements(), title='Gruppe')
    save_path = os.path.join(save_path, f"{optimal_n_clusters}-means_{cluster_name}_2d.png")
    plt.savefig(save_path, dpi=300)
    plt.tight_layout()
    plt.show()
def plot_3d(X_pca: np.ndarray, labels: np.ndarray, n_clusters: int, save_path:str, cluster_name:str='cluster'):
        """
        Generates 3D Scatter plot.
        - Renders inside VS Code.
        - Adds an animation button to rotate the camera.
        """
        if X_pca.shape[1] < 3:
            print("   :warnung: Not enough components for 3D plotting")
            return
        df_plot = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
        df_plot[cluster_name] = labels
        # Create the base figure
        fig = px.scatter_3d(
            df_plot,
            x='PC1', y='PC2', z='PC3',
            color=cluster_name,
            title=f'K-Means Clustering with {n_clusters} Clusters (3D PCA)',
            labels={cluster_name: 'Cluster Group'},
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=4))
        # --- STEP 2: ADD ROTATION ANIMATION (OPTIONAL) ---
        # This makes the "animate" part real. It adds a button to spin the view.
        x_eye = -1.25
        y_eye = 2
        z_eye = 0.5
        fig.update_layout(
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=1, x=0.8, xanchor='left', yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(
                    label='Play Animation',
                    method='animate',
                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                     fromcurrent=True, mode='immediate')]
                )]
            )]
        )
        # Create frames for the rotation
        frames = []
        for t in np.arange(0, 6.28, 0.1):
            xe, ye = 1.25 * np.cos(t), 1.25 * np.sin(t)
            frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=z_eye))))
        fig.frames = frames
        # -----------------------------------------------
        # Save HTML for sharing
        save_path = os.path.join(save_path, f'{n_clusters}_means_clusters_3d.html')
        fig.write_html(save_path)
        fig.show()
        print(f"   :balkendiagramm: Saved 3D plot: {save_path}")










