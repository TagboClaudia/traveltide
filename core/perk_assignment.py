# core/perk_assignment.py
import os
import numpy as np # type: ignore  # noqa: F401
import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.cluster import KMeans, DBSCAN # type: ignore
from sklearn.metrics import silhouette_score, calinski_harabasz_score # type: ignore
import matplotlib.pyplot as plt # type: ignore  # noqa: F401
import seaborn as sns # type: ignore  # noqa: F401
import plotly.graph_objects as go # type: ignore
import plotly.io as pio # type: ignore
pio.renderers.default = "notebook_connected"
class PerkAssignment:
    def __init__(self, df, features):
        self.df = df.copy()
        self.features = features
        self.perks = [
            "1 night free hotel plus flight",
            "exclusive discounts",
            "free checked bags",
            "free hotel meal",
            "no cancellation fees"
        ]
        self.group_names = {
            "1 night free hotel plus flight": "Premium Explorers",        # Statt "High-Value Travelers"
            "exclusive discounts":  "Standard Travelers",      # Statt "Baseline/General Users"
            "free checked bags": "Jetsetters",               # Statt "Frequent Flyers"
            "free hotel meal": "Luxury Stay Seekers",      # Statt "Hotel Enthusiasts"
            "no cancellation fees": "Spontaneous Planners"      # Statt "Indecisive/Last-Minute Bookers"
        }
        self.X = df[features].fillna(0)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
    def run_kmeans(self, n_clusters=6):
        print("="*70)
        print("K-MEANS CLUSTERING")
        print("="*70)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(self.X_scaled)
        sil_score = silhouette_score(self.X_scaled, labels)
        ch_score = calinski_harabasz_score(self.X_scaled, labels)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Calinski-Harabasz Score: {ch_score:.1f}")
        print("\nCluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            pct = (size / len(labels)) * 100
            status = ":weißes_häkchen: Balanced" if 12 <= pct <= 23 else ":warnung: Imbalanced"
            print(f"  Cluster {cluster_id}: {size} users ({pct:.1f}%) {status}")
        cluster_profiles = self._profile_kmeans_clusters(labels)
        perk_assignments = self._assign_perks_kmeans(cluster_profiles)
        return {
            'labels': labels,
            'silhouette': sil_score,
            'ch_score': ch_score,
            'cluster_sizes': cluster_sizes,
            'perk_assignments': perk_assignments,
            'profiles': cluster_profiles
        }
    def run_dbscan(self, eps=0.5, min_samples=10):
        print("\n" + "="*70)
        print("DBSCAN CLUSTERING")
        print("="*70)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Number of clusters: {n_clusters}")
        print(f"Noise points: {n_noise} ({(n_noise/len(labels)*100):.1f}%)")
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            pct = (size / len(labels)) * 100
            label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"
            print(f"  {label}: {size} users ({pct:.1f}%)")
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes
        }
    def _profile_kmeans_clusters(self, labels):
        profiles = {}
        for i in range(max(labels) + 1):
            mask = labels == i
            cluster_data = self.df[mask][self.features]
            profiles[i] = {
                'size': sum(mask),
                'means': cluster_data.mean().to_dict(),
                'medians': cluster_data.median().to_dict()
            }
        return profiles
    def _assign_perks_kmeans(self, profiles):
        assignments = {}
        cluster_scores = {}
        for cluster_id, profile in profiles.items():
            means = profile['means']
            scores = {
                "1 night free hotel plus flight": (
                    means.get('avg_money_spent_flight', 0) * 0.3 +
                    means.get('avg_money_spent_hotel', 0) * 0.3 +
                    means.get('num_flights', 0) * 0.2 +
                    means.get('num_hotels', 0) * 0.2
                ),
                "free checked bags": (
                    means.get('avg_bags', 0) * 0.6 +
                    means.get('num_flights', 0) * 0.4
                ),
                "free hotel meal": (
                    means.get('avg_money_spent_hotel', 0) * 0.5 +
                    means.get('num_hotels', 0) * 0.3 +
                    means.get('num_trips', 0) * 0.2
                ),
                "no cancellation fees": (
                    means.get('cancelation_rate', 0) * 0.5 +
                    means.get('empty_session_ratio', 0) * 0.3 +
                    means.get('num_sessions', 0) * 0.2
                ),
                "exclusive discounts": 1.0
            }
            cluster_scores[cluster_id] = scores
        available_perks = set(self.perks)
        sorted_clusters = sorted(profiles.keys(),
                                 key=lambda x: profiles[x]['size'],
                                 reverse=True)
        for cluster_id in sorted_clusters:
            scores = cluster_scores[cluster_id]
            best_perk = max(available_perks, key=lambda p: scores[p])
            assignments[cluster_id] = {
                "perk": best_perk,
                "group_name": self.group_names[best_perk]
            }
            available_perks.remove(best_perk)
            if not available_perks:
                available_perks = {"exclusive discounts"}
        return assignments
    def compare_methods(self, kmeans_results, dbscan_results):
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print("\n1. COVERAGE:")
        dbscan_coverage = len(dbscan_results['labels']) - dbscan_results['n_noise']
        print("   K-means: 100% of users assigned to clusters")
        print(f"   DBSCAN: {(dbscan_coverage/len(dbscan_results['labels'])*100):.1f}% assigned, "
              f"{dbscan_results['n_noise']} outliers")
        print("\n2. CLUSTER QUALITY:")
        print(f"   K-means Silhouette: {kmeans_results['silhouette']:.3f}")
        print("   DBSCAN Silhouette: N/A (not calculated here)")
        print("\n3. CLUSTER BALANCE:")
        for c, size in kmeans_results['cluster_sizes'].items():
            pct = (size/len(kmeans_results['labels'])*100)
            status = ":weißes_häkchen: Balanced" if 12 <= pct <= 23 else ":warnung: Imbalanced"
            perk_info = kmeans_results['perk_assignments'][c]
            print(f"     Cluster {c}: {pct:.1f}% → {perk_info['perk']} ({perk_info['group_name']}) {status}")
        print("\n4. BUSINESS CONSIDERATIONS:")
        print("   K-means Pros: ✓ Full coverage, ✓ Predictable sizes, ✓ Easy to explain")
        print("   K-means Cons: ✗ May force dissimilar users together, ✗ Sensitive to outliers")
        print("   DBSCAN Pros: ✓ Identifies outliers, ✓ Finds natural groups, ✓ Robust to noise")
        print("   DBSCAN Cons: ✗ Leaves users unassigned, ✗ Cluster count ≠ 5, ✗ Harder to tune")
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(":weißes_häkchen: K-MEANS is recommended for perk assignment (balanced, predictable, full coverage).")
        print("   Use DBSCAN only for exploratory analysis or outlier detection.")
    #### Visualization ###############
    def plot_balanced_distribution(self, df: pd.DataFrame, save_path:str):
        """
        Plot cluster distribution with target ranges (12–23%).
        """
        cluster_counts = df['cluster'].value_counts().sort_index()
        total = len(df)
        percentages = (cluster_counts / total * 100).values
        group_names = group_names = {
            "Premium Explorers",        # Statt "High-Value Travelers"
            "Standard Travelers",      # Statt "Baseline/General Users"
            "Jetsetters",               # Statt "Frequent Flyers"
            "Luxury Stay Seekers",      # Statt "Hotel Enthusiasts"
            "Spontaneous Planners"      # Statt "Indecisive/Last-Minute Bookers"
        }
        target_dist = {"min_pct": 12.0, "max_pct": 23.0}
        min_pct = target_dist.get('min_pct', 12.0)
        max_pct = target_dist.get('max_pct', 23.0)
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2ECC71' if min_pct <= p <= max_pct else '#E74C3C'
                  for p in percentages]
        bars = ax.bar(range(len(percentages)), percentages, color=colors, alpha=0.7)
        # Target range shading
        ax.axhspan(min_pct, max_pct, alpha=0.2, color='green',
                   label=f'Target Range ({min_pct}-{max_pct}%)')
        # Value labels
        for i, (bar, pct, count) in enumerate(zip(bars, percentages, cluster_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%\n({count:,})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12)
        ax.set_ylabel('Percentage of Users (%)', fontsize=12)
        ax.set_title('Balanced Segment Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_path, "balanced_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f":weißes_häkchen: Saved balanced distribution plot: {save_path}")
    def plot_segment_summary(self, df: pd.DataFrame, save_path:str):
        """
        Create dynamic, responsive segment summary visualization including perks.
        """
        group_names = {
            "Premium Explorers",        # Statt "High-Value Travelers"
            "Standard Travelers",      # Statt "Baseline/General Users"
            "Jetsetters",               # Statt "Frequent Flyers"
            "Luxury Stay Seekers",      # Statt "Hotel Enthusiasts"
            "Spontaneous Planners"      # Statt "Indecisive/Last-Minute Bookers"
        }
        all_perks = self.perks
        summary_data = []
        for i, name in enumerate(group_names):
            cluster_df = df[df['cluster'] == i]
            count = len(cluster_df)
            pct = (count / len(df)) * 100
            summary_data.append({
                'Segment': name,
                'Assigned Perk': all_perks[i] if i < len(all_perks) else "N/A",
                'Count': count,
                'Percentage': f'{pct:.1f}%',
                'Avg Spend': f'${cluster_df["total_money_spent"].mean():.0f}' if "total_money_spent" in cluster_df else "N/A",
                'Avg Trips': f'{cluster_df["num_trips"].mean():.1f}' if "num_trips" in cluster_df else "N/A"
            })
        summary_df = pd.DataFrame(summary_data)
        # Interactive Plotly table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(summary_df.columns),
                fill_color='#4CAF50',
                font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                fill_color=[['#F9F9F9', '#FFFFFF'] * (len(summary_df)//2 + 1)],
                align='center',
                font=dict(size=11)
            )
        )])
        fig.update_layout(
            title="Segment Summary with Perks",
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        save_path = os.path.join(save_path, "segment_summary_table.html")
        fig.write_html(save_path, include_plotlyjs='cdn')
        fig.show()
        print(f":balkendiagramm: Saved interactive segment summary: {save_path}")
        # Console summary
        print("\n" + "="*80)
        print("SEGMENT SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))












        