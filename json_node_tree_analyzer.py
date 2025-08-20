import json
import os
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Node:
    """Represents a node in the JSON tree structure"""
    key: str
    value: Any
    node_type: str  # 'leaf', 'object', 'array'
    path: str
    parent: 'Node' = None
    children: List['Node'] = field(default_factory=list)
    frequency: int = 1
    similar_nodes: List['Node'] = field(default_factory=list)
    semantic_group: int = -1
    
    def __post_init__(self):
        if self.path == "":
            self.path = self.key
    
    def add_child(self, child: 'Node'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def get_full_path(self) -> str:
        """Get the full path from root to this node"""
        if self.parent is None:
            return self.key
        return f"{self.parent.get_full_path()}.{self.key}"
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return self.node_type == 'leaf'
    
    def get_text_representation(self) -> str:
        """Get text representation for semantic analysis"""
        if self.is_leaf():
            return f"{self.key}: {str(self.value)}"
        return self.key

class JSONNodeTreeAnalyzer:
    """Main class for analyzing JSON files and creating node trees"""
    
    def __init__(self, json_folder_path: str, min_group_size: int = 1, min_frequency: int = 1, max_frequency: int = float('inf')):
        self.json_folder_path = json_folder_path
        self.min_group_size = min_group_size  # Minimum nodes required to keep a semantic group
        self.min_frequency = min_frequency  # Minimum frequency required to show node in visualization
        self.max_frequency = max_frequency  # Maximum frequency allowed to show node in visualization
        self.all_nodes: Dict[str, List[Node]] = defaultdict(list)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.65
        self.semantic_groups: Dict[int, List[Node]] = defaultdict(list)
        self.pruned_groups: Dict[int, List[Node]] = {}  # Store pruned groups for reference
        
    def parse_json_to_nodes(self, data: Dict, parent_key: str = "", parent_node: Node = None) -> List[Node]:
        """Recursively parse JSON data into Node objects"""
        nodes = []
        
        for key, value in data.items():
            # Create path for this node
            current_path = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Object node
                node = Node(
                    key=key,
                    value=value,
                    node_type='object',
                    path=current_path,
                    parent=parent_node
                )
                # Recursively parse children
                child_nodes = self.parse_json_to_nodes(value, current_path, node)
                for child in child_nodes:
                    node.add_child(child)
                
            elif isinstance(value, list):
                # Array node
                node = Node(
                    key=key,
                    value=value,
                    node_type='array',
                    path=current_path,
                    parent=parent_node
                )
                # Handle array elements
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_nodes = self.parse_json_to_nodes(item, f"{current_path}[{i}]", node)
                        for item_node in item_nodes:
                            node.add_child(item_node)
                    else:
                        # Leaf array element
                        leaf_node = Node(
                            key=f"[{i}]",
                            value=item,
                            node_type='leaf',
                            path=f"{current_path}[{i}]",
                            parent=node
                        )
                        node.add_child(leaf_node)
            else:
                # Leaf node
                node = Node(
                    key=key,
                    value=value,
                    node_type='leaf',
                    path=current_path,
                    parent=parent_node
                )
            
            nodes.append(node)
            # Group nodes by key for frequency analysis
            self.all_nodes[key].append(node)
        
        return nodes
    
    def load_json_files(self) -> Dict[str, List[Node]]:
        """Load all JSON files from the specified folder"""
        file_trees = {}
        
        if not os.path.exists(self.json_folder_path):
            print(f"Error: Folder {self.json_folder_path} not found!")
            return file_trees
        
        json_files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files to process...")
        
        for filename in json_files:
            filepath = os.path.join(self.json_folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nodes = self.parse_json_to_nodes(data)
                    file_trees[filename] = nodes
                    print(f"✓ Processed {filename}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"✗ Error processing {filename}: {str(e)}")
                continue
        
        return file_trees
    
    def calculate_node_frequencies(self):
        """Calculate frequency of each node key across all files"""
        for key, nodes in self.all_nodes.items():
            frequency = len(nodes)
            for node in nodes:
                node.frequency = frequency
    
    def compute_semantic_embeddings(self):
        """Compute semantic embeddings for all nodes"""
        print("Computing semantic embeddings...")
        
        for key, nodes in self.all_nodes.items():
            if not nodes:
                continue
            
            # Get text representations
            texts = [node.get_text_representation() for node in nodes]
            
            # Compute embeddings
            embeddings = self.semantic_model.encode(texts)
            
            # Store embeddings
            for i, node in enumerate(nodes):
                node_id = f"{node.get_full_path()}_{id(node)}"
                self.node_embeddings[node_id] = embeddings[i]
    
    def find_semantic_similarities(self):
        """Find semantically similar nodes using cosine similarity"""
        print("Finding semantic similarities...")
        
        # Group nodes by key first (exact matches)
        for key, nodes in self.all_nodes.items():
            if len(nodes) <= 1:
                continue
            
            # Get embeddings for nodes with same key
            node_ids = [f"{node.get_full_path()}_{id(node)}" for node in nodes]
            embeddings = [self.node_embeddings[node_id] for node_id in node_ids]
            
            if len(embeddings) < 2:
                continue
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find similar nodes
            for i, node in enumerate(nodes):
                similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
                for j in similar_indices:
                    if i != j:
                        node.similar_nodes.append(nodes[j])
        
        # Cross-key semantic similarity
        self._find_cross_key_similarities()
    
    def _find_cross_key_similarities(self):
        """Find semantic similarities across different keys"""
        all_embeddings = list(self.node_embeddings.values())
        all_node_refs = []
        
        # Create mapping from embeddings to nodes
        for key, nodes in self.all_nodes.items():
            for node in nodes:
                node_id = f"{node.get_full_path()}_{id(node)}"
                if node_id in self.node_embeddings:
                    all_node_refs.append(node)
        
        if len(all_embeddings) < 2:
            return
        
        # Use DBSCAN clustering for semantic grouping
        clustering = DBSCAN(eps=1-self.similarity_threshold, metric='cosine', min_samples=2)
        cluster_labels = clustering.fit_predict(all_embeddings)
        
        # Group nodes by clusters
        cluster_groups = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                cluster_groups[label].append(all_node_refs[i])
                all_node_refs[i].semantic_group = label
        
        self.semantic_groups = dict(cluster_groups)
        
        # Prune groups that are smaller than minimum size
        self._prune_small_groups()
    
    def _prune_small_groups(self):
        """Prune semantic groups that have fewer nodes than min_group_size"""
        if self.min_group_size <= 1:
            return
        
        print(f"Pruning groups with fewer than {self.min_group_size} nodes...")
        
        # Identify groups to prune
        groups_to_prune = []
        for group_id, nodes in self.semantic_groups.items():
            if len(nodes) < self.min_group_size:
                groups_to_prune.append(group_id)
        
        # Store pruned groups for reference
        pruned_count = 0
        for group_id in groups_to_prune:
            nodes = self.semantic_groups[group_id]
            self.pruned_groups[group_id] = nodes
            
            # Reset semantic group assignment for pruned nodes
            for node in nodes:
                node.semantic_group = -1
            
            # Remove from active groups
            del self.semantic_groups[group_id]
            pruned_count += 1
        
        print(f"Pruned {pruned_count} groups with fewer than {self.min_group_size} nodes")
        print(f"Remaining semantic groups: {len(self.semantic_groups)}")
    
    def _compute_group_centroids(self):
        """Compute centroids for semantic groups for clustering"""
        print("Computing group centroids for clustering...")
        
        self.group_centroids = {}
        
        for group_id, nodes in self.semantic_groups.items():
            if not nodes:
                continue
            
            # Get embeddings for all nodes in this group
            group_embeddings = []
            for node in nodes:
                node_id = f"{node.get_full_path()}_{id(node)}"
                if node_id in self.node_embeddings:
                    group_embeddings.append(self.node_embeddings[node_id])
            
            if group_embeddings:
                # Compute centroid as mean of all node embeddings in the group
                centroid = np.mean(group_embeddings, axis=0)
                self.group_centroids[group_id] = centroid
        
        print(f"Computed centroids for {len(self.group_centroids)} semantic groups")
    
    def _cluster_semantic_groups(self):
        """Cluster semantic groups based on their centroids"""
        if len(self.group_centroids) < 2:
            return {}
        
        print("Clustering semantic groups...")
        
        # Extract centroids and group IDs
        group_ids = list(self.group_centroids.keys())
        centroids = [self.group_centroids[gid] for gid in group_ids]
        
        # Cluster the group centroids
        clustering = DBSCAN(eps=0.3, metric='cosine', min_samples=1)
        cluster_labels = clustering.fit_predict(centroids)
        
        # Map group IDs to cluster labels
        group_to_cluster = {}
        for i, group_id in enumerate(group_ids):
            group_to_cluster[group_id] = cluster_labels[i]
        
        # Create cluster to groups mapping
        cluster_to_groups = defaultdict(list)
        for group_id, cluster_id in group_to_cluster.items():
            cluster_to_groups[cluster_id].append(group_id)
        
        print(f"Created {len(cluster_to_groups)} meta-clusters from {len(group_ids)} semantic groups")
        
        return group_to_cluster, dict(cluster_to_groups)
    
    def create_graph_visualization(self, output_path: str = "node_graph.html"):
        """Create an interactive graph visualization showing only semantic group centroids"""
        print("Creating centroid-based graph visualization...")
        
        G = nx.Graph()
        
        # Get valid semantic group IDs (groups that weren't pruned)
        valid_group_ids = set(self.semantic_groups.keys())
        
        # Compute group centroids and clustering
        self._compute_group_centroids()
        group_to_cluster, cluster_to_groups = self._cluster_semantic_groups()
        
        if not valid_group_ids:
            print("No semantic groups to visualize")
            return None
        
        # Color palette for meta-clusters
        cluster_colors = px.colors.qualitative.Dark24
        
        # Create nodes for semantic group centroids only
        centroid_positions = {}
        node_colors = {}
        node_sizes = {}
        node_labels = {}
        nodes_in_freq_range = 0
        total_nodes_in_groups = 0
        
        for group_id in valid_group_ids:
            if group_id not in self.semantic_groups:
                continue
                
            nodes_in_group = self.semantic_groups[group_id]
            
            # Count nodes that would be in frequency range
            freq_filtered_nodes = [
                node for node in nodes_in_group 
                if self.min_frequency <= node.frequency <= self.max_frequency
            ]
            
            if not freq_filtered_nodes:
                continue  # Skip groups with no nodes in frequency range
            
            total_nodes_in_groups += len(nodes_in_group)
            nodes_in_freq_range += len(freq_filtered_nodes)
            
            # Create centroid node ID
            centroid_id = f"centroid_group_{group_id}"
            G.add_node(centroid_id)
            
            # Set color based on meta-cluster
            if group_id in group_to_cluster:
                cluster_id = group_to_cluster[group_id]
                color = cluster_colors[cluster_id % len(cluster_colors)]
            else:
                color = '#95A5A6'
            
            node_colors[centroid_id] = color
            
            # Size based on number of nodes in group (scaled for visibility)
            node_sizes[centroid_id] = max(20, min(100, len(freq_filtered_nodes) * 3))
            
            # Get representative keys and statistics
            unique_keys = list(set([node.key for node in freq_filtered_nodes]))
            avg_frequency = sum(node.frequency for node in freq_filtered_nodes) / len(freq_filtered_nodes)
            
            # Create descriptive label
            cluster_info = ""
            if group_id in group_to_cluster:
                cluster_info = f", Meta-cluster: {group_to_cluster[group_id]}"
            
            node_labels[centroid_id] = (
                f"Semantic Group {group_id}\n"
                f"Nodes: {len(freq_filtered_nodes)}\n"
                f"Avg Frequency: {avg_frequency:.1f}\n"
                f"Sample Keys: {', '.join(unique_keys[:5])}\n"
                f"{cluster_info}"
            )
        
        # Add edges between semantically related centroids
        # Connect centroids in the same meta-cluster
        if cluster_to_groups:
            for cluster_id, group_ids in cluster_to_groups.items():
                cluster_centroids = [f"centroid_group_{gid}" for gid in group_ids if f"centroid_group_{gid}" in G.nodes()]
                # Connect all centroids in the same cluster
                for i, centroid1 in enumerate(cluster_centroids):
                    for centroid2 in cluster_centroids[i+1:]:
                        G.add_edge(centroid1, centroid2)
        
        print(f"Graph contains {len(G.nodes())} semantic group centroids")
        print(f"Representing {nodes_in_freq_range} nodes from {total_nodes_in_groups} total nodes in groups")
        
        # Create layout - simple circular layout for centroids
        if len(G.nodes()) > 0:
            pos = self._create_centroid_layout(G, group_to_cluster)
        else:
            pos = {}
        
        # Create Plotly visualization
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_colors_list = []
        node_text = []
        node_sizes_list = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors_list.append(node_colors.get(node, '#95A5A6'))
            node_text.append(node_labels.get(node, node))
            node_sizes_list.append(node_sizes.get(node, 20))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_sizes_list,
                color=node_colors_list,
                line=dict(width=3, color='white')
            )
        )
        
        # Create figure with legend for meta-clusters
        title_text = f'JSON Schema - Semantic Group Centroids'
        if self.min_group_size > 1:
            title_text += f' (Groups < {self.min_group_size} nodes excluded)'
        if self.min_frequency > 1 or self.max_frequency != float('inf'):
            freq_range = f"{self.min_frequency}-{self.max_frequency if self.max_frequency != float('inf') else '∞'}"
            title_text += f' (Frequency: {freq_range})'
        
        # Create legend traces for meta-clusters
        legend_traces = []
        if group_to_cluster and cluster_to_groups:
            for cluster_id, group_ids in cluster_to_groups.items():
                # Get sample keys from this cluster
                sample_keys = []
                for group_id in group_ids[:2]:  # Show up to 2 groups per cluster
                    if group_id in self.semantic_groups:
                        group_keys = list(set([node.key for node in self.semantic_groups[group_id][:3]]))
                        sample_keys.extend(group_keys)
                
                cluster_color = cluster_colors[cluster_id % len(cluster_colors)]
                legend_name = f"Meta-cluster {cluster_id}: {', '.join(sample_keys[:4])}..."
                
                # Add invisible trace for legend
                legend_traces.append(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=15, color=cluster_color),
                    name=legend_name,
                    showlegend=True
                ))
        
        # Combine all traces
        all_traces = [edge_trace, node_trace] + legend_traces
        
        fig = go.Figure(data=all_traces,
                       layout=go.Layout(
                           title=dict(text=title_text, font=dict(size=16)),
                           showlegend=True,
                           legend=dict(
                               x=1.02,
                               y=1,
                               xanchor='left',
                               yanchor='top',
                               bgcolor='rgba(255,255,255,0.9)',
                               bordercolor='rgba(0,0,0,0.3)',
                               borderwidth=2
                           ),
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=200,t=40),  # More space for legend
                           annotations=[ dict(
                               text="Each node = semantic group centroid, size = group size, hover for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        # Save visualization
        fig.write_html(output_path)
        print(f"Centroid-based graph visualization saved to: {output_path}")
        
        return fig
    
    def _create_centroid_layout(self, G, group_to_cluster):
        """Create a layout for semantic group centroids"""
        pos = {}
        
        if not group_to_cluster:
            # Simple circular layout if no clustering
            return nx.circular_layout(G, scale=2)
        
        # Group centroids by meta-cluster
        cluster_centroids = defaultdict(list)
        for node in G.nodes():
            if node.startswith('centroid_group_'):
                group_id = int(node.split('_')[-1])
                if group_id in group_to_cluster:
                    cluster_id = group_to_cluster[group_id]
                    cluster_centroids[cluster_id].append(node)
        
        # Position meta-clusters in a circle
        cluster_ids = list(cluster_centroids.keys())
        cluster_positions = {}
        
        for i, cluster_id in enumerate(cluster_ids):
            angle = 2 * np.pi * i / len(cluster_ids)
            cluster_positions[cluster_id] = (
                4 * np.cos(angle),  # Larger radius for better separation
                4 * np.sin(angle)
            )
        
        # Position centroids within each meta-cluster
        for cluster_id, centroids in cluster_centroids.items():
            cluster_center = cluster_positions[cluster_id]
            
            if len(centroids) == 1:
                pos[centroids[0]] = cluster_center
            else:
                # Arrange centroids in a small circle around cluster center
                for j, centroid in enumerate(centroids):
                    sub_angle = 2 * np.pi * j / len(centroids)
                    pos[centroid] = (
                        cluster_center[0] + 0.8 * np.cos(sub_angle),
                        cluster_center[1] + 0.8 * np.sin(sub_angle)
                    )
        
        return pos
    
    def generate_analysis_report(self) -> Dict:
        """Generate a comprehensive analysis report"""
        report = {
            'total_files_processed': len([f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]),
            'total_unique_keys': len(self.all_nodes),
            'total_nodes': sum(len(nodes) for nodes in self.all_nodes.values()),
            'semantic_groups': len(self.semantic_groups),
            'pruned_groups': len(self.pruned_groups),
            'min_group_size_threshold': self.min_group_size,
            'most_frequent_keys': [],
            'semantic_group_details': {},
            'pruned_group_details': {},
            'key_frequency_distribution': {}
        }
        
        # Most frequent keys
        key_frequencies = [(key, len(nodes)) for key, nodes in self.all_nodes.items()]
        key_frequencies.sort(key=lambda x: x[1], reverse=True)
        report['most_frequent_keys'] = key_frequencies[:20]
        
        # Key frequency distribution
        report['key_frequency_distribution'] = dict(key_frequencies)
        
        # Semantic group details
        for group_id, nodes in self.semantic_groups.items():
            keys_in_group = list(set([node.key for node in nodes]))
            report['semantic_group_details'][group_id] = {
                'size': len(nodes),
                'unique_keys': keys_in_group,
                'sample_paths': [node.get_full_path() for node in nodes[:5]]
            }
        
        # Pruned group details
        for group_id, nodes in self.pruned_groups.items():
            keys_in_group = list(set([node.key for node in nodes]))
            report['pruned_group_details'][group_id] = {
                'size': len(nodes),
                'unique_keys': keys_in_group,
                'sample_paths': [node.get_full_path() for node in nodes[:5]],
                'reason': f'Group size ({len(nodes)}) below threshold ({self.min_group_size})'
            }
        
        return report
    
    def save_analysis_results(self, output_dir: str = "analysis_results_2"):
        """Save analysis results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save analysis report
        report = self.generate_analysis_report()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        report = convert_numpy_types(report)
        
        with open(os.path.join(output_dir, "analysis_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save semantic groups
        groups_data = {}
        for group_id, nodes in self.semantic_groups.items():
            groups_data[str(group_id)] = [
                {
                    'key': node.key,
                    'path': node.get_full_path(),
                    'frequency': int(node.frequency),
                    'node_type': node.node_type,
                    'value_sample': str(node.value)[:100] if node.is_leaf() else 'N/A'
                }
                for node in nodes
            ]
        
        with open(os.path.join(output_dir, "semantic_groups.json"), 'w') as f:
            json.dump(groups_data, f, indent=2)
        
        print(f"Analysis results saved to: {output_dir}")
        
        return report
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete analysis pipeline"""
        print("Starting JSON Node Tree Analysis...")
        print("=" * 50)
        
        # Load JSON files
        file_trees = self.load_json_files()
        if not file_trees:
            print("No JSON files were successfully processed.")
            return {}
        
        # Calculate frequencies
        self.calculate_node_frequencies()
        print(f"Calculated frequencies for {len(self.all_nodes)} unique keys")
        
        # Compute semantic embeddings
        self.compute_semantic_embeddings()
        print(f"Computed embeddings for {len(self.node_embeddings)} nodes")
        
        # Find similarities
        self.find_semantic_similarities()
        print(f"Found {len(self.semantic_groups)} semantic groups")
        
        # Compute group centroids
        self._compute_group_centroids()
        
        # Cluster semantic groups
        group_to_cluster, cluster_to_groups = self._cluster_semantic_groups()
        
        # Create visualizations
        self.create_graph_visualization()
        
        # Save results
        report = self.save_analysis_results()
        
        print("=" * 50)
        print("Analysis Complete!")
        print(f"Processed {report['total_files_processed']} files")
        print(f"Found {report['total_unique_keys']} unique keys")
        print(f"Created {report['semantic_groups']} semantic groups")
        
        return report

def main():
    """Main function to run the analysis"""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='JSON Node Tree Analyzer with Semantic Grouping')
    parser.add_argument('--min-group-size', type=int, default=1,
                       help='Minimum number of nodes required to keep a semantic group (default: 1)')
    parser.add_argument('--min-frequency', type=int, default=1,
                       help='Minimum frequency required to show node in visualization (default: 1)')
    parser.add_argument('--max-frequency', type=int, default=float('inf'),
                       help='Maximum frequency allowed to show node in visualization (default: no limit)')
    parser.add_argument('--json-folder', type=str, 
                       default="/home/med_data/<user>/Schema_analysis/all_output_phi4_sampling",
                       help='Path to the folder containing JSON files')
    
    args = parser.parse_args()
    
    # Set the path to your JSON folder
    json_folder = args.json_folder
    min_group_size = args.min_group_size
    min_frequency = args.min_frequency
    max_frequency = args.max_frequency
    
    print(f"Using minimum group size threshold: {min_group_size}")
    print(f"Using frequency range: {min_frequency} - {max_frequency}")
    print(f"JSON folder: {json_folder}")
    
    # Create analyzer instance with minimum group size and frequency
    analyzer = JSONNodeTreeAnalyzer(json_folder, min_group_size=min_group_size, min_frequency=min_frequency, max_frequency=max_frequency)
    
    # Run complete analysis
    report = analyzer.run_complete_analysis()
    
    # Print summary
    if report:
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {report['total_files_processed']}")
        print(f"Total unique keys found: {report['total_unique_keys']}")
        print(f"Total nodes created: {report['total_nodes']}")
        print(f"Minimum group size threshold: {report['min_group_size_threshold']}")
        print(f"Semantic groups identified: {report['semantic_groups']}")
        print(f"Groups pruned (too small): {report['pruned_groups']}")
        
        print("\nTop 10 Most Frequent Keys:")
        for key, freq in report['most_frequent_keys'][:10]:
            print(f"  {key}: {freq} occurrences")
        
        if report['semantic_groups'] > 0:
            print(f"\nRemaining Semantic Groups (>= {min_group_size} nodes):")
            for group_id, details in report['semantic_group_details'].items():
                print(f"  Group {group_id}: {details['size']} nodes, Keys: {details['unique_keys'][:3]}...")
        
        if report['pruned_groups'] > 0:
            print(f"\nPruned Groups (< {min_group_size} nodes):")
            for group_id, details in report['pruned_group_details'].items():
                print(f"  Group {group_id}: {details['size']} nodes, Keys: {details['unique_keys'][:3]}...")


if __name__ == "__main__":
    main()