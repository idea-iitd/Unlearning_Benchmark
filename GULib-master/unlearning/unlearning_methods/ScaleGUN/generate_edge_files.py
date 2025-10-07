#!/usr/bin/env python
# Script to generate edge deletion files for ScaleGUN experiments

import os
import numpy as np
import torch
from torch_geometric.utils import degree
import random
import sys
import traceback

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Percentages of edges to select for deletion
PERCENTAGES = [5, 10, 20, 30, 50]

# Base directory for data
DATA_DIR = "/home/gcond/unlearning/GNNDelete/ScaleGUN/data"

# Directory to store deletion files
OUTPUT_DIR = "/home/gcond/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/GULib-master/unlearning/unlearning_methods/ScaleGUN/unlearning_data_edge_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create directories for each dataset
datasets = ["cora", "CiteSeer", "ogbn-arxiv", "Photo", "roman_empire", "amazon_ratings"]
for dataset in datasets:
    os.makedirs(os.path.join(OUTPUT_DIR, dataset), exist_ok=True)

def load_dataset_direct(name):
    """Load a dataset directly from the processed PyG files without downloading"""
    print(f"Loading dataset directly from files: {name}")
    try:
        if name.lower() == 'cora':
            processed_dir = os.path.join(DATA_DIR, 'Cora', 'processed')
        elif name.lower() == 'citeseer':
            processed_dir = os.path.join(DATA_DIR, 'citeseer', 'processed')
        elif name.lower() == 'ogbn-arxiv':
            processed_dir = os.path.join(DATA_DIR, 'ogbn-arxiv', 'processed')
        elif name.lower() == 'photo':
            processed_dir = os.path.join(DATA_DIR, 'Photo', 'processed')
        elif name.lower() == 'roman_empire':
            processed_dir = os.path.join(DATA_DIR, 'roman_empire', 'processed')
        elif name.lower() == 'amazon_ratings':
            processed_dir = os.path.join(DATA_DIR, 'amazon_ratings', 'processed')
        else:
            print(f"Unknown dataset: {name}")
            return None
        
        # Find and load the processed data file
        if not os.path.exists(processed_dir):
            print(f"Processed directory not found: {processed_dir}")
            return None
            
        data_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        if not data_files:
            print(f"No .pt files found in {processed_dir}")
            return None
            
        data_path = os.path.join(processed_dir, data_files[0])
        print(f"Loading data from: {data_path}")
        
        data = torch.load(data_path)
        # Handle different formats of saved PyG datasets
        print(f"Data type: {type(data)}")
        
        # Handle different data formats
        if isinstance(data, tuple):
            print(f"Data is a tuple with {len(data)} elements")
            # For tuple, typically the first element is the graph data
            if len(data) > 0 and hasattr(data[0], 'edge_index'):
                data = data[0]
            else:
                print("Tuple doesn't contain valid PyG data objects")
                return None
        elif isinstance(data, list):
            print(f"Data is a list with {len(data)} elements")
            if len(data) > 0 and hasattr(data[0], 'edge_index'):
                data = data[0]
            else:
                print("List doesn't contain valid PyG data objects")
                return None
        elif hasattr(data, 'data'):
            print("Data has 'data' attribute")
            data = data.data
        
        # Verify the data has required attributes
        if not hasattr(data, 'edge_index') or not hasattr(data, 'num_nodes'):
            print(f"Data doesn't have required attributes: edge_index={hasattr(data, 'edge_index')}, num_nodes={hasattr(data, 'num_nodes')}")
            if hasattr(data, '__dict__'):
                print(f"Available attributes: {data.__dict__.keys()}")
            return None
            
        return data
        
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        traceback.print_exc()
        return None

def calculate_edge_scores(edge_index, num_nodes):
    """Calculate edge scores based on average degree of connected nodes"""
    row, col = edge_index
    node_degrees = degree(row, num_nodes)
    
    edge_scores = []
    for i in range(edge_index.size(1)):
        node1, node2 = edge_index[0, i].item(), edge_index[1, i].item()
        avg_degree = (node_degrees[node1] + node_degrees[node2]) / 2
        edge_scores.append((i, avg_degree.item()))
    
    return edge_scores

def create_del_edges(dataset_name):
    """Generate deletion files for a dataset with different strategies and percentages"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    data = load_dataset_direct(dataset_name)
    if data is None:
        print(f"Failed to load dataset {dataset_name}. Skipping...")
        return False
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get edge indices and number of edges
    if not hasattr(data, 'edge_index'):
        print(f"Dataset {dataset_name} has no edge_index attribute. Available attributes: {dir(data)}")
        return False
        
    edge_index = data.edge_index
    total_edges = edge_index.size(1)
    
    # If data doesn't have num_nodes attribute, calculate it from edge_index
    if not hasattr(data, 'num_nodes'):
        num_nodes = edge_index.max().item() + 1
        print(f"Dataset {dataset_name} has no num_nodes attribute. Estimated: {num_nodes}")
    else:
        num_nodes = data.num_nodes
    
    print(f"Dataset: {dataset_name}, Total edges: {total_edges}")
    print(f"Number of nodes: {num_nodes}")
    
    # Calculate edge scores
    print(f"Calculating edge scores for {dataset_name}...")
    edge_scores = calculate_edge_scores(edge_index, num_nodes)
    
    # Create edge files for different percentages and strategies
    for percent in PERCENTAGES:
        num_edges = int(total_edges * percent / 100)
        print(f"Processing {percent}% deletion for {dataset_name} ({num_edges} edges)")
        
        # Random strategy
        random_indices = random.sample(range(total_edges), num_edges)
        random_edges = edge_index[:, random_indices].t().cpu().numpy()
        random_file = os.path.join(output_dir, f"{dataset_name}_random_{percent}percent.txt")
        np.savetxt(random_file, random_edges, fmt='%d')
        print(f"Saved random edges to {random_file}")
        
        # Ascending strategy (lowest degree first)
        sorted_asc = sorted(edge_scores, key=lambda x: x[1])
        asc_indices = [e[0] for e in sorted_asc[:num_edges]]
        asc_edges = edge_index[:, asc_indices].t().cpu().numpy()
        asc_file = os.path.join(output_dir, f"{dataset_name}_ascending_{percent}percent.txt")
        np.savetxt(asc_file, asc_edges, fmt='%d')
        print(f"Saved ascending edges to {asc_file}")
        
        # Descending strategy (highest degree first)
        sorted_desc = sorted(edge_scores, key=lambda x: x[1], reverse=True)
        desc_indices = [e[0] for e in sorted_desc[:num_edges]]
        desc_edges = edge_index[:, desc_indices].t().cpu().numpy()
        desc_file = os.path.join(output_dir, f"{dataset_name}_descending_{percent}percent.txt")
        np.savetxt(desc_file, desc_edges, fmt='%d')
        print(f"Saved descending edges to {desc_file}")
    
    print(f"Completed processing {dataset_name}")
    return True

def list_dataset_directories():
    """List all available dataset directories"""
    print("\nAvailable dataset directories:")
    for item in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, item)
        if os.path.isdir(path):
            print(f"- {item}")
            proc_dir = os.path.join(path, 'processed')
            if os.path.exists(proc_dir):
                files = os.listdir(proc_dir)
                pt_files = [f for f in files if f.endswith('.pt')]
                print(f"  - Processed dir: {len(pt_files)} .pt files")
            else:
                print("  - No 'processed' directory")

def main():
    """Generate edge deletion files for all datasets"""
    print(f"Generating edge deletion files in {OUTPUT_DIR}")
    
    # List available dataset directories
    list_dataset_directories()
    
    # Process all datasets
    for dataset in datasets:
        success = create_del_edges(dataset)
        if success:
            print(f"Successfully created deletion files for {dataset}")
        else:
            print(f"Failed to create deletion files for {dataset}")
    
    print("Completed processing all datasets")

if __name__ == "__main__":
    main() 