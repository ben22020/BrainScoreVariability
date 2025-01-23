import os
import torch
import numpy as np
import similarity
import pandas as pd

# Define the similarity measure
measure = similarity.make("measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=pearson")

# File structure
base_dir = "./model_activations"
layers = [
    "classifier_0_activations.pt",
    "classifier_3_activations.pt",
    "features_0_activations.pt",
    "features_3_activations.pt",
    "features_6_activations.pt",
    "features_8_activations.pt",
    "features_10_activations.pt"
]

# Get all training seed directories
training_seeds = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

# Create a directory to save CSV results
output_dir = "./similarity_results"
os.makedirs(output_dir, exist_ok=True)

# Compute pairwise similarity for each layer
for layer in layers:
    print(f"Processing layer: {layer}")
    # Create a DataFrame to store results
    results_df = pd.DataFrame(index=training_seeds, columns=training_seeds)

    for seed_1 in training_seeds:
        activation_1_path = os.path.join(base_dir, seed_1, layer)
        activation_1 = torch.load(activation_1_path)
        activation_1 = activation_1.view(activation_1.size(0), -1).detach().cpu().numpy()

        for seed_2 in training_seeds:
            activation_2_path = os.path.join(base_dir, seed_2, layer)
            activation_2 = torch.load(activation_2_path)
            activation_2 = activation_2.view(activation_2.size(0), -1).detach().cpu().numpy()

            # Compute similarity
            score = measure(activation_1, activation_2)

            # Store the score
            results_df.loc[seed_1, seed_2] = score

    # Save the results to a CSV file
    output_csv_path = os.path.join(output_dir, f"{layer.replace('_activations.pt', '')}_similarity.csv")
    results_df.to_csv(output_csv_path)
    print(f"Saved similarity results for layer {layer} to {output_csv_path}")
