# **BrainScore Variability**

This repository contains resources and models used in the research paper *"Individual Differences Among Deep Neural Network Models"*. The folder structure and contents are organized to facilitate reproducibility and accessibility.

---

## **Repository Contents**

- **`model.py`**  
  Provides the PyTorch implementation of the AlexNet model.
- **`extract_activations.py`**  
  Extracts the activations of each layer of each training seed in response to a subset (500 images) of the ILSVRC2012 Validation Set 
- **`print_vars.py`**  
  Prints the layer names and parameters for a given tensorflow checkpoint
- **`convert_tensorflow_checkpoint.py`**
  Converts a tensorflow checkpoint to a '.pth' file compatable with Pytorch
- **`similarity_eval.py`**
  Measures the Pearson correlation between the squared Euclidean distance matrices (RDMs) of the layers from two models. Saves results to a csv.
---
