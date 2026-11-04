# SNNs: Spiking Neural Networks, and their Advantages

I am a 2026 COSMOS scholar, and I was part of Cluster 1. My project, which has culminated over the course of 4 weeks of discussion, looks at spiking neural networks and the claimed energy efficiency "advantage" that it holds, according to popularized research papers. Claude Opus used in assistance to code specific sampling methods and create LeNet-style CNN.

For the overview of the project and the poster, please see `COSMOS_SNN_Project.png` (Project Poster).

# Project Setup

Note: The Project uses two single-layer neural network, which may lead to biased results.

Control Group 1: Single-layer Perceptron, 784
Experimental Group 1: Single-layer LIF Neuron, Poisson Rate-Encoding, 784
Control Group 2: Single-layer Perceptron, 84
Experimental Group 2: Single-layer LIF Neuron, Poisson Rate-Encoding, 84

PoMNIST: Poisson Rate-Encoded MNIST data used for Experimental Group 1
EmbedMNIST: CNN-converted MNIST data with preserved labels, used for Group class 2; encoded into Poisson spike rates within `ann_vs_lif_embedding_sized.ipynb` for Experimental Group 2

Originally planned to use SHD (Spiking Heidelberg Digits) dataset, data is too bad

Visuals within code files (not shown in project poster):
- Customizable Raster plot per sample for neural activity
- Spike graph per sample for neural activity

# Credit

Credit to Prof. H. Tad Blair of the UCLA Psychology Department for helping me with some of the core ideas for experimental construction in this AI project.
