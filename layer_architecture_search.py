import itertools
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf


def setup_layer_architecture_search(input_dim, output_activation='sigmoid',
                                    min_layers=2, max_layers=5,
                                    node_options=[32, 64, 128],
                                    sample_size=None):
    """
    Extension to the existing grid search setup that focuses on layer architecture.
    This function creates a modified parameter grid for layer architecture search.

    Parameters:
    input_dim (int): Input dimension for the model
    output_activation (str): Activation function for output layer
    min_layers (int): Minimum number of hidden layers to test
    max_layers (int): Maximum number of hidden layers to test
    node_options (list): List of node counts to test for each layer
    sample_size (int): If set, randomly sample this many architectures

    Returns:
    tuple: (model, param_grid) for use with existing grid search functionality
    """
    # Import from your existing modules
    from models import create_grid_search_model

    # Create the model builder function - using your existing model creator
    def model_builder(hidden_layers=[128, 64], dropout_rates=None, learning_rate=0.001):
        """Model builder that handles variable-length architectures"""
        # Ensure dropout_rates matches hidden_layers in length
        if dropout_rates is None:
            dropout_rates = [0.2] * len(hidden_layers)
        elif len(dropout_rates) != len(hidden_layers):
            dropout_rates = [dropout_rates[0]] * len(hidden_layers)

        return create_grid_search_model(
            hidden_layers=hidden_layers,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
            input_dim=input_dim
        )

    # Create the KerasClassifier
    keras_clf = KerasClassifier(
        model=model_builder,
        verbose=1,
        epochs=50,  # Lower for grid search
        batch_size=32
    )

    # Generate architecture configurations
    architecture_configs = []

    # For each layer depth
    for num_layers in range(min_layers, max_layers + 1):
        # Generate all combinations of the specified node options
        layer_combinations = list(itertools.product(node_options, repeat=num_layers))
        architecture_configs.extend(layer_combinations)

    print(f"Generated {len(architecture_configs)} different layer architectures")

    # Sample configurations if requested
    if sample_size and sample_size < len(architecture_configs):
        np.random.seed(42)  # For reproducibility
        sampled_indices = np.random.choice(len(architecture_configs), size=sample_size, replace=False)
        architecture_configs = [architecture_configs[i] for i in sampled_indices]
        print(f"Sampled {len(architecture_configs)} architectures for testing")

    # Create parameter grid focused on layer architecture
    param_grid = {
        'model__hidden_layers': architecture_configs,
        'model__dropout_rates': [[0.2]],  # same dropout rate for all layers
        'model__learning_rate': [0.001],  # Fixed learning rate for architecture search
        'batch_size': [32],  # Fixed batch size
        'epochs': [50]  # Fixed epochs
    }

    return keras_clf, param_grid