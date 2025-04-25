# Neural Network Architecture Optimization Framework

A comprehensive TensorFlow-based framework for building, optimizing, and evaluating neural network models for binary classification tasks.

## Overview

This project provides a flexible architecture for training neural networks with various configurations and hyperparameters. It includes tools for data preparation, model creation, training, hyperparameter optimization through grid search, and layer architecture search.

## Features

- **Data Processing Pipeline**
  - Loading and preprocessing CSV data
  - Automatic class balancing
  - Train/validation/test splitting
  - Feature scaling

- **Model Building**
  - Configurable network architecture
  - Support for various activation functions
  - Customizable loss functions including Focal Loss
  - Flexible dropout rates

- **Advanced Optimization Techniques**
  - Standard grid search for hyperparameter optimization
  - Layer architecture search to find optimal network topology
  - Early stopping and learning rate reduction
  - Support for L2 regularization

- **Comprehensive Evaluation**
  - Performance metrics (accuracy, AUC, etc.)
  - Confusion matrix visualization
  - Learning curve plots
  - Threshold optimization
  - Detailed experiment summaries

## Project Structure

- **Core Modules**
  - `models.py` - Neural network architecture definitions
  - `data_processing.py` - Data loading and preparation
  - `training.py` - Model training and grid search functionality
  - `callbacks.py` - Custom callbacks for training monitoring

- **Advanced Search Algorithms**
  - `layer_architecture_search.py` - Search for optimal network topology
  - `run_layer_search.py` - CLI for running architecture search experiments

- **Utilities**
  - `utils.py` - Plotting and reporting functions

## Installation

```bash
# Clone repository
git clone https://github.com/nicolenadine/CS171_assignment3.git
cd neural-network-framework

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Model Training

```bash
python main.py --data_file your_data.csv --hidden_layers 128 64 32 --dropout_rates 0.3 0.2 0.1 --learning_rate 0.001 --epochs 100
```

### Grid Search for Hyperparameter Optimization

```bash
python main.py --data_file your_data.csv --grid_search
```

### Layer Architecture Search

```bash
python run_layer_search.py --data_file your_data.csv --layer_search --min_layers 2 --max_layers 4 --node_options 32 64 128 256
```

## Command Line Arguments

The framework supports numerous command line arguments to customize the model and training process:

### Model Architecture
- `--hidden_layers`: Sizes of hidden layers (space-separated integers)
- `--dropout_rates`: Dropout rates for each layer (space-separated floats)
- `--output_activation`: Activation function for output layer (`sigmoid` or `softmax`)

### Training Parameters
- `--learning_rate`: Initial learning rate
- `--weight_decay`: L2 regularization strength
- `--batch_size`: Training batch size
- `--epochs`: Maximum number of training epochs

### Early Stopping & LR Reduction
- `--patience`: Patience for early stopping
- `--monitor`: Metric to monitor (`val_auc`, `val_loss`, `val_accuracy`)
- `--lr_factor`: Factor by which to reduce learning rate
- `--lr_patience`: Patience for learning rate reduction
- `--min_lr`: Minimum learning rate

### Loss Function
- `--loss`: Loss function (`binary_crossentropy`, `focal_loss`, `categorical_crossentropy`)
- `--focal_gamma`: Gamma parameter for focal loss
- `--focal_alpha`: Alpha parameter for focal loss

### Layer Architecture Search
- `--layer_search`: Enable layer architecture search
- `--min_layers`: Minimum number of hidden layers to test
- `--max_layers`: Maximum number of hidden layers to test
- `--node_options`: Node count options for each layer
- `--sample_size`: Number of architectures to sample

## Example Workflows

### Finding the Optimal Network Architecture

1. Run a layer architecture search to find the best topology:
   ```bash
   python run_layer_search.py --data_file your_data.csv --layer_search --experiment_name "arch_search_1"
   ```

2. Use the best configuration for final model training:
   ```bash
   python main.py --data_file your_data.csv --hidden_layers 256 128 64 --dropout_rates 0.3 0.2 0.1 --experiment_name "final_model"
   ```

### Handling Class Imbalance

1. Experiment with Focal Loss:
   ```bash
   python main.py --data_file your_data.csv --loss focal_loss --focal_gamma 2.0 --focal_alpha 0.25 --experiment_name "focal_loss_test"
   ```

2. Compare with standard binary cross-entropy:
   ```bash
   python main.py --data_file your_data.csv --loss binary_crossentropy --experiment_name "bce_baseline"
   ```

## Output and Results

The framework automatically creates an output directory for each run with:

- Trained model file (.keras)
- Configuration summary (.txt)
- Confusion matrix plot (.png)
- Learning curves plot (.png)
- Detailed performance metrics

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Choose appropriate license]

## Acknowledgments

- TensorFlow and Keras teams
- scikit-learn contributors
- [Any additional acknowledgments]
