"""
Layer Architecture Search for Neural Networks
Extends the existing grid search to focus on finding optimal layer architecture.
"""
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# Import from existing modules
from config import setup_output_directory, print_configuration, get_file_paths
from data_processing import prepare_data
from models import create_model
from training import run_grid_search, train_model, evaluate_model, get_experiment_summary
from callbacks import get_callbacks
from utils import save_model_summary, plot_confusion_matrix, plot_learning_curves

# Import the layer architecture search extension
from layer_architecture_search import setup_layer_architecture_search

def parse_layer_search_arguments():
    """Parse command line arguments including layer architecture search parameters"""
    parser = argparse.ArgumentParser(description='Neural Network Layer Architecture Search')

    # Model architecture parameters
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 64, 32],
                        help='Size of hidden layers')
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[0.4, 0.3, 0.2],
                        help='Dropout rates for each layer')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 regularization strength (weight decay parameter in Adam optimizer)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')

    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--monitor', type=str, default='val_accuracy',
                        choices=['val_auc', 'val_loss', 'val_accuracy'],
                        help='Metric to monitor for early stopping')

    # Learning rate reduction parameters
    parser.add_argument('--lr_factor', type=float, default=0.2,
                        help='Factor by which to reduce learning rate')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Patience for learning rate reduction')
    parser.add_argument('--min_lr', type=float, default=0.00001,
                        help='Minimum learning rate')

    # Prediction threshold
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold for binary classification')

    # Data parameters
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Test set size (proportion)')
    parser.add_argument('--val_size', type=float, default=0.20,
                        help='Validation set size (proportion of training data)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')

    # File parameters
    parser.add_argument('--data_file', type=str, default='cleaned_data3.csv',
                        help='Path to the input CSV file')

    # Loss function
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        choices=['binary_crossentropy', 'focal_loss', 'categorical_crossentropy'],
                        help='Loss function to use')
    parser.add_argument("--output_activation", type=str, default="sigmoid", choices=["sigmoid", "softmax"],
                        help="Activation function for the output layer")

    # Focal loss parameters
    parser.add_argument('--focal_gamma', type=float, default=1.0,
                        help='Gamma parameter for focal loss (higher values focus more on hard examples)')
    parser.add_argument('--focal_alpha', type=float, default=0.50,
                        help='Alpha parameter for focal loss (balancing factor for positive class)')

    # Standard grid search flag
    parser.add_argument('--grid_search', action='store_true',
                        help='Enable standard grid search for hyperparameter optimization')

    # Experiment name
    parser.add_argument('--experiment_name', type=str, default='',
                        help='Optional name for this experiment run')

    # Layer architecture search parameters
    parser.add_argument('--layer_search', action='store_true',
                      help='Run layer architecture search instead of standard grid search')
    parser.add_argument('--min_layers', type=int, default=2,
                      help='Minimum number of hidden layers to test')
    parser.add_argument('--max_layers', type=int, default=5,
                      help='Maximum number of hidden layers to test')
    parser.add_argument('--node_options', type=int, nargs='+', default=[32, 64, 128],
                      help='Node count options for each layer')
    parser.add_argument('--sample_size', type=int, default=0,
                      help='Number of architectures to sample (0 = test all)')

    return parser.parse_args()

def run_layer_architecture_search(data, args):
    """Run grid search specifically focused on layer architecture"""
    print("Running layer architecture grid search...")

    # Setup the layer architecture search
    model, param_grid = setup_layer_architecture_search(
        input_dim=data['input_dim'],
        output_activation=args.output_activation,
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        node_options=args.node_options,
        sample_size=args.sample_size if args.sample_size > 0 else None
    )

    # Use existing grid search function
    best_model, grid_result = run_grid_search(data, args, custom_model=model, custom_param_grid=param_grid)

    return best_model, grid_result

def main():
    """Main function to run layer architecture search"""
    # Parse arguments
    args = parse_layer_search_arguments()

    # Print configuration
    print_configuration(args)

    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    # Prepare data using existing function
    data = prepare_data(args)

    # Create output directory and file paths
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_dir = setup_output_directory(args.experiment_name, timestamp)
    file_paths = get_file_paths(args, output_dir, timestamp)

    # Run appropriate search
    if args.layer_search:
        print("Running specialized layer architecture search...")
        best_model, grid_result = run_layer_architecture_search(data, args)
    elif args.grid_search:
        print("Running standard grid search...")
        best_model, grid_result = run_grid_search(data, args)
    else:
        print("Running standard model training (no grid search)...")
        # Import and run standard training
        from main import main as standard_main
        return standard_main()

    # Follows existing code flow for evaluating and saving results
    # Extract best parameters
    try:
        best_params = best_model.get_params()
        best_hidden_layers = best_params.get('model__hidden_layers', args.hidden_layers)
        best_dropout_rates = best_params.get('model__dropout_rates', args.dropout_rates)
        best_learning_rate = best_params.get('model__learning_rate', args.learning_rate)

        print("\nBest architecture found:")
        print(f"Hidden layers: {best_hidden_layers}")
    except Exception as e:
        print(f"Could not extract best parameters: {str(e)}")
        print("Using default parameters")
        best_hidden_layers = args.hidden_layers
        best_dropout_rates = args.dropout_rates
        best_learning_rate = args.learning_rate

    # Create and train final model with best architecture
    print("\nTraining final model with best architecture...")

    # Customize args for final model
    args.hidden_layers = best_hidden_layers
    args.dropout_rates = best_dropout_rates if len(best_dropout_rates) == len(best_hidden_layers) else [0.2] * len(best_hidden_layers)
    args.learning_rate = best_learning_rate

    # Create and train final model 
    model = create_model(
        args=args,
        input_dim=data['input_dim'],
        hidden_layers=args.hidden_layers,
        dropout_rates=args.dropout_rates,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha
    )

    # Get callbacks
    callbacks = get_callbacks(
        monitor=args.monitor,
        patience=args.patience,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr
    )

    # Train the model
    history = train_model(
        args,
        model,
        data['X_train_scaled'],
        data['y_train'],
        data['X_val_scaled'],
        data['y_val'],
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save the model
    model.save(file_paths['best_model'])

    # Evaluate the model
    results = evaluate_model(
        model,
        data['X_test_scaled'],
        data['y_test'],
        args.threshold
    )

    # Get experiment summary
    summary_text = get_experiment_summary(args, results)

    # Save results and plots 
    save_model_summary(
        file_paths['config'],
        args,
        results,
        args.experiment_name,
        summary_text
    )

    plot_confusion_matrix(
        file_paths['confusion_matrix'],
        results['confusion_matrix'],
        summary_text
    )

    plot_confusion_matrix(
        file_paths['confusion_matrix_std'],
        results['confusion_matrix'],
        summary_text
    )

    plot_learning_curves(
        file_paths['learning_curves'],
        history,
        summary_text
    )

    plot_learning_curves(
        file_paths['learning_curves_std'],
        history,
        summary_text
    )

    # Print completion message
    print("\n" + "=" * 50)
    print(f"ARCHITECTURE SEARCH COMPLETED: {args.experiment_name}")
    print("=" * 50)
    print(f"Best layer architecture: {args.hidden_layers}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
