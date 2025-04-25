import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_model_summary(file_path, args, results, experiment_name, summary_text):
    """
    Save a detailed summary of the experiment to a text file.

    Parameters:
    file_path (str): Path to save the summary
    args: Command-line arguments
    results (dict): Evaluation results
    experiment_name (str): Name of the experiment
    summary_text (str): Text summary of the experiment
    """
    with open(file_path, 'w') as f:
        f.write(f"{experiment_name} - Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("CONFIGURATION PARAMETERS:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        f.write(summary_text + "\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n\nTraining ended at: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("\nFiles saved:\n")
        f.write(f"- Summary: {file_path}\n")
        f.write(f"- Model: {os.path.join(os.path.dirname(file_path), 'best_model.keras')}\n")

    print(f"Saved experiment summary to '{file_path}'")


def plot_confusion_matrix(output_path, cm, summary_text):
    """
    Plot and save confusion matrix with hyperparameter information.

    Parameters:
    output_path (str): Path to save the plot
    cm: Confusion matrix
    summary_text (str): Text to display alongside the confusion matrix
    """
    print("Creating confusion matrix...")
    plt.figure(figsize=(10, 8))

    # Main plot area for confusion matrix (left side)
    plt.subplot(1, 5, (1, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Text area for hyperparameters (right side)
    plt.subplot(1, 5, (4, 5))
    plt.axis('off')
    plt.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved confusion matrix to '{output_path}'")


def plot_learning_curves(output_path, history, summary_text):
    """
    Plot and save learning curves with hyperparameter information.

    Parameters:
    output_path (str): Path to save the plot
    history: Training history
    summary_text (str): Text to display alongside the learning curves
    """
    print("Creating learning curves...")
    plt.figure(figsize=(15, 8))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # AUC plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    # Text area for hyperparameters
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.text(0, 0.5, summary_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved learning curves to '{output_path}'")


def print_completion_message(experiment_name, results):
    """
    Print a completion message with key results.

    Parameters:
    experiment_name (str): Name of the experiment
    results (dict): Evaluation results
    """
    print("\n" + "=" * 50)
    print(f"EXPERIMENT COMPLETED: {experiment_name}")
    print("=" * 50)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")
    print("=" * 50)