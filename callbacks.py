import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class DetailedProgress(tf.keras.callbacks.Callback):
    """
    Custom callback that provides detailed progress information during training.
    """

    def on_epoch_end(self, epoch, logs=None):
        """Print detailed metrics at the end of each epoch."""
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']} completed")
        print(f"Training metrics: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}")
        print(f"Validation metrics: Loss={logs['val_loss']:.4f}, Accuracy={logs['val_accuracy']:.4f}")
        print(f"Current learning rate: {tf.keras.backend.get_value(self.model.optimizer.learning_rate):.8f}")


def get_callbacks(monitor='val_auc', patience=5, lr_factor=0.2, lr_patience=5, min_lr=0.00001):
    """
    Create a list of callbacks for training.

    Parameters:
    monitor (str): Metric to monitor for early stopping and learning rate reduction
    patience (int): Number of epochs with no improvement after which training will stop
    lr_factor (float): Factor by which the learning rate will be reduced
    lr_patience (int): Number of epochs with no improvement after which learning rate will be reduced
    min_lr (float): Minimum learning rate

    Returns:
    list: List of callbacks
    """
    print(f"Setting up callbacks with monitor={monitor}, patience={patience}")

    # Set the correct mode based on the monitored metric
    mode = 'max' if 'auc' in monitor or 'accuracy' in monitor else 'min'

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=1
    )

    # Custom progress callback
    detailed_progress = DetailedProgress()

    return [early_stopping, reduce_lr, detailed_progress]