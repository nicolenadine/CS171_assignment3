import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from models import create_model, setup_grid_search
from callbacks import get_callbacks
from tensorflow.keras.utils import to_categorical


def train_model(args, model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, callbacks=None):
    """
    Train the model using the provided data.

    Parameters:
    model: Compiled Keras model
    X_train: Training features
    y_train: Training targets
    X_val: Validation features
    y_val: Validation targets
    batch_size (int): Batch size for training
    epochs (int): Maximum number of epochs to train
    callbacks (list): List of callbacks for training

    Returns:
    history: Training history
    """

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Make sure y values are integers for sparse_categorical_crossentropy
    # For softmax activation, convert to one-hot encoding
    if args.output_activation == 'softmax':
        # Make sure y values are integers
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)

        # Convert to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

        print("Using one-hot encoded targets with categorical_crossentropy")

    print(f"Training model with batch_size={batch_size}, epochs={epochs}")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history


def run_grid_search(data, args, custom_model=None, custom_param_grid=None):
    """
    Run grid search for hyperparameter tuning.

    Parameters:
    data (dict): Dictionary containing the data
    args: Command-line arguments
    custom_model: Optional custom model for specialized grid searches
    custom_param_grid: Optional custom parameter grid for specialized grid searches

    Returns:
    best_model: Best model found by grid search
    grid_result: Grid search results
    """
    print("Setting up grid search...")

    # Use custom model and parameters if provided, otherwise use default
    if custom_model is not None and custom_param_grid is not None:
        print("Using custom model and parameter grid for specialized search")
        keras_clf = custom_model
        param_grid = custom_param_grid
    else:
        # Use the default setup from models.py
        from models import setup_grid_search
        keras_clf, param_grid = setup_grid_search(data['input_dim'])

    # GridSearchCV setup
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(
        estimator=keras_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=1  # Force single-threaded to avoid potential issues with TensorFlow
    )

    # Fit Grid Search
    print("Running grid search (this may take a while)...")
    grid_result = grid.fit(data['X_train_scaled'], data['y_train'])

    # Print best results
    print("Best parameters found: ", grid_result.best_params_)
    print("Best accuracy found: ", grid_result.best_score_)

    # Get best model
    best_model = grid_result.best_estimator_

    return best_model, grid_result


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the model on the test set.
    """
    print("Evaluating model on test set...")

    # Check if model has softmax activation by examining the output layer
    # Look at the last layer's configuration instead of directly accessing output_shape
    last_layer = model.layers[-1]
    output_units = last_layer.get_config()['units']
    has_softmax = output_units == 2  # If 2 units, we're using softmax for binary classification

    print(f"Model has {'softmax' if has_softmax else 'sigmoid'} output layer")

    if has_softmax:
        # Save original test labels for classification report
        y_test_original = y_test.copy() if isinstance(y_test, np.ndarray) else y_test.values.copy()

        # Convert target to one-hot for categorical crossentropy
        from tensorflow.keras.utils import to_categorical
        y_test_cat = to_categorical(y_test.astype(np.int32), num_classes=2)

        # Get model metrics with one-hot encoded targets
        test_loss, test_acc, test_auc = model.evaluate(X_test, y_test_cat)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        # Get predictions - for softmax, it's class probabilities for both classes
        y_pred_prob = model.predict(X_test)
        # Take the probability of class 1
        y_pred_class1_prob = y_pred_prob[:, 1]
        # Apply threshold to get binary predictions
        y_pred = (y_pred_class1_prob > threshold).astype(int)

        # Use original non-one-hot labels for metrics
        y_test = y_test_original
    else:
        # Get model metrics
        test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        # Get predictions
        print(f"Generating predictions with threshold={threshold}")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > threshold).astype(int).flatten()

    # Print classification report
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Try different thresholds
    print("\nEvaluating different thresholds:")
    threshold_results = {}
    thresholds = [0.4, 0.5, 0.55, 0.6]

    for t in thresholds:
        if has_softmax:
            y_pred_t = (y_pred_class1_prob > t).astype(int)
        else:
            y_pred_t = (y_pred_prob > t).astype(int).flatten()

        print(f"\nThreshold: {t}")
        threshold_report = classification_report(y_test, y_pred_t)
        print(threshold_report)
        threshold_results[t] = {
            'report': threshold_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred_t)
        }

    return {
        'accuracy': test_acc,
        'auc': test_auc,
        'loss': test_loss,
        'predictions': y_pred_prob if not has_softmax else y_pred_class1_prob,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'threshold_results': threshold_results
    }


def get_experiment_summary(args, results):
    """
    Create a text summary of the experiment.

    Parameters:
    args: Command-line arguments
    results (dict): Evaluation results

    Returns:
    str: Text summary of the experiment
    """
    # Add focal loss parameters if using focal loss
    focal_loss_text = ""
    if args.loss == 'focal_loss':
        focal_loss_text = f"Focal γ: {args.focal_gamma}, α: {args.focal_alpha}\n"

    experiment_name = args.experiment_name if args.experiment_name else "Experiment"

    hyperparam_text = (
        f"{experiment_name}\n\n"
        f"Model Config:\n"
        f"Learning Rate: {args.learning_rate}\n"
        f"Weight Decay (L2): {args.weight_decay}\n"
        f"Hidden Layers: {args.hidden_layers}\n"
        f"Dropout Rates: {args.dropout_rates}\n"
        f"Batch Size: {args.batch_size}\n"
        f"Loss: {args.loss}\n"
        f"{focal_loss_text}"
        f"Early Stop: {args.patience} epochs on {args.monitor}\n"
        f"LR Reduction: factor={args.lr_factor}, patience={args.lr_patience}\n"
        f"Threshold: {args.threshold}\n\n"
        f"Results:\nTest Acc: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}"
    )

    return hyperparam_text