import tensorflow as tf
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


def get_focal_loss(gamma, alpha):
    """
    Create a focal loss function with specified parameters.

    Parameters:
    gamma (float): Higher values focus more on hard examples.
    alpha (float): Balancing factor for positive class.

    Returns:
    function: Compiled focal loss function ready to use in model.compile()
    """

    # Experiment with focal_loss
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = tf.keras.backend.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.sum(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1)) \
            - tf.keras.backend.sum((1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(1. - pt_0))

    return focal_loss_fixed


def create_model(args, input_dim, hidden_layers, dropout_rates, learning_rate=0.0005,
                 weight_decay=0.0, loss='binary_crossentropy', focal_gamma=1.0, focal_alpha=0.5):
    """Create a neural network model for binary classification."""
    print("Creating model with the following architecture:")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Dropout rates: {dropout_rates}")
    print(f"Creating model with input dimension: {input_dim}")
    print(f"Output activation: {args.output_activation}")

    if len(hidden_layers) != len(dropout_rates):
        raise ValueError("The number of hidden layers must match the number of dropout rates")

    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rates[0]))

    # Hidden layers
    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation='relu'))
        model.add(Dropout(dropout_rates[i]))

    # Output layer
    if args.output_activation == 'softmax':
        # For softmax with binary classification, need 2 output nodes
        model.add(Dense(2, activation='softmax'))
    else:
        # Default sigmoid for binary classification
        model.add(Dense(1, activation='sigmoid'))
        # Use specified loss function

    if loss == 'focal_loss':
        loss_fn = get_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        loss_fn = loss

    print(f"Using loss function: {loss_fn}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Weight decay (L2 regularization): {weight_decay}")

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def create_grid_search_model(hidden_layers=[128, 64], dropout_rates=[0.2, 0.1], learning_rate=0.001, input_dim=10):
    """
    Create a model for grid search optimization.

    This function is designed to be used with KerasClassifier and GridSearchCV.

    Parameters:
    hidden_layers (list): List of integers specifying the number of units in each hidden layer
    dropout_rates (list): List of floats specifying the dropout rate for each layer
    learning_rate (float): Initial learning rate for the Adam optimizer
    input_dim (int): Input dimension (number of features)

    Returns:
    model: Compiled Keras model
    """
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[0]))

    # Hidden layers
    for units, dropout_rate in zip(hidden_layers[1:], dropout_rates[1:]):
        model.add(Dense(units, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

    # Input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[0]))

    # Hidden layers
    for units, dropout_rate in zip(hidden_layers[1:], dropout_rates[1:]):
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def setup_grid_search(input_dim):
    """
    Set up grid search for hyperparameter optimization.

    Parameters:
    input_dim (int): Input dimension for the model

    Returns:
    dict: Parameter grid for GridSearchCV
    """

    # Create a function that creates models with the fixed input dimension
    def model_builder(hidden_layers=[128, 64], dropout_rates=[0.2, 0.1], learning_rate=0.001):
        # Use partial to bind the input_dim to the model creation function
        return create_grid_search_model(
            hidden_layers=hidden_layers,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
            input_dim=input_dim
        )

    # Create a KerasClassifier wrapper
    from scikeras.wrappers import KerasClassifier

    keras_clf = KerasClassifier(
        model=model_builder,
        verbose=0,
        epochs=50,
        batch_size=64
    )

    # Parameter grid
    param_grid = {
        'model__hidden_layers': [[128, 64], [256, 128, 64], [128, 64, 32]],
        'model__dropout_rates': [[0.2, 0.2], [0.3, 0.1, 0.1], [0.2, 0.1, 0.1]],
        'model__learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64],
        'epochs': [50, 100],
    }

    return keras_clf, param_grid