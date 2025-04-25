import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler


def load_data(file_path):
    """Load and preprocess the data from a CSV file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Remove any unnamed index columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Remove any additional problematic columns (adjust as needed)
    if '4' in df.columns:
        df = df.drop(columns=['4'])

    # Extract features and target
    X = df.iloc[1:, :-1]  # All rows, all columns except the last
    y = df.iloc[1:, -1]

    # Print initial class distribution
    print_class_distribution(y, "Full dataset")

    return X, y, df.columns[:-1]  # Return column names for reference


def print_class_distribution(y, dataset_name="Dataset"):
    """Print the class distribution of a dataset."""
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    total = neg_count + pos_count

    print(f"\n{dataset_name} class distribution:")
    print(f"Class 0: {neg_count} ({neg_count / total:.2%})")
    print(f"Class 1: {pos_count} ({pos_count / total:.2%})")
    print(f"Total: {total} samples")


def shuffle_data(X, y, random_state=42):
    """Shuffle the features and target together."""
    print("\nShuffling the dataset...")
    # Create a DataFrame with both X and y to shuffle them together
    combined_df = pd.concat([X, pd.Series(y, name='target')], axis=1)
    # Shuffle the combined DataFrame
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Split back into X and y
    X = combined_df.iloc[:, :-1]
    y = combined_df['target']
    print("Dataset shuffled successfully")
    return X, y


def balance_dataset(X_data, y_data, random_state=42):
    """
    Balance the dataset by downsampling the majority class to match the minority class count.
    This function keeps all positive samples and randomly selects an equal number of negative samples.
    """
    # Count the number of samples in each class
    neg_count = np.sum(y_data == 0)
    pos_count = np.sum(y_data == 1)
    total = neg_count + pos_count

    print(
        f"Original class distribution - Negative: {neg_count} ({neg_count / total:.2%}), "
        f"Positive: {pos_count} ({pos_count / total:.2%})"
    )

    # If already balanced, return original data
    if neg_count == pos_count:
        print("Data is already balanced. No changes made.")
        return X_data, y_data

    # Create a balanced dataset with 1:1 ratio (keeping all positive samples)
    target_ratio = {0: pos_count, 1: pos_count}  # Keep all positives, sample same number of negatives

    # Use RandomUnderSampler to downsample the majority class
    undersampler = RandomUnderSampler(sampling_strategy=target_ratio, random_state=random_state)
    X_balanced, y_balanced = undersampler.fit_resample(X_data, y_data)

    # Print the new distribution
    new_neg_count = np.sum(y_balanced == 0)
    new_pos_count = np.sum(y_balanced == 1)
    new_total = new_neg_count + new_pos_count

    print(
        f"Balanced class distribution - Negative: {new_neg_count} ({new_neg_count / new_total:.2%}), "
        f"Positive: {new_pos_count} ({new_pos_count / new_total:.2%})"
    )

    return X_balanced, y_balanced


def split_data(X, y, test_size=0.15, val_size=0.20, random_state=42):
    """Split data into training, validation, and test sets."""
    print("Splitting data into train/validation/test sets...")

    # First split: separate test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    # Print number of samples
    print("\nDataset splits:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler for better handling of outliers."""
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_data(args):
    """Complete data preparation pipeline."""
    # Load data
    X, y, feature_names = load_data(args.data_file)

    # Shuffle data
    X, y = shuffle_data(X, y, random_state=args.random_state)

    # Balance dataset
    X_balanced, y_balanced = balance_dataset(X, y, random_state=args.random_state)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_balanced, y_balanced,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    # Return all necessary components
    data = {
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'input_dim': X_train.shape[1]
    }

    return data