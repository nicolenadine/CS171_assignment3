# CS171_assignment3
Code files related to Assignment 3

### data_exploration.ipynb
Explore data features. Fill in missing data. Check for duplicate samples. Save to new cleaned csv

### config.py: 
Handles command line argument parsing, output directory setup, and file path generation.

parse_arguments() - Handles command line arguments
setup_output_directory() - Creates the experiment directory
print_configuration() - Displays experiment config
get_file_paths() - Generates file paths

### data_processing.py: 
Manages data loading, preprocessing, class balancing, and dataset splitting.

load_data() - Loads and preprocesses data
print_class_distribution() - Displays class stats
shuffle_data() - Shuffles data
balance_dataset() - Balances class distribution
prepare_data() - Full data pipeline

### models.py: 
Contains model definition functions including:

* The standard model creation function
* The grid search model creation function
* The focal loss implementation

get_focal_loss() - Creates focal loss function
create_model() - Builds standard model
create_grid_search_model() - Builds model for grid search
setup_grid_search() - Prepares grid search

### callbacks.py: 
Contains the custom detailed progress callback and a function to generate all needed callbacks.

DetailedProgress class - Custom progress callback
get_callbacks() - Creates callback list

### training.py: 
Handles model training, evaluation, and grid search implementation.

train_model() - Main training function
run_grid_search() - Runs hyperparameter search
evaluate_model() - Evaluates on test data
get_experiment_summary() - Creates experiment summary

### utils.py: 
Provides utility functions for plotting results and saving experiment summaries.

save_model_summary() - Saves experiment details
plot_confusion_matrix() - Creates confusion matrix plot
plot_learning_curves() - Creates learning curve plots
print_completion_message() - Displays final output

### main.py: 
Orchestrates the entire workflow, calling functions from the other modules in the correct sequence.

