import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

from src.data_preprocessing import load_data, preprocess_data
from src.data_visualization import plot_data
from src.model_training import create_sequences, train_model
from src.model_evaluation import evaluate_model

# Load and preprocess data
index_data, index_info, index_processed = load_data()
processed_data = preprocess_data(index_processed)

# Visualize data
plot_data(processed_data, 'CloseUSD')


# Prepare training and testing datasets
sequence_length = 60
X, y = create_sequences(processed_data['Normalized_Column'].values, sequence_length)
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

# Train model
model = train_model(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)
