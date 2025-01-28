import matplotlib.pyplot as plt

def plot_data(data, column):
    # Ensure the column exists in the dataset
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data[column], label=column, color='blue')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(f'{column} Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
