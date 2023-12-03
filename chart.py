import matplotlib.pyplot as plt
import numpy as np

def drawChart(actual_prices, predicted_prices_algo1, predicted_prices_algo2, numOfValues = 100):
    num_data_points = len(actual_prices)
    print("Actual Prices:", num_data_points)
    print("Actual Prices:", actual_prices[:5])
    print("Predicted Prices (Algorithm 1):", predicted_prices_algo1[:5])
    print("Predicted Prices (Algorithm 2):", predicted_prices_algo2[:5])

    # Generate a sample dataset
    data_points = np.arange(num_data_points)[:numOfValues]

    # Plotting the lines
    plt.plot(data_points, actual_prices[:numOfValues], label='Actual Prices', marker='o')
    plt.plot(data_points, predicted_prices_algo1[:numOfValues], label='Predicted (Algorithm 1)', marker='x')
    plt.plot(data_points, predicted_prices_algo2[:numOfValues], label='Predicted (Algorithm 2)', marker='s')

    # Adding labels and title
    plt.xlabel('Data Points')
    plt.ylabel('House Prices')
    plt.title('Comparison of Predictions from Two Algorithms')
    plt.legend()

    # Show the plot
    plt.show()


# np.random.seed(42)

#     # Generate fake data
# num_data_points = 1000

# actual_prices = np.random.uniform(100000, 500000, num_data_points)
# noise = np.random.normal(0, 50000, num_data_points)

# predicted_prices_algo1 = actual_prices + noise
# predicted_prices_algo2 = actual_prices + 2 * noise

# drawChart(actual_prices, predicted_prices_algo1, predicted_prices_algo2, 50)