# Program to calculate the total time and price for going from one city to another
# based on the distance and speed of the vehicle and plot the results from a data file.
# import matplotlib.pyplot as plt 
# import numpy as np
# def calculate_time_and_price(distance, speed, price_per_km):
#     """
#     Calculate the total time and price for a trip.
    
#     Parameters:
#     distance (float): Distance to travel in kilometers.
#     speed (float): Speed of the vehicle in km/h.
#     price_per_km (float): Price per kilometer in currency units.
    
#     Returns:
#     tuple: Total time in hours and total price in currency units.
#     """
#     time = distance / speed
#     price = distance * price_per_km
#     return time, price
# def plot_results(data):
#     """ Plot the results from the data file.
#     Parameters: 
#     data (list of tuples): Each tuple contains (distance, speed, price_per_km).
#     """
#     distances = [d[0] for d in data]
#     speeds = [d[1] for d in data]
#     prices = [d[2] for d in data]
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(distances, speeds, marker='o')
#     plt.title('Distance vs Speed')
#     plt.xlabel('Distance (km)')
#     plt.ylabel('Speed (km/h)')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(distances, prices, marker='o', color='orange')
#     plt.title('Distance vs Price')
#     plt.xlabel('Distance (km)')
#     plt.ylabel('Price (currency units)')
    
#     plt.tight_layout()
#     plt.show()
# def main():
#     # Example data: (distance, speed, price_per_km)
#     data = [
#         (100, 60, 0.5),
#         (200, 80, 0.4),
#         (300, 100, 0.3),
#         (400, 120, 0.2)
#     ]
    
#     results = []
#     for distance, speed, price_per_km in data:
#         time, price = calculate_time_and_price(distance, speed, price_per_km)
#         results.append((distance, speed, price))
#         print(f"Distance: {distance} km, Speed: {speed} km/h, Time: {time:.2f} hours, Price: {price:.2f} currency units")
    
#     plot_results(results)
# if __name__ == "__main__":
#     main()
# This code calculates the time and price for trips based on given distances, speeds, and prices per kilometer.
# It also plots the results for visual analysis.
# The code is structured to be modular, allowing for easy updates and modifications.


# Program 2
# Performing hypothesis tests using Pyhton's SciPy library
import numpy as np
from scipy import stats 
def perform_hypothesis_test(data1, data2, alpha=0.05):
    """
    Perform a two-sample t-test to compare the means of two independent samples.
    
    Parameters:
    data1 (list): First sample data.
    data2 (list): Second sample data.
    alpha (float): Significance level for the test.
    
    Returns:
    tuple: t-statistic, p-value, and whether to reject the null hypothesis.
    """
    t_statistic, p_value = stats.ttest_ind(data1, data2)
    reject_null = p_value < alpha
    return t_statistic, p_value, reject_null
def main():
    # Example data for two independent samples
    sample1 = [12, 15, 14, 10, 18, 20, 22]
    sample2 = [22, 25, 24, 20, 30, 28, 26]
    
    t_statistic, p_value, reject_null = perform_hypothesis_test(sample1, sample2)
    
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if reject_null:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")
if __name__ == "__main__":
    main()
# This code performs a two-sample t-test to compare the means of two independent samples.
# It uses the SciPy library to calculate the t-statistic and p-value, and determines
# whether to reject the null hypothesis based on a specified significance level.
# The main function demonstrates the usage with example data.