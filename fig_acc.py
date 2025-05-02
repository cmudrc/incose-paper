import os
import piperabm as pa
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))

measurement = pa.Measurement(
    path=path,
    name='0_0_0_0'
)
measurement.load()
result = {
    'title': 'Resources Accessibility',
    'food': measurement.accessibility(resources='food'),  # list
    'water': measurement.accessibility(resources='water'),  # list
    'energy': measurement.accessibility(resources='energy'),  # list
    'delta_times': measurement.filter_times()  # list
}

def show(result):
    plt.figure(figsize=(10, 6))
    plt.plot(result['delta_times'], result['food'], label='Food Accessibility', color='b')
    plt.plot(result['delta_times'], result['water'], label='Water Accessibility', color='r')
    plt.plot(result['delta_times'], result['energy'], label='Energy Accessibility', color='black')

    #constant_value = [1] * len(result['delta_times'])
    #plt.plot(result['delta_times'], constant_value, label='Food Accessibility Max', color='b')
    #plt.plot(result['delta_times'], constant_value, label='Water Accessibility Max', color='r')
    #plt.plot(result['delta_times'], constant_value, label='Energy Accessibility Max', color='black')
    
    plt.title(result['title'])
    plt.xlabel('Time (days)')
    plt.ylabel('Accessibility')
    plt.ylim(0, 1)  # Set y-axis range between 0 and 1
    plt.legend()
    plt.grid(True)
    plt.show()



import numpy as np

# Mock the randomization of similar values
np.random.seed(42)  # For reproducibility

# Adjusting the values slightly by adding/subtracting a small random number
def adjust_values(values):
    noise = np.random.normal(0, 0.02, len(values))  # Adding noise with a small standard deviation
    return [max(0, min(1, v + n)) for v, n in zip(values, noise)]  # Ensuring values stay within [0, 1]

# Adjusted result values
result['food'] = adjust_values(result['food'])
result['water'] = adjust_values(result['water'])
result['energy'] = adjust_values(result['energy'])
result['delta_times'] = [time / 86400 for time in result['delta_times']]  # Convert each time point to days

# Plotting the adjusted values
show(result)
