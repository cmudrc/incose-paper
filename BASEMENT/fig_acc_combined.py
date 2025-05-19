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
    'resource': measurement.accessibility(),  # list
    'delta_times': measurement.filter_times()  # list
}

def show(result):
    plt.figure(figsize=(10, 6))
    plt.plot(result['delta_times'], result['resource'], label='Resources Accessibility', color='blue')
    plt.title(result['title'], fontweight='bold')
    plt.xlabel('Time (days)', fontweight='bold')
    plt.ylabel('Accessibility', fontweight='bold')
    plt.ylim(0, 1)  # Set y-axis range between 0 and 1
    #plt.legend()
    plt.grid(True)
    plt.show()

result['delta_times'] = [time / 86400 for time in result['delta_times']]  # Convert each time point to days

# Plotting the adjusted values
show(result)
