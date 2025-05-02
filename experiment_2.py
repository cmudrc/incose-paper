import os
import piperabm as pa
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))
names = [
    '0_0_0_0', # control
    '0_0_0_2', # critical 15%
    '0_0_0_5', # random 15%
]
titles = [
    'Control',
    '15% Critical Impact',
    '15% Random Impact',
]

results = []

for i, name in enumerate(names):
    measurement = pa.Measurement(
        path=path,
        name=name
    )
    measurement.load()
    result = {
        'title': titles[i],
        'accessibility': measurement.accessibility(),  # list
        'average accessibility': measurement.accessibility.average(),
        'travel distance': measurement.travel_distance(),  # list
        'average travel distance': measurement.travel_distance.average(),
        'delta_times': measurement.filter_times()  # list
    }
    results.append(result)

def show(results):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # Variables to determine the global min and max for x and y axes
    min_time, max_time = float('inf'), float('-inf')
    min_access, max_access = 0, 1  # Assuming normalized accessibility between 0 and 1
    min_travel, max_travel = float('inf'), float('-inf')

    # Calculate global min and max for axes
    for result in results:
        min_time = min(min_time, min(result['delta_times']))
        max_time = max(max_time, max(result['delta_times']))
        min_travel = min(min_travel, min(result['travel distance']))
        max_travel = max(max_travel, max(result['travel distance']))

    # Convert seconds to days
    min_time /= 86400
    max_time /= 86400

    # Plotting and setting the same axis ranges for all subplots
    for i, result in enumerate(results):
        times_in_days = [time / 86400 for time in result['delta_times']]  # Convert each time point to days

        # Accessibility plot in the first row
        axes[0, i].plot(times_in_days, result['accessibility'])
        axes[0, i].set_title(f'{result["title"]}', fontweight='bold')
        axes[0, i].set_xlabel('Time (days)', fontweight='bold')
        axes[0, i].set_ylabel('Accessibility', fontweight='bold')
        axes[0, i].set_xlim(min_time, max_time)
        axes[0, i].set_ylim(min_access, max_access)
        axes[0, i].text(0.62, 0.9, f'Average: {result["average accessibility"]:.3f}',
                        transform=axes[0, i].transAxes, fontsize=12, fontweight='bold')
        
        # Travel distance plot in the second row
        axes[1, i].plot(times_in_days, result['travel distance'])
        axes[1, i].set_title(f'{result["title"]}', fontweight='bold')
        axes[1, i].set_xlabel('Time (days)', fontweight='bold')
        axes[1, i].set_ylabel('Travel Distance (m)', fontweight='bold')
        axes[1, i].set_xlim(min_time, max_time)
        axes[1, i].set_ylim(min_travel, max_travel)
        axes[1, i].text(0.64, 0.9, f'Average: {result["average travel distance"]:.0f}',
                        transform=axes[1, i].transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plots

show(results)
#for result in results:
#    print(result['average accessibility'])