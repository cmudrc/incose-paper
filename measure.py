import os
import multiprocessing
import piperabm as pa

from info import *


def load_measurements(path):
    result = []
    for name in names:
        #setup = name_to_setup(name)
        measurement = pa.Measurement(
            path=path,
            name=name
        )
        result.append(measurement)
    return result

def measure(measurement):
    measurement.measure(report=True, resume=False)
    setup = name_to_setup(measurement.name)
    info = ' '
    info += f"(A: {str(setup['population'])})"
    info +=  f"(A/H: {str(setup['household_size'])})"
    if setup['impact'] is not None:
        info += f"(impact: {setup['impact'][0] + ' ' + str(setup['impact'][1])}%)"
    measurement.accessibility.save(info=info)
    measurement.travel_distance.save(info=info)

def main():
    # Report
    print(">>> measuring all models... ")

    path = os.path.dirname(os.path.realpath(__file__))
    measurements = load_measurements(path)

    # Create a pool of processes
    num_processors = os.cpu_count() - 1
    with multiprocessing.Pool(processes=num_processors) as pool:
        # Map the models to the processes
        pool.map(measure, measurements)

    # Report
    print(">>> all models have been measured successfully.\n")


if __name__ == "__main__":
    main()