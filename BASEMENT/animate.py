import os
import multiprocessing
import piperabm as pa

from info import *


def load_models(path):
    result = []
    for name in names:
        model = pa.Model(
            name=name,
            path=path
        )
        result.append(model)
    return result


def animate(model):
    model.animate()


def main():
    # Report
    print(">>> animating all results...")

    path = os.path.dirname(os.path.realpath(__file__))
    models = load_models(path)

    # Create a pool of processes
    num_processors = os.cpu_count() - 1
    with multiprocessing.Pool(processes=num_processors) as pool:
        # Map the models to the processes
        pool.map(animate, models)

    # Report
    print(">>> all results have been animated successfully.\n")


if __name__ == "__main__":
    main()