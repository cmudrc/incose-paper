import os
import multiprocessing
import piperabm as pa

from info import *


def load_models(path):
    result = []
    for name in names_impacted:
        setup = name_to_setup(name)
        model = pa.Model(
            path=path,
            name=name,
            seed=setup['seed']
        )
        result.append(model)
    return result


def run_model(model):
    model.run(
        n=steps,
        step_size=step_size,
        save=True,
        save_transactions=True,
        resume=False,
        report=True
    )


def main():
    # Report
    print(">>> running impacted models... ")

    path = os.path.dirname(os.path.realpath(__file__))
    models = load_models(path)

    # Create a pool of processes
    num_processors = os.cpu_count() - 1
    with multiprocessing.Pool(processes=num_processors) as pool:
        # Map the models to the processes
        pool.map(run_model, models)

    # Report
    print(">>> all impacted models have completed their runs successfully.\n")


if __name__ == "__main__":
    main()