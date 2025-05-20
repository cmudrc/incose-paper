"""
Run a single simulation of the model with a simple setup.
"""
import os

import piperabm as pa

# Setup model
path = os.path.dirname(os.path.realpath(__file__))
model = pa.Model(path=path, seed=2)

# Setup infrastructure
model.infrastructure.generate(
    homes_num=5,
    grid_size=[10, 10],
    grid_num=[5, 5]
)

# Setup society
model.society.generate(
    num=10,
    gini_index=0.4
)

# Run the model
model.run(n=100, step_size=7200, save=True, save_transactions=True, report=True)

# Measure
measurement = pa.Measurement(path=path)
measurement.measure(report=True, resume=False)
measurement.accessibility.save()
measurement.travel_distance.save()