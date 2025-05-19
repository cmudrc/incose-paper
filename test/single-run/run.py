"""
Run a single simulation of the model with a simple setup.
"""
import os

import piperabm as pa


path = os.path.dirname(os.path.realpath(__file__))
model = pa.Model(path=path, seed=2)

model.infrastructure.generate(
    homes_num=5,
    grid_size=[10, 10],
    grid_num=[5, 5]
)
model.society.generate(
    num=10,
    gini_index=0.4
)
model.run(n=100, step_size=7200, save=True, save_transactions=True, report=True)