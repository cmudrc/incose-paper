import os
from copy import deepcopy
import piperabm as pa

from info import *


def impact_models(path):
    for name in names_unimpacted:
        model = pa.Model(
            path=path,
            name=name
        )
        impacted_setups = unimpacted_to_impacteds(name)
        for impacted_setup in impacted_setups:
            model_impacted = deepcopy(model)
            model_impacted.load_final()
            if impacted_setup['impact'][0] == 'critical':
                edges = model_impacted.infrastructure.top_degraded_edges(percent=impacted_setup['impact'][1])
            elif impacted_setup['impact'][0] == 'random':
                edges = model_impacted.infrastructure.random_edges(percent=impacted_setup['impact'][1])
            else:
                raise ValueError("impact type not recognized")
            model_impacted.load_initial()
            model_impacted.infrastructure.impact(edges=edges)
            model_impacted.name = setup_to_name(impacted_setup)
            model_impacted.save_initial()
                
                
def main():
    # Report
    print(">>> impacting models...")

    path = os.path.dirname(os.path.realpath(__file__))
    impact_models(path)

    # Report
    print(">>> models impacted successfully.\n")


if __name__ == "__main__":
    main()