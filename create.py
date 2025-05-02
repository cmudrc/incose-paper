import os
import piperabm as pa

from info import *


def main():
    # Report
    print(">>> creating models...")

    path = os.path.dirname(os.path.realpath(__file__))
    for name in names_unimpacted:
        setup = name_to_setup(name)
        homes_num = calculate_homes_num(setup['population'], setup['household_size'])
            
        # Create model
        model = pa.Model(
            path=path,
            seed=setup['seed'],
            prices=prices,
            name=name
        )

        # Create infrastructure
        model.infrastructure.coeff_usage = infrastructure_coeff_usage
        market_resources = {}
        for name in average_resources:
            market_resources[name] = average_resources[name] * setup['population'] * market_resource_factor
        model.infrastructure.add_market(
            name='market',
            id=0,
            pos=market_pos,
            resources=market_resources
        )
        model.infrastructure.generate(
            homes_num=homes_num,
            grid_size=grid_size,
            grid_num=grid_num,
            imperfection_percentage=initial_imperfection
        )

        # Create society
        model.society.max_time_outside = max_time_outside
        model.society.activity_cycle = activity_cycle
        model.society.neighbor_radius = neighbor_radius
        model.society.average_income = average_income
        model.society.transportation_resource_rates = transportation_resource_rates
        model.society.idle_resource_rates = idle_resource_rates
        model.society.speed = speed
        model.society.generate(
            num=setup['population'],
            gini_index=gini_index,
            average_resources=average_resources,
            average_balance=average_balance
        )
        model.save_initial()
            
        # Report
        '''print("name: ", model.name)
        print(model.infrastructure.stat)
        print(model.society.stat)
        '''
        #model.infrastructure.show()

    # Report
    print(">>> models created successfully.\n")


if __name__ == "__main__":
    main()