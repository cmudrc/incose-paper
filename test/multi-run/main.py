"""
Run the simulation pipleline.
"""

from datetime import datetime

from create import main as create
from run import main as run
from impact import main as impact
from run_impacted import main as run_impacted


def main():
    start_time = datetime.now()
    print("start time: ", start_time)
    
    create()  # Create unimpacted models
    run()  # Run unimpacted models
    impact()  # Create impacted models
    run_impacted()  # Run impacted models
    
    end_time = datetime.now()
    print("end time: ", end_time)
    duration = end_time - start_time
    print("duration: ", duration)


if __name__ == "__main__":
    main()