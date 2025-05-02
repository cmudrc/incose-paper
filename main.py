from datetime import datetime
import csv
import os

import piperabm as pa

from create import main as create
from run import main as run
from impact import main as impact
from run_impacted import main as run_impacted

from info import names, name_to_setup


def main():
    start_time = datetime.now()
    print("start time: ", start_time)
    
    
    create()
    run()
    impact()
    run_impacted()

    
    end_time = datetime.now()
    print("end time: ", end_time)
    duration = end_time - start_time
    print("duration: ", duration)


if __name__ == "__main__":
    main()