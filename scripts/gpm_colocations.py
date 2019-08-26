################################################################################
# Extract GPM colocations for a given range of days.
################################################################################

#
# Read command line args.
#

import argparse
parser = argparse.ArgumentParser(prog = "gpm_colocations",
                                 description = "Extract radar/radiometer "
                                 "colocations from the GPM core observatory.")
parser.add_argument('start_day', metavar = 'start_day', type = int, nargs = 1)
parser.add_argument('end_day',   metavar = 'end_day',   type = int, nargs = 1)
parser.add_argument('year',      metavar = 'year',      type = int, nargs = 1)
parser.add_argument('path',      metavar = 'path',      type = str, nargs = 1)
args = parser.parse_args()
start_day = args.start_day[0]
end_day   = args.end_day[0]
year      = args.year[0]
path      = args.path[0]

#
# Actual code.
#

from cloud_colocations.colocations import Colocations, ProcessDay
from cloud_colocations.colocations.formats import GPMGMI1C, GPMCMB, GPM
from cloud_colocations.colocations.products import set_cache

from datetime import datetime
t = datetime(2015, 1, 1)

for i in range(start_day, end_day):
    pd = ProcessDay("cloud_colocations",
                    year, i,
                    path,
                    GPMGMI1C,
                    GPM)
    pd.run()
