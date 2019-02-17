#!/usr/bin/python

import sys
from cloud_colocations.colocations import Colocation, ProcessDay


day_0 = int(sys.argv[1])
day_1 = int(sys.argv[2])

year = 2009

for day in range(day_0, day_1):
    print("processing day ", day, " ...")
    cols = ProcessDay(year, day,
                      "/home/simonpf/Dendrite/UserAreas/Simon/cloud_colocations",
                      dn = 100)
    cols.run()



