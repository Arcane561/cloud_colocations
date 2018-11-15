#!/usr/bin/python

import sys
from cloud_collocations.collocations import Collocation, ProcessDay


day_0 = int(sys.argv[1])
day_1 = int(sys.argv[2])

year = 2009

for day in range(day_0, day_1):
    print("processing day ", day, " ...")
    colls = ProcessDay(year, day, "/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations")
    colls.run()



