import cloud_colocations
from cloud_colocations.formats import file_cache
from cloud_colocations.products import file_cache, FileCache, modis, caliop, cloudsat
from datetime import datetime

#
# Caliop
#

# define directory for data 
cloud_colocations.products.file_cache = FileCache("data")
t = datetime(2013, 2, 1)
# get file name
filename = caliop.get_file_by_date(t)
# download file
output_name= caliop.download_file(filename)

#
# Cloudsat 
#

# define directory for data 
cloud_colocations.products.file_cache = FileCache("data")
t = datetime(2013, 2, 1)
# get file name
filename = cloudsat.get_file_by_date(t)
# download file
output_name= cloudsat.download_file(filename)
