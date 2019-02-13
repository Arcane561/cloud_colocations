import cloud_colocations
from cloud_colocations.formats import modis, file_cache, caliop
from cloud_colocations.products import file_cache, FileCache
from datetime import datetime

# define directory for data 
cloud_colocations.products.file_cache = FileCache("data")
t = datetime(2011, 1, 1)
# get file name
filename = caliop.get_file_by_date(t)
# download file
output_name= caliop.download_file(filename)

from cloud_colocations.formats import Caliop01kmclay
caliop_file = Caliop01kmclay(output_name)



