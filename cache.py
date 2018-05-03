import os
import glob

cache_folder = "cache"
if not os.path.isdir(cache_folder):
    os.makedirs(cache_folder)

def get_cached_files(subfolder, pattern):
    path = os.path.join(cache_folder, subfolder, pattern)
    print(path)
    files = glob.glob(path)
    return [os.path.basename(f) for f in files]

class CachedFile():
   def __init__(self, subfolder, filename):
       self.folder = os.path.join(cache_folder, subfolder)
       self.file = os.path.join(self.folder, filename)

       if not os.path.exists(self.folder):
           os.makedirs(self.folder)

       if not os.path.exists(self.file):
           self.get_file(self.file)

   def remove(self):
       if os.path.exists(self.file):
           os.remove(self.file)



