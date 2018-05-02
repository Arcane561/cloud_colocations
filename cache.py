import os

cache_folder = "cache"
if not os.path.isdir(cache_folder):
    os.path.mkdir(cache_folder)

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



