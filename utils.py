import os

def ensure_extension(name, ext):
    if ext[0] != ".":
        ext = "." + ext
    name_base, name_ext = os.path.splitext(name)
    if name_ext != ext:
        return name + ext
    else:
        return name

