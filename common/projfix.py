"""Import before rasterio/pyproj.

On machines with PostgreSQL/PostGIS installed, a global PROJ_LIB often points at
an older proj.db and every CRS lookup fails. Point PROJ at the database bundled
with the rasterio/pyproj wheels instead. Harmless everywhere else.
"""
import os
import sysconfig

_sp = sysconfig.get_paths()["purelib"]
for _p in (os.path.join(_sp, "rasterio", "proj_data"),
           os.path.join(_sp, "pyproj", "proj_dir", "share", "proj")):
    if os.path.exists(os.path.join(_p, "proj.db")):
        os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"] = _p
        break
