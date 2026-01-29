import os
import re

FILENAME_RE = re.compile(r"^(?P<vehicle>[1-8])_10X(?P<camera>[1-9])(?P<series>\d{3,})\.jpg$", re.IGNORECASE)


def parse_filename(path: str):
    """Parse filename into (vehicle_id, camera_id, series)."""
    name = os.path.basename(path)
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid filename: {name}")
    vehicle_id = int(m.group("vehicle"))
    camera_id = int(m.group("camera"))
    series = int(m.group("series"))
    return vehicle_id, camera_id, series
