"""All this stuff is to get the version from setup.py."""

from pkg_resources import get_distribution, DistributionNotFound
import os.path

try:
    _dist = get_distribution('pyprf')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'pyprf')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Version information not found. Please install this project \
                   through pip.)'
else:
    __version__ = _dist.version
