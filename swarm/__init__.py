"""project-swarm — top-level package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("project-swarm")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
