import os
import sys
from pathlib import Path


def run_path_setup():
    script_path = os.getcwd()
    sys.path.append(script_path)
    sys.path.append(str(Path(script_path).parents[0]))
