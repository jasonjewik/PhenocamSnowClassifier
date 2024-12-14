import configparser
import os

PROJECT_ROOT_DIR = os.getcwd()

ini_canonical_path = os.path.join(PROJECT_ROOT_DIR, "setup.ini")
CONFIG = configparser.ConfigParser()
CONFIG.read(ini_canonical_path)
