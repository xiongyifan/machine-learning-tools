"""Machine Learning Tools For Python"""
from os.path import dirname, realpath
from logging.config import fileConfig

from mlt import stats
from mlt import file
from mlt import models
from mlt import preprocessing
from mlt import utils

fileConfig('{}/logging.ini'.format(dirname(realpath(__file__))))  # initial logger
