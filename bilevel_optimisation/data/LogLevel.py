import logging
from enum import Enum

class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @staticmethod
    def from_string(log_level_str: str) -> int:
        try:
            return LogLevel[log_level_str.upper()].value
        except KeyError:
            raise ValueError('Invalid log level {:s}'.format(log_level_str))

