"""
log tool, depend on logging.
later if I want to change another package to log message, I don't need to modify everywhere.
"""

import logging

logger = logging.getLogger()


def info(msg, *args, **kwargs):
    """
    Log 'msg % args' with severity 'INFO'.

    To pass exception information, use the keyword argument exc_info with
    a true value, e.g.

    logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
    """
    logger.info(msg, args, **kwargs)
