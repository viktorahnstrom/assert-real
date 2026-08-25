"""
Per-IP rate limiting via slowapi.

Import ``limiter`` and the rate-limit strings wherever a decorator is
needed.  The limiter is registered on the FastAPI app in main.py.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

RATE_LIMIT_ANALYSES = os.getenv("RATE_LIMIT_ANALYSES", "10/hour")
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "60/hour")
