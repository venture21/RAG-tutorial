"""Block test connections while preserving the standard library's socketpair.

Windows implements socketpair with a private loopback TCP connection. Only that
synchronous bootstrap may connect; ordinary loopback and external calls fail.
"""
from contextlib import contextmanager, ExitStack
from ipaddress import ip_address
import socket
from threading import local
from unittest.mock import patch

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_socketpair = socket.socketpair
_bootstrap = local()


def _allowed_socketpair_address(address):
    if not getattr(_bootstrap, "active", False):
        return False
    try:
        return ip_address(address[0]).is_loopback
    except (ValueError, TypeError, IndexError):
        return False


def _guarded_connect(sock, address):
    if _allowed_socketpair_address(address):
        return _connect(sock, address)
    raise AssertionError("Offline checks forbid network access")


def _guarded_connect_ex(sock, address):
    if _allowed_socketpair_address(address):
        return _connect_ex(sock, address)
    raise AssertionError("Offline checks forbid network access")


def _guarded_socketpair(*args, **kwargs):
    previous = getattr(_bootstrap, "active", False)
    _bootstrap.active = True
    try:
        return _socketpair(*args, **kwargs)
    finally:
        _bootstrap.active = previous


@contextmanager
def offline_network():
    with ExitStack() as stack:
        stack.enter_context(patch("socket.socket.connect", new=_guarded_connect))
        stack.enter_context(patch("socket.socket.connect_ex", new=_guarded_connect_ex))
        stack.enter_context(patch("socket.socketpair", new=_guarded_socketpair))
        stack.enter_context(patch("socket.create_connection", side_effect=AssertionError("Offline checks forbid network access")))
        yield
