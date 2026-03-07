from __future__ import annotations

import ipaddress
import socket
from typing import Any

_ENABLED = False
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_CONNECT = socket.socket.connect
_ORIG_CONNECT_EX = socket.socket.connect_ex


def _host_from_address(address: Any) -> str | None:
    if isinstance(address, tuple) and address:
        host = address[0]
        return str(host) if host is not None else None
    return None


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    lowered = host.strip().lower()
    if lowered in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def is_loopback_host(host: str | None) -> bool:
    return _is_loopback_host(host)


def _assert_loopback(address: Any) -> None:
    host = _host_from_address(address)
    if not _is_loopback_host(host):
        raise OSError(f"Outbound network blocked by policy. Host not loopback: {host!r}")


def enable_loopback_only_network() -> None:
    global _ENABLED
    if _ENABLED:
        return

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        _assert_loopback(address)
        return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        _assert_loopback(address)
        return _ORIG_CONNECT(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        _assert_loopback(address)
        return _ORIG_CONNECT_EX(self, address)

    socket.create_connection = guarded_create_connection
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    _ENABLED = True
