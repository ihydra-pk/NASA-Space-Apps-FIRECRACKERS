"""Check whether this server can reach the AI gateway.

The UPB gateway only accepts connections from the university network or its
VPN, so hosted copies (Render) cannot reach it. ``diagnose`` shows that,
address by address, without using the key.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _seconds(start: float) -> float:
    return round(time.perf_counter() - start, 2)


def diagnose(
    base_urls: list[str],
    *,
    timeout: float = 5.0,
    resolve: Callable[..., list[tuple]] = socket.getaddrinfo,
    connect: Callable[..., Any] = socket.create_connection,
    open_url: Callable[..., Any] = urlopen,
) -> list[dict[str, Any]]:
    """Check DNS, a TCP connection per address, and one HTTPS request per gateway URL.

    The HTTPS request goes to ``<base_url>/models`` without credentials: a
    reachable gateway answers 401, a blocked one times out or refuses.
    """

    reports: list[dict[str, Any]] = []
    for base_url in dict.fromkeys(base_urls):
        parsed = urlparse(base_url)
        host, port = parsed.hostname or "", parsed.port or (443 if parsed.scheme == "https" else 80)
        report: dict[str, Any] = {"base_url": base_url, "host": host, "addresses": [], "https": None}
        start = time.perf_counter()
        try:
            found = resolve(host, port, 0, socket.SOCK_STREAM)
        except OSError as exc:
            report["dns_error"] = f"{type(exc).__name__}: {exc}"
            reports.append(report)
            continue
        for family, _, _, _, sockaddr in dict.fromkeys(found):
            entry = {"ip": sockaddr[0], "family": "IPv6" if family == socket.AF_INET6 else "IPv4"}
            start = time.perf_counter()
            try:
                connect((sockaddr[0], port), timeout=timeout).close()
                entry.update(tcp="connected", seconds=_seconds(start))
            except OSError as exc:
                entry.update(tcp=f"{type(exc).__name__}: {exc}", seconds=_seconds(start))
            report["addresses"].append(entry)
        start = time.perf_counter()
        try:
            with open_url(Request(base_url.rstrip("/") + "/models"), timeout=timeout) as response:
                report["https"] = {"status": response.status, "seconds": _seconds(start)}
        except HTTPError as exc:  # 401 without a key still proves the gateway is reachable
            report["https"] = {"status": exc.code, "seconds": _seconds(start)}
        except OSError as exc:
            report["https"] = {"error": f"{type(exc).__name__}: {exc}", "seconds": _seconds(start)}
        reports.append(report)
    return reports
