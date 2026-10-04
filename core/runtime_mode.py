"""Cloud admission and quiet recovery; no credentials or model requests in probes."""
import asyncio
import socket
import ssl
import time


async def provider_reachable(provider, timeout=2.0):
    host = 'agent.deepgram.com' if provider == 'deepgram' else 'generativelanguage.googleapis.com'
    writer = None
    try:
        async with asyncio.timeout(timeout):
            _, writer = await asyncio.open_connection(host, 443, ssl=ssl.create_default_context())
        return True
    except (OSError, TimeoutError):
        return False
    finally:
        if writer is not None:
            writer.close()


def error_kind(error):
    """Inspect nested TaskGroup failures without logging exception payloads."""
    children = getattr(error, 'exceptions', ())
    kinds = [error_kind(child) for child in children]
    if 'auth' in kinds:
        return 'auth'
    if 'network' in kinds:
        return 'network'
    low = str(error).lower()
    if any(word in low for word in ('unauthorized', 'invalid api key', 'api key not valid',
                                    'invalid token', 'permission_denied', '401', '403')):
        return 'auth'
    if isinstance(error, (socket.gaierror, ConnectionError, TimeoutError, ssl.SSLError)):
        return 'network'
    if isinstance(error, OSError) and getattr(error, 'errno', None) in (
            101, 104, 110, 111, 113, 10051, 10054, 10060, 10061, 11001):
        return 'network'
    if type(error).__name__.startswith('ConnectionClosed') or (
            type(error).__module__.startswith('websockets') and
            type(error).__name__ in ('InvalidHandshake', 'InvalidStatus', 'InvalidStatusCode', 'ProtocolError')):
        return 'network'
    if any(word in low for word in ('getaddrinfo', 'network unreachable', 'network is unreachable',
                                    'timed out', 'cannot connect', 'connection refused',
                                    'connection reset', 'connection aborted', 'socket error',
                                    'dns failure', 'websocket transport', 'no close frame',
                                    'http 502', 'http 503', 'http 504')):
        return 'network'
    return 'other'


class RuntimeMode:
    def __init__(self, settings, probe=provider_reachable, clock=time.monotonic):
        self.settings, self.probe, self.clock = settings, probe, clock
        self.offline = False
        self.reason = ''
        self.next_probe = 0.0
        self.blocked_credential = None

    @property
    def mode(self):
        return self.settings()['mode']

    def failed(self, kind, credential):
        self.offline = True
        self.reason = 'Authentication failed; update the live provider key.' if kind == 'auth' else 'Internet/live provider unavailable.'
        self.next_probe = self.clock() + 45.0
        if kind == 'auth':
            self.blocked_credential = credential

    async def ready(self, provider, credential, idle=True):
        mode = self.mode
        if mode == 'offline':
            self.offline, self.reason = True, 'Offline mode selected.'
            return False
        if mode == 'live':
            self.offline = False
            return True
        if not credential or credential == self.blocked_credential:
            self.offline, self.reason = True, 'Live provider key missing or rejected; local mode available.'
            return False
        if self.offline and self.clock() < self.next_probe:
            return False
        if self.offline and not idle:
            return False
        reachable = await self.probe(provider)
        self.offline = not reachable
        if not reachable:
            self.reason = 'Internet/live provider unavailable.'
            self.next_probe = self.clock() + 45.0
        return reachable
