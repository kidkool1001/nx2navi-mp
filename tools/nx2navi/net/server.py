#!/usr/bin/env python3
"""nx2navi-mp relay server.

This is the piece of a Super Mario Odyssey Online style mod that does not depend on
having the game cracked open: the room server. It is deliberately boring and
deterministic, because everything interesting happens on the console/emulator side
and you do not want to debug two hard things at once.

What it does
------------
* Accepts game clients over UDP (JSON-free, fixed binary layout from protocol.py).
* Runs a "room": every client in it learns about every other client.
* Fans out state packets unreliably (a dropped position update is worthless anyway)
  and world events exactly once (a duplicate chest flag is a corrupt save).
* Keeps a bounded journal of world events and replays it to a late joiner, so you
  can join a friend's session mid-run instead of starting from the title screen.
* Serves a live dashboard over HTTP + WebSocket so you can *see* the session while
  you build the console side. That dashboard is the only way to test this without
  owning a dumped copy of the game.

Run it:
    python3 tools/nx2navi/net/server.py --port 8099 --simulate 3

and open the preview URL. `--simulate 3` starts three fake Links walking around
Hyrule so the dashboard has something to show before the game side exists.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.nx2navi.net.protocol import (  # noqa: E402
    F_ALIVE,
    MAGIC,
    MAX_BODY,
    VERSION,
    FlagSet,
    Hello,
    ItemGive,
    PacketType,
    PeerEvent,
    PeerJoin,
    Ping,
    PlayerState,
    Welcome,
    decode,
    describe,
    encode,
)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
JOURNAL_LIMIT = 4096          # world events kept for late joiners
EVENT_FEED = 250              # messages kept for the dashboard feed
MAX_CLIENTS = 10
MAX_PACKETS_PER_SEC = 400     # per client, protects the relay from a runaway hook
STALE_AFTER_S = 5.0           # a peer that sends nothing for this long is greyed out


@dataclass
class Peer:
    peer_id: int
    addr: tuple[str, int]
    name: str = "link"
    joined: float = field(default_factory=time.time)
    last_state: PlayerState | None = None
    last_seen: float = field(default_factory=time.time)
    packets_in: int = 0
    packets_out: int = 0
    seq_last: int = -1
    dropped: int = 0           # inferred from sequence gaps: the real network loss signal
    tokens: float = float(MAX_PACKETS_PER_SEC)
    tokens_at: float = field(default_factory=time.time)

    def alive(self) -> bool:
        return (time.time() - self.last_seen) < STALE_AFTER_S

    def refill(self) -> bool:
        """Token bucket. Returns False when this client should be dropped for abuse."""
        now = time.time()
        self.tokens = min(
            MAX_PACKETS_PER_SEC, self.tokens + (now - self.tokens_at) * MAX_PACKETS_PER_SEC
        )
        self.tokens_at = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


class Room:
    def __init__(self) -> None:
        self.peers: dict[int, Peer] = {}
        self.journal: list[bytes] = []      # encoded reliable messages
        self.feed: list[dict] = []          # decoded, for the dashboard
        self.next_id = 1
        self.totals = {"in": 0, "out": 0, "events": 0, "replayed": 0, "bad": 0}
        self.subscribers: set[asyncio.StreamWriter] = set()

    # -- peers ---------------------------------------------------------------
    def add_peer(self, addr: tuple[str, int]) -> Peer | None:
        if len(self.peers) >= MAX_CLIENTS:
            return None
        peer_id = self.next_id
        self.next_id += 1
        peer = Peer(peer_id=peer_id, addr=addr)
        self.peers[peer_id] = peer
        return peer

    def remove_peer(self, peer: Peer) -> None:
        self.peers.pop(peer.peer_id, None)

    def others(self, peer: Peer) -> list[Peer]:
        return [p for p in self.peers.values() if p.peer_id != peer.peer_id]

    # -- message plumbing ---------------------------------------------------
    def note(self, msg, peer: Peer | None) -> None:
        entry = describe(msg)
        entry["t"] = round(time.time(), 3)
        if peer is not None:
            entry["from"] = peer.peer_id
        self.feed.append(entry)
        if len(self.feed) > EVENT_FEED:
            del self.feed[: len(self.feed) - EVENT_FEED]

    def journal_add(self, datagram: bytes, msg) -> None:
        if msg.TYPE not in (PacketType.FLAG_SET, PacketType.ITEM_GIVE, PacketType.GAME_MODE):
            return
        self.journal.append(datagram)
        if len(self.journal) > JOURNAL_LIMIT:
            del self.journal[: len(self.journal) - JOURNAL_LIMIT]
        self.totals["events"] += 1


class Relay:
    def __init__(self, host: str, port: int, quiet: bool = False) -> None:
        self.host = host
        self.port = port
        self.quiet = quiet
        self.room = Room()
        self.transport: asyncio.DatagramTransport | None = None
        self.started = time.time()
        self.background: set[asyncio.Task] = set()

    # -- UDP -----------------------------------------------------------------
    async def start_udp(self) -> None:
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _UdpProtocol(self), local_addr=(self.host, self.port)
        )
        self.log(f"udp   listening on {self.host}:{self.port}")

    def send(self, datagram: bytes, addr: tuple[str, int]) -> None:
        if self.transport is None:
            return
        try:
            self.transport.sendto(datagram, addr)
            self.room.totals["out"] += 1
        except OSError:
            pass  # a peer vanished between the check and the send; UDP has no delivery

    def drop(self, peer: Peer, reason: str) -> None:
        """Remove a peer and tell the room. Silence would leave stale puppets behind."""
        self.room.remove_peer(peer)
        leave = encode(PeerEvent(peer_id=peer.peer_id, owner_name=peer.name.encode()))
        for other in self.room.peers.values():
            self.send(leave, other.addr)
        self.log(f"leave #{peer.peer_id} {peer.name} ({reason})")
        self.push_dashboard({"kind": "leave", "id": peer.peer_id, "reason": reason})

    def log(self, text: str) -> None:
        if not self.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    # -- packet handling -----------------------------------------------------
    def handle(self, data: bytes, addr: tuple[str, int]) -> None:
        room = self.room
        room.totals["in"] += 1
        try:
            msg = decode(data)
        except Exception as exc:  # noqa: BLE001 - any malformed datagram is just counted
            room.totals["bad"] += 1
            self.log(f"bad packet from {addr[0]}:{addr[1]}: {exc}")
            return
        if msg is None:
            return  # a packet type this build does not implement yet

        peer = next((p for p in room.peers.values() if p.addr == addr), None)

        # Session control first: it is the only thing allowed before a HELLO.
        if isinstance(msg, Hello):
            if peer is not None:
                # A reconnect from the same address: keep the room consistent.
                self.drop(peer, "reconnect")
            peer = room.add_peer(addr)
            if peer is None:
                self.log(f"refused {addr[0]}:{addr[1]}: room is full")
                return
            peer.name = msg.owner_name.rstrip(b"\x00").decode("utf-8", "replace") or "link"
            self.log(f"join  #{peer.peer_id} {peer.name} from {addr[0]}:{addr[1]}")
            self.send(encode(Welcome(peer.peer_id, len(room.peers), 20)), addr)
            # Tell the newcomer who is already here, and everyone else about them.
            # From the newcomer's point of view every existing peer did just join,
            # so both directions use PEER_JOIN; PEER_LEAVE is only for departures.
            for other in room.others(peer):
                self.send(
                    encode(PeerJoin(peer_id=other.peer_id, owner_name=other.name.encode())),
                    addr,
                )
                if other.last_state is not None:
                    self.send(encode(other.last_state), addr)
            join = PeerJoin(peer_id=peer.peer_id, owner_name=peer.name.encode())
            for other in room.others(peer):
                self.send(encode(join), other.addr)
            # Replay world state so a late joiner does not restart the world.
            for datagram in room.journal:
                self.send(datagram, addr)
                room.totals["replayed"] += 1
            room.note(msg, peer)
            self.push_dashboard({"kind": "join", "id": peer.peer_id, "name": peer.name})
            return

        if peer is None:
            return  # state before HELLO: ignore rather than guess who it is

        if not peer.refill():
            self.drop(peer, f"over {MAX_PACKETS_PER_SEC} packets/s")
            return

        peer.packets_in += 1
        peer.last_seen = time.time()

        if isinstance(msg, Ping):
            self.send(encode(Ping(nonce=msg.nonce, client_ms=msg.client_ms)), addr)
            return

        if isinstance(msg, PlayerState):
            # Stale reordering is normal on UDP; drop it instead of snapping a puppet.
            if peer.seq_last >= 0:
                gap = (msg.seq - peer.seq_last) & 0xFFFF
                if gap == 0 or gap > 0x8000:
                    return
                if gap > 1:
                    peer.dropped += gap - 1
            peer.seq_last = msg.seq
            peer.last_state = msg
            datagram = data
            for other in room.others(peer):
                if other.alive():
                    self.send(datagram, other.addr)
                    other.packets_out += 1
            if room.totals["in"] % 25 == 0:
                room.note(msg, peer)
            return

        # World events: exactly once, and journalled.
        room.journal_add(data, msg)
        room.note(msg, peer)
        for other in room.others(peer):
            self.send(data, other.addr)
            other.packets_out += 1
        self.push_dashboard({"kind": "event", "type": msg.TYPE.name})

    # -- dashboard -----------------------------------------------------------
    def snapshot(self) -> dict:
        room = self.room
        now = time.time()
        players = []
        for peer in sorted(room.peers.values(), key=lambda p: p.peer_id):
            state = peer.last_state
            players.append(
                {
                    "id": peer.peer_id,
                    "name": peer.name,
                    "addr": f"{peer.addr[0]}:{peer.addr[1]}",
                    "alive": peer.alive(),
                    "state": None
                    if state is None
                    else {
                        "x": round(state.pos_x, 1),
                        "y": round(state.pos_y, 1),
                        "z": round(state.pos_z, 1),
                        "yaw": state.yaw,
                        "age": state.age,
                        "anim": state.anim_id,
                        "room": state.room_id,
                        "flags": state.flags,
                        "seq": state.seq,
                    },
                    "packets_in": peer.packets_in,
                    "dropped": peer.dropped,
                    "uptime_s": round(now - peer.joined, 1),
                }
            )
        return {
            "uptime_s": round(now - self.started, 1),
            "room_size": len(room.peers),
            "journal": len(room.journal),
            "totals": room.totals,
            "players": players,
            "feed": room.feed[-40:],
            "now": round(now, 3),
        }

    def push_dashboard(self, event: dict) -> None:
        if not self.room.subscribers:
            return
        payload = json.dumps(event).encode()
        frame = _ws_frame(payload)
        for writer in list(self.room.subscribers):
            try:
                writer.write(frame)
            except Exception:  # noqa: BLE001 - a dead dashboard is not a server error
                self.room.subscribers.discard(writer)


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, relay: Relay) -> None:
        self.relay = relay

    def datagram_received(self, data: bytes, addr) -> None:
        self.relay.handle(data, addr)

    def error_received(self, exc: Exception) -> None:
        self.relay.log(f"udp error: {exc}")


# ---------------------------------------------------------------------------
# Minimal WebSocket (RFC 6455) so the dashboard needs no dependencies
# ---------------------------------------------------------------------------


def s16(binary_angle: int) -> int:
    """Wrap a 0..0xFFFF binary angle into the signed range the wire format uses."""
    angle = int(binary_angle) & 0xFFFF
    return angle - 0x10000 if angle >= 0x8000 else angle


def _ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([n])
    elif n < 65536:
        header += bytes([126]) + struct.pack(">H", n)
    else:
        header += bytes([127]) + struct.pack(">Q", n)
    return header + payload


async def _read_ws_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    b1, b2 = await reader.readexactly(2)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack(">H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if masked else b"\x00\x00\x00\x00"
    data = bytearray(await reader.readexactly(length))
    if masked:
        for i in range(length):
            data[i] ^= mask[i % 4]
    return opcode, bytes(data)


STATIC = Path(__file__).resolve().parent


async def _http_client(relay: Relay, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        writer.close()
        return
    lines = request.decode("latin-1").split("\r\n")
    method, path, _ = (lines[0].split(" ") + ["", "", ""])[:3]
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    if headers.get("upgrade", "").lower() == "websocket":
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        writer.write(
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
        )
        await writer.drain()
        relay.room.subscribers.add(writer)
        try:
            writer.write(_ws_frame(json.dumps(relay.snapshot()).encode()))
            while True:
                opcode, _ = await _read_ws_frame(reader)
                if opcode == 0x8:  # close
                    break
        except Exception:  # noqa: BLE001 - dashboard went away
            pass
        finally:
            relay.room.subscribers.discard(writer)
            writer.close()
        return

    if path in ("/", "/index.html"):
        body = (STATIC / "dashboard.html").read_bytes()
        ctype = "text/html; charset=utf-8"
    elif path.startswith("/snapshot.json"):
        body = json.dumps(relay.snapshot()).encode()
        ctype = "application/json"
    elif path == "/healthz":
        body = b"ok\n"
        ctype = "text/plain"
    else:
        body = b"not found\n"
        ctype = "text/plain"
        writer.write(
            b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n"
            b"Content-Length: 10\r\nConnection: close\r\n\r\n" + body
        )
        await writer.drain()
        writer.close()
        return

    # No caching, ever: the dashboard is regenerated constantly during development.
    writer.write(
        f"HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\n"
        f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\n"
        "Connection: close\r\n\r\n".encode()
        + body
    )
    await writer.drain()
    writer.close()


# ---------------------------------------------------------------------------
# Fake Links, so the dashboard is testable before the game side exists
# ---------------------------------------------------------------------------


async def _simulate(host: str, port: int, count: int, quiet: bool) -> None:
    """Each fake client is a real UDP peer: same code path as the console."""
    names = ["navi", "saria", "darunia", "ruto", "zelda", "malon", "sheik", "impa"]
    tasks = []
    for i in range(count):
        tasks.append(asyncio.create_task(_simulate_one(host, port, i, names[i % len(names)])))
    if not quiet:
        print(f"[sim] {count} fake Links walking around", flush=True)
    await asyncio.gather(*tasks)


async def _simulate_one(host: str, port: int, index: int, name: str) -> None:
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind(("0.0.0.0", 0))
    target = (host if host != "0.0.0.0" else "127.0.0.1", port)

    def send(datagram: bytes) -> None:
        try:
            sock.sendto(datagram, target)
        except OSError:
            pass

    # Retry HELLO until a WELCOME arrives: the server may not be up yet.
    me = None
    received: list[bytes] = []

    def drain() -> None:
        while True:
            try:
                data, _ = sock.recvfrom(2048)
            except BlockingIOError:
                return
            received.append(data)

    while me is None:
        send(encode(Hello(VERSION, 0, 1, 0, 0xDEADBEEF, name.encode().ljust(8, b"\x00")[:8])))
        await asyncio.sleep(0.5)
        drain()
        for data in received:
            msg = decode(data)
            if isinstance(msg, Welcome):
                me = msg.player_id
        received.clear()

    # A slow walk in a circle around Hyrule Field, plus the occasional chest.
    boot = time.monotonic()
    t = random.random() * math.tau
    radius = 300.0 + index * 90.0
    seq = 0
    while True:
        t += 0.06
        seq = (seq + 1) & 0xFFFF
        x = math.cos(t) * radius
        z = math.sin(t) * radius
        send(
            encode(
                PlayerState(
                    player_id=me,
                    anim_id=0x2B if seq % 40 < 20 else 0x2C,
                    pos_x=x,
                    pos_y=0.0,
                    pos_z=z,
                    yaw=s16((t * 65536 / math.tau) % 65536),
                    pitch=0,
                    room_id=0,
                    flags=F_ALIVE,
                    seq=seq,
                    anim_frame=(seq * 3) & 0xFFFF,
                    anim_rate=1.0,
                    age=1,
                )
            )
        )
        if seq % 250 == 0:
            send(
                encode(
                    FlagSet(
                        player_id=me,
                        group=0,
                        scene_id=0x51,
                        index=seq % 512,
                        bitfield=1 << (seq % 32),
                        value=1,
                    )
                )
            )
        if seq % 700 == 0:
            send(encode(ItemGive(player_id=me, item_id=0x21, slot=3, flags=0, delta=10)))
        if seq % 100 == 0:
            send(encode(Ping(nonce=seq, client_ms=int((time.monotonic() - boot) * 1000) & 0xFFFFFFFF)))
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    relay = Relay(args.host, args.port, args.quiet)
    await relay.start_udp()
    server = await asyncio.start_server(
        lambda r, w: _http_client(relay, r, w), args.host, args.http_port
    )
    relay.log(f"http  dashboard on http://{args.host}:{args.http_port}/")
    if args.simulate:
        # Keep a reference: a bare create_task() can be garbage collected mid-run,
        # which looks exactly like "the simulator stopped after 100 packets".
        relay.background.add(
            asyncio.create_task(_simulate(args.host, args.port, args.simulate, args.quiet))
        )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="nx2navi-mp relay server")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 for the preview)")
    parser.add_argument("--port", type=int, default=8099, help="UDP port for game clients")
    parser.add_argument("--http-port", type=int, default=8099, help="HTTP/WebSocket port")
    parser.add_argument("--simulate", type=int, default=0, help="spawn N fake Links")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
