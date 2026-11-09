#!/usr/bin/env python3
"""Talk to a running Lichii debug server from the shell.

Lichii ships an opt-in loopback JSON-lines server (off unless you set
`EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545`). Its adapter lives in
`tools/nx2/debug_mcp.py`; this file is a deliberately small, dependency-free
client for it, plus the two experiments that decide whether an Odyssey-Online
style mod is even possible before you invest in a hook:

  status   is the server up, what is loaded, how fast is it presenting frames
  find     locate a float triple (your position) by search -> move -> narrow,
           which is how you get a stable address with no symbols and no hook
  stream   sample that address at a fixed rate and print it, or relay it as
           nx2navi-mp packets so the room server shows a real player
  nudge    inject stick input, to confirm the debug link can also drive the game

The important asymmetry this exposes: the debug server can *read* memory and
*inject input*, but it cannot write memory and cannot call into game code. So
`find` + `stream` gives you a read-only "ghost" of Link, and a real multiplayer
mod needs the hook described in docs/ocarina-online-via-lichii.md.

Usage:
    python3 tools/nx2navi/probe.py status
    python3 tools/nx2navi/probe.py find --start 0x0 --end 0xFFFFFFFF
    python3 tools/nx2navi/probe.py stream --address 0x81234567 --hz 20
    python3 tools/nx2navi/probe.py nudge --stick -0.5,1.0 --frames 30
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class DebugLink:
    """One request/response at a time, as the server was designed for."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6545, timeout: float = 130.0) -> None:
        self.addr = (host, port)
        self.timeout = timeout
        self.next_id = 1

    def call(self, method: str, **params) -> dict:
        request = {"id": self.next_id, "method": method, "params": params}
        self.next_id += 1
        line = (json.dumps(request) + "\n").encode()
        with socket.create_connection(self.addr, timeout=10) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(line)
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    raise RuntimeError("server closed the connection mid-answer")
                buf += chunk
        reply = json.loads(buf.split(b"\n", 1)[0])
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply.get("result", {})

    # -- helpers over the raw methods --------------------------------------
    def status(self) -> dict:
        return self.call("status")

    def read_floats(self, address: int, count: int = 3) -> list[float]:
        result = self.call("read_memory", address=hex(address), size=4 * count, **{"as": "f32"})
        return [float(v) for v in result["f32"]]

    def search(self, pattern: bytes, start: int, end: int, max_hits: int = 64) -> dict:
        return self.call(
            "search_memory",
            pattern=pattern.hex(),
            start=hex(start),
            end=hex(end),
            max_hits=max_hits,
            regions="all",
        )


class RelayClient:
    """Publishes positions to a relay as if we were the game.

    This is what makes Path A useful: a read-only memory feed becomes a real room
    member, so friends can watch you play while the mod itself is still vapour.
    """

    def __init__(self, host: str, port: int, name: str) -> None:
        from tools.nx2navi.net.protocol import VERSION, Hello, Welcome, decode, encode

        self._encode = encode
        self._PlayerState = None
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        hello = encode(
            Hello(VERSION, 0, 1, 0, 0x0000_0000, name.encode().ljust(8, b"\x00")[:8])
        )
        self.player_id = 0
        for _ in range(20):
            self.sock.sendto(hello, self.addr)
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            msg = decode(data)
            if isinstance(msg, Welcome):
                self.player_id = msg.player_id
                break
        if self.player_id == 0:
            raise SystemExit(f"relay at {host}:{port} did not answer HELLO")
        self._seq = 0

    def publish(self, x: float, y: float, z: float, tick: int) -> None:
        from tools.nx2navi.net.protocol import F_ALIVE, PlayerState

        self._seq = (self._seq + 1) & 0xFFFF
        self.sock.sendto(
            self._encode(
                PlayerState(
                    player_id=self.player_id,
                    anim_id=0,
                    pos_x=x,
                    pos_y=y,
                    pos_z=z,
                    yaw=0,
                    pitch=0,
                    room_id=0,
                    flags=F_ALIVE,
                    seq=self._seq,
                    anim_frame=0,
                    anim_rate=1.0,
                    age=1,
                )
            ),
            self.addr,
        )


def _f32(x: float) -> bytes:
    import struct

    return struct.pack("<f", x)


def cmd_status(args: argparse.Namespace) -> int:
    link = DebugLink(args.host, args.port)
    try:
        info = link.status()
    except OSError as exc:
        print(f"no debug server at {args.host}:{args.port} ({exc})", file=sys.stderr)
        print(
            "start Lichii with EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545 and load the game",
            file=sys.stderr,
        )
        return 2
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key:14} {value}")
    # The doc names these keys; if the build has grown, print what it does have
    # instead of pretending the missing ones are broken.
    for key in ("title", "program_id", "build", "paused", "fps", "renderer"):
        if key not in info:
            print(f"{key:14} (not in this build's status)")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    """Search for a float triple, read it, and print the hits.

    The workflow is three steps by hand: stand still and run `find` with the
    position you read off the screen, walk a few units, run it again, intersect.
    This command does one pass and explains the next one, because guessing a
    coordinate for the user would be a lie.
    """
    link = DebugLink(args.host, args.port)
    pattern = _f32(args.x) + _f32(args.y) + _f32(args.z)
    hits = link.search(pattern, args.start, args.end)
    addresses = hits.get("addresses") or hits.get("hits") or []
    print(f"pattern {pattern.hex()} -> {len(addresses)} hit(s), complete={hits.get('complete')}")
    for addr in addresses[: args.limit]:
        print(f"  {addr}")
    if len(addresses) > args.limit:
        print(f"  ... {len(addresses) - args.limit} more")
    if addresses:
        print("\nnext: move a few units in game, then re-run with the new position and")
        print("keep the addresses present in both lists; that is Link's position vector.")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    link = DebugLink(args.host, args.port)
    relay = None
    if args.relay:
        relay_host, _, relay_port = args.relay.partition(":")
        relay = RelayClient(relay_host, int(relay_port or 8099), args.name)
        print(f"relaying to {relay_host}:{relay_port or 8099} as #{relay.player_id}", file=sys.stderr)
    period = 1.0 / args.hz
    print(f"sampling {hex(args.address)} at {args.hz} Hz, ctrl-c to stop", file=sys.stderr)
    count = 0
    started = time.time()
    try:
        while args.count == 0 or count < args.count:
            tick = time.perf_counter()
            try:
                xyz = link.read_floats(args.address)
            except RuntimeError as exc:
                # One failed sample is not a reason to tear the feed down; the game
                # may be mid-loading-zone with the page unmapped.
                print(f"{count:6d}  read failed: {exc}", file=sys.stderr)
            else:
                print(f"{count:6d}  x={xyz[0]:9.2f} y={xyz[1]:9.2f} z={xyz[2]:9.2f}")
                if relay is not None:
                    relay.publish(xyz[0], xyz[1], xyz[2], count)
            count += 1
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    elapsed = time.time() - started
    print(
        f"\n{count} samples in {elapsed:.1f}s = {count / max(elapsed, 1e-6):.1f} Hz average",
        file=sys.stderr,
    )
    return 0


def cmd_nudge(args: argparse.Namespace) -> int:
    link = DebugLink(args.host, args.port)
    left = [float(v) for v in args.stick.split(",")]
    link.call("input", left_stick=left, frames=args.frames)
    print(f"held stick {left} for {args.frames} frames")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lichii debug-server probe for nx2navi-mp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6545)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="is the debug server up and what is loaded")

    find = sub.add_parser("find", help="search guest memory for a float triple")
    find.add_argument("--x", type=float, required=True)
    find.add_argument("--y", type=float, required=True)
    find.add_argument("--z", type=float, required=True)
    find.add_argument("--start", type=lambda s: int(s, 0), default=0)
    find.add_argument("--end", type=lambda s: int(s, 0), default=0xFFFFFFFF)
    find.add_argument("--limit", type=int, default=20)

    stream = sub.add_parser("stream", help="sample a float triple at a fixed rate")
    stream.add_argument("--address", type=lambda s: int(s, 0), required=True)
    stream.add_argument("--hz", type=float, default=20.0)
    stream.add_argument("--count", type=int, default=0, help="0 = forever")
    stream.add_argument("--relay", default="", help="host:port of a nx2navi-mp relay to publish to")
    stream.add_argument("--name", default="link", help="name shown to other players")

    nudge = sub.add_parser("nudge", help="inject stick input")
    nudge.add_argument("--stick", default="0,0", help="left stick as x,y in -1..1")
    nudge.add_argument("--frames", type=int, default=30)

    args = parser.parse_args()
    handlers = {"status": cmd_status, "find": cmd_find, "stream": cmd_stream, "nudge": cmd_nudge}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
