#!/usr/bin/env python3
"""A fake game client that speaks the real wire protocol.

Use it to prove the pipeline end to end without the console in the loop:

    python3 tools/nx2navi/net/server.py --simulate 0  # terminal 1
    python3 tools/nx2navi/send_test.py --players 2 # terminal 2

Each player joins the room, walks a circle, sets a chest flag and picks up
rupees. The dashboard should show them, and if you connect a second process the
two sets of players should see each other. Anything that breaks here is a
protocol bug, not a game bug, and that separation is the whole point of having
this file before the hook exists.
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.nx2navi.net.protocol import (  # noqa: E402
    F_ALIVE,
    VERSION,
    FlagSet,
    Hello,
    ItemGive,
    Ping,
    PlayerState,
    Welcome,
    decode,
    encode,
)


def s16(binary_angle: int) -> int:
    angle = int(binary_angle) & 0xFFFF
    return angle - 0x10000 if angle >= 0x8000 else angle


def run_player(index: int, target: tuple[str, int], hz: float, seconds: float) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))

    hello = encode(Hello(VERSION, 0, 1, 0, 0xC0FFEE, f"link{index}".encode().ljust(8, b"\x00")))
    me = None
    deadline = time.time() + 10
    while me is None and time.time() < deadline:
        sock.sendto(hello, target)
        sock.settimeout(0.5)
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        msg = decode(data)
        if isinstance(msg, Welcome):
            me = msg.player_id
    if me is None:
        print(f"player {index}: no WELCOME from {target}", file=sys.stderr)
        return
    print(f"player {index}: joined as #{me}")

    period = 1.0 / hz
    boot = time.monotonic()
    ends = time.time() + seconds
    seq = 0
    t = index * 1.7
    while time.time() < ends:
        tick = time.perf_counter()
        t += period * 0.9
        seq = (seq + 1) & 0xFFFF
        radius = 240.0 + index * 110
        sock.sendto(
            encode(
                PlayerState(
                    player_id=me,
                    anim_id=0x2B,
                    pos_x=math.cos(t) * radius,
                    pos_y=0.0,
                    pos_z=math.sin(t) * radius,
                    yaw=s16((t * 65536 / math.tau) % 65536),
                    pitch=0,
                    room_id=0,
                    flags=F_ALIVE,
                    seq=seq,
                    anim_frame=(seq * 4) & 0xFFFF,
                    anim_rate=1.0,
                    age=1,
                )
            ),
            target,
        )
        if seq % int(hz * 3) == 0:
            sock.sendto(
                encode(FlagSet(me, 0, 0x51, seq % 512, 1 << (seq % 32), 1)), target
            )
        if seq % int(hz * 9) == 0:
            sock.sendto(encode(ItemGive(me, 0x33, 0xFF, 0, 5)), target)
            sock.sock.sendto(encode(Ping(seq, int((time.monotonic() - boot) * 1000) & 0xFFFFFFFF)), target)
        time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    print(f"player {index}: sent {seq} packets")


def main() -> int:
    parser = argparse.ArgumentParser(description="fake nx2navi-mp game client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--players", type=int, default=1)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    for i in range(args.players):
        run_player(i, (args.host, args.port), args.hz, args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
