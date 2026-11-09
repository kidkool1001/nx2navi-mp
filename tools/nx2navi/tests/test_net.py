"""Tests for the transport: framing, the relay's room behaviour, and replay.

Run with:
    python3 -m unittest discover -s tools/nx2navi/tests -v

They are deliberately behavioural: the relay is started in-process on an ephemeral
port and driven with real UDP sockets, because the bugs that matter in a relay are
"two clients never learned about each other" and "a late joiner got half a world",
not "the struct packed".
"""

from __future__ import annotations

import asyncio
import socket
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.nx2navi.net.protocol import (  # noqa: E402
    F_ALIVE,
    VERSION,
    FlagSet,
    Hello,
    PacketType,
    PeerJoin,
    PlayerState,
    ProtocolError,
    Welcome,
    decode,
    encode,
)
from tools.nx2navi.net.server import Relay  # noqa: E402


def _hello(name: str) -> bytes:
    return encode(Hello(VERSION, 0, 1, 0, 0xC0FFEE, name.encode().ljust(8, b"\x00")))


def _state(player_id: int, seq: int, x: float) -> bytes:
    return encode(
        PlayerState(player_id, 0x2B, x, 0.0, 0.0, 0, 0, 0, F_ALIVE, seq, 0, 1.0, 1)
    )


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_every_type(self) -> None:
        cases = [
            PlayerState(3, 0x2B, 1.5, -2.5, 3.5, -32768, 32767, 4, F_ALIVE, 65535, 12, -1.0, 1),
            Hello(VERSION, 0, 0, 0, 0xDEADBEEF, b"navi\x00\x00\x00\x00"),
            Welcome(7, 3, 20),
            PeerJoin(2, b"saria\x00\x00\x00"),
            FlagSet(1, 0, 0x51, 511, 0xFFFFFFFF, 1),
        ]
        for case in cases:
            self.assertEqual(decode(encode(case)), case)

    def test_bad_magic(self) -> None:
        packet = bytearray(encode(Welcome(1, 1, 20)))
        packet[0] ^= 0xFF
        with self.assertRaises(ProtocolError):
            decode(bytes(packet))

    def test_truncated_and_oversized(self) -> None:
        packet = encode(PeerJoin(1, b"x"))
        with self.assertRaises(ProtocolError):
            decode(packet[:4])
        with self.assertRaises(ProtocolError):
            decode(packet[:-1])

    def test_unknown_type_is_ignored_not_fatal(self) -> None:
        # A future packet type must not crash an older client.
        from tools.nx2navi.net.protocol import HEADER, MAGIC

        self.assertIsNone(decode(HEADER.pack(MAGIC, VERSION, 0x0999, 0, 0)))

    def test_field_that_does_not_fit_names_itself(self) -> None:
        with self.assertRaises(ProtocolError) as caught:
            encode(PlayerState(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 400))
        self.assertIn("age", str(caught.exception))


class RelayTests(unittest.TestCase):
    """Each test runs its own relay on an ephemeral UDP port."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.relay = Relay("127.0.0.1", 0, quiet=True)
        self.transport, _ = self.loop.run_until_complete(
            self.loop.create_datagram_endpoint(
                lambda: _Collect(self.relay), local_addr=("127.0.0.1", 0)
            )
        )
        # The relay sends through its own attribute; the test owns the socket.
        self.relay.transport = self.transport
        self.port = self.transport.get_extra_info("sockname")[1]

    def tearDown(self) -> None:
        self.transport.close()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    # -- helpers ----------------------------------------------------------
    def client(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(0.05)
        self.addCleanup(sock.close)
        return sock

    def pump(self, seconds: float = 0.15) -> None:
        """Let the relay's datagram callbacks run."""
        end = time.time() + seconds
        while time.time() < end:
            self.loop.run_until_complete(asyncio.sleep(0.005))

    def drain(self, sock: socket.socket) -> list:
        out = []
        while True:
            try:
                data, _ = sock.recvfrom(2048)
            except (socket.timeout, BlockingIOError):
                return out
            out.append(decode(data))

    def join(self, sock: socket.socket, name: str) -> tuple[int, list]:
        """Join and return (assigned id, everything the relay sent in reply).

        The reply is returned rather than discarded, because the interesting part
        of a join is the backlog that comes with it.
        """
        sock.sendto(_hello(name), ("127.0.0.1", self.port))
        self.pump()
        messages = self.drain(sock)
        for msg in messages:
            if isinstance(msg, Welcome):
                return msg.player_id, messages
        self.fail(f"no WELCOME for {name}")

    # -- tests ------------------------------------------------------------
    def test_join_assigns_ids_and_greets(self) -> None:
        a, b = self.client(), self.client()
        id_a, _ = self.join(a, "navi")
        id_b, b_msgs = self.join(b, "saria")
        self.assertEqual((id_a, id_b), (1, 2))
        self.assertTrue(any(isinstance(m, Welcome) and m.room_size == 2 for m in b_msgs))
        self.assertTrue(any(isinstance(m, PeerJoin) and m.peer_id == id_a for m in b_msgs))

    def test_state_is_forwarded_to_peers_only(self) -> None:
        a, b = self.client(), self.client()
        id_a, _ = self.join(a, "navi")
        self.join(b, "saria")
        self.drain(a)
        self.drain(b)
        a.sendto(_state(id_a, 1, 42.0), ("127.0.0.1", self.port))
        self.pump()
        got = [m for m in self.drain(b) if isinstance(m, PlayerState)]
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0].pos_x, 42.0)
        self.assertEqual(self.drain(a), [], "a peer must not receive its own state")

    def test_stale_sequence_is_dropped(self) -> None:
        a, b = self.client(), self.client()
        id_a, _ = self.join(a, "navi")
        self.join(b, "saria")
        self.drain(b)
        for seq, x in ((10, 1.0), (9, 2.0), (11, 3.0)):
            a.sendto(_state(id_a, seq, x), ("127.0.0.1", self.port))
            self.pump(0.05)
        forwarded = [m.pos_x for m in self.drain(b) if isinstance(m, PlayerState)]
        self.assertEqual(forwarded, [1.0, 3.0])

    def test_late_joiner_receives_journalled_world_events(self) -> None:
        a = self.client()
        id_a, _ = self.join(a, "navi")
        a.sendto(encode(FlagSet(id_a, 0, 0x51, 7, 1, 1)), ("127.0.0.1", self.port))
        self.pump()
        self.assertEqual(len(self.relay.room.journal), 1)

        b = self.client()
        _, b_msgs = self.join(b, "saria")
        replay = [m for m in b_msgs if isinstance(m, FlagSet)]
        self.assertEqual(len(replay), 1, "the journal should be replayed on join")
        self.assertEqual(replay[0].index, 7)
        self.assertEqual(self.relay.room.totals["replayed"], 1)

    def test_state_before_hello_is_ignored(self) -> None:
        sock = self.client()
        sock.sendto(_state(1, 1, 1.0), ("127.0.0.1", self.port))
        self.pump()
        self.assertEqual(len(self.relay.room.peers), 0)

    def test_leave_is_announced_to_the_room(self) -> None:
        a, b = self.client(), self.client()
        _, _ = self.join(a, "navi")
        self.join(b, "saria")
        self.drain(b)
        self.join(a, "navi")  # re-hello from the same address = reconnect
        self.pump()
        leaves = [m for m in self.drain(b) if m is not None and m.TYPE.name == "PEER_LEAVE"]
        self.assertEqual([m.peer_id for m in leaves], [1])

    def test_garbage_is_counted_not_fatal(self) -> None:
        sock = self.client()
        sock.sendto(b"\x00" * 3, ("127.0.0.1", self.port))
        sock.sendto(b"\xde\xad\xbe\xef" * 8, ("127.0.0.1", self.port))
        self.pump()
        self.assertEqual(self.relay.room.totals["bad"], 2)
        self.join(sock, "navi")  # still serving

    def test_event_types_are_the_ones_that_get_journalled(self) -> None:
        a = self.client()
        id_a, _ = self.join(a, "navi")
        a.sendto(_state(id_a, 1, 1.0), ("127.0.0.1", self.port))
        a.sendto(encode(FlagSet(id_a, 0, 1, 0, 1, 1)), ("127.0.0.1", self.port))
        self.pump()
        self.assertEqual(len(self.relay.room.journal), 1, "state packets must not be journalled")


class _Collect(asyncio.DatagramProtocol):
    def __init__(self, relay: Relay) -> None:
        self.relay = relay

    def datagram_received(self, data: bytes, addr) -> None:
        self.relay.handle(data, addr)


if __name__ == "__main__":
    unittest.main()
