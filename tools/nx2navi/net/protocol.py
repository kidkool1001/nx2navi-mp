"""Wire protocol for nx2navi-mp.

One file, shared by the server, the Lichii-side client and the probes, so the
packet layout is defined exactly once. Only stdlib is used: this has to run under
Python 3.11 on any machine, and it is small enough to be re-implemented in C++ or
C# in an afternoon.

Design rules (deliberate, see docs/ocarina-online-via-lichii.md):

* Little endian everywhere, fixed-size structs, no pointers, no strings with
  lengths buried in the middle of a body.
* Every packet is 4-byte aligned so an ARM64 or ARM32 implementation can memcpy
  the body straight out of a receive buffer.
* Twelve bytes of header, and the receiver validates `size` before touching the
  body, so a truncated or hostile datagram can never overrun a buffer.
* Everything is additive: an old client must be able to ignore a packet type it
  does not know. `decode` returns `None` for unknown types instead of raising.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Any, ClassVar

MAGIC = 0x4E58324D  # "NX2M"
VERSION = 1

HEADER = struct.Struct("<IHHHH")  # magic, version, type, body_size, flags
HEADER_SIZE = HEADER.size
MAX_BODY = 4096  # one datagram, no fragmentation: we are not re-implementing TCP


class PacketType(IntEnum):
    """Message kinds.

    The 1xx block is the state stream (one-way, unreliable is fine), the 2xx block
    is world events that must arrive exactly once (send reliably), and the 3xx
    block is session control.
    """

    UNKNOWN = 0

    # --- session control -------------------------------------------------
    HELLO = 0x0301        # client -> server: who am I, which stage/seed am I on
    WELCOME = 0x0302      # server -> client: your id, the room, everyone else
    PEER_JOIN = 0x0303    # server -> client: someone else arrived
    PEER_LEAVE = 0x0304   # server -> client: someone else left
    PING = 0x0305
    PONG = 0x0306

    # --- state stream, ~20 Hz -------------------------------------------
    PLAYER_STATE = 0x0101  # one actor's transform + animation + status flags
    SAVE_STATE = 0x0108    # periodic full snapshot for late joiners

    # --- world events, exactly once -------------------------------------
    FLAG_SET = 0x0201      # a scene/quest flag turned on (chest, door, song...)
    ITEM_GIVE = 0x0202     # an inventory item (or ammo count delta)
    GAME_MODE = 0x0203     # age change, cutscene start/end, warp, title screen
    ROOM_STATE = 0x0204    # dungeon room visited mask, per scene


# ---------------------------------------------------------------------------
# Body definitions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Message:
    """Base class. Subclasses declare `BODY = struct.Struct(...)` and fields in
    the same order. `pack`/`unpack` derive the layout automatically."""

    TYPE: ClassVar[PacketType] = PacketType.UNKNOWN
    BODY: ClassVar[struct.Struct] = struct.Struct("<")

    def pack_body(self) -> bytes:
        values = [getattr(self, f.name) for f in fields(self)]
        try:
            return self.BODY.pack(*values)
        except struct.error as exc:
            # Name the field that does not fit: a bare "short format requires ..."
            # costs an hour of bisecting fields by hand. The struct's own format
            # characters line up one-to-one with the dataclass fields (no `x` pad).
            for f, char, value in zip(fields(self), _format_chars(self.BODY.format), values):
                try:
                    struct.pack("<" + char, value)
                except struct.error:
                    raise ProtocolError(
                        f"{type(self).__name__}.{f.name} = {value!r} does not fit "
                        f"the wire format ({char}): {exc}"
                    ) from exc
            raise ProtocolError(f"{type(self).__name__} does not pack: {exc}") from exc

    @classmethod
    def unpack_body(cls, data: bytes) -> "Message":
        values = cls.BODY.unpack(data[: cls.BODY.size])
        return cls(**{f.name: v for f, v in zip(fields(cls), values)})


# flags carried in PlayerState.flags
F_ALIVE = 1 << 0
F_ON_EPONA = 1 << 1
F_Z_TARGETING = 1 << 2
F_FIRST_PERSON = 1 << 3
F_IN_CUTSCENE = 1 << 4
F_SWIMMING = 1 << 5
F_CLIMBING = 1 << 6
F_SHIELD = 1 << 7
F_ITEM_ACTION = 1 << 8  # spinning, playing ocarina, hookshot in flight...


@dataclass(slots=True)
class PlayerState(Message):
    """The packet that matters most, and the one SMOO sends every third frame.

    Positions are the game's own float coordinates (the same units the actor
    table uses after Link's world->room transform), not quantised, because 4
    bytes of float is cheaper here than the code to pack a 21-bit fixpoint and
    the precision bugs that come with it.
    """

    TYPE: ClassVar[PacketType] = PacketType.PLAYER_STATE
    BODY: ClassVar[struct.Struct] = struct.Struct("<BHfffhhhIHHfB")

    player_id: int          # u16  server-assigned
    anim_id: int            # u8   animation id from the hook, index into the anim map
    pos_x: float            # f32
    pos_y: float            # f32
    pos_z: float            # f32
    # yaw and pitch are signed binary angles (s16), the game's own unit: 0x4000 is
    # a quarter turn. -32768 and 32767 are the same heading, so a puppet must
    # interpolate the short way round or it will spin.
    yaw: int                # s16
    pitch: int              # s16
    room_id: int            # s16  which room/scene the actor is in, so puppets can hide
    flags: int              # u32  F_* above
    seq: int                # u16  per-sender sequence, drops stale packets
    anim_frame: int         # u16  animation playback position, for blending
    anim_rate: float        # f32  negative plays backwards
    age: int                # u8   0 = child, 1 = adult

    @property
    def is_alive(self) -> bool:
        return bool(self.flags & F_ALIVE)


@dataclass(slots=True)
class Hello(Message):
    TYPE: ClassVar[PacketType] = PacketType.HELLO
    BODY: ClassVar[struct.Struct] = struct.Struct("<BBBB I 8s")

    protocol_version: int
    session_kind: int       # 0 = room (all Links visible), 1 = state sync only
    age: int
    reserved: int
    seed_hash: int          # u32, so two different randomizer seeds cannot mix
    owner_name: bytes       # 8 bytes, UTF-8, not null-terminated; not trusted


@dataclass(slots=True)
class Welcome(Message):
    TYPE: ClassVar[PacketType] = PacketType.WELCOME
    BODY: ClassVar[struct.Struct] = struct.Struct("<HHH")

    player_id: int
    room_size: int
    tick_hz: int


@dataclass(slots=True)
class PeerEvent(Message):
    """PEER_JOIN and PEER_LEAVE share one body; two subtypes keep the type field
    honest instead of mutating a class attribute at runtime."""

    TYPE: ClassVar[PacketType] = PacketType.PEER_LEAVE
    BODY: ClassVar[struct.Struct] = struct.Struct("<H 8s")

    peer_id: int
    owner_name: bytes


@dataclass(slots=True)
class PeerJoin(PeerEvent):
    TYPE: ClassVar[PacketType] = PacketType.PEER_JOIN


@dataclass(slots=True)
class FlagSet(Message):
    """A scene flag turned on. This is where a remake differs from SMOO: SMOO syncs
    a handful of hand-picked things (moons, captures), OoT has thousands of flags
    behind `gSaveContext`, so we sync named flag groups instead of individual
    actors."""

    TYPE: ClassVar[PacketType] = PacketType.FLAG_SET
    BODY: ClassVar[struct.Struct] = struct.Struct("<HBBHII")

    player_id: int
    group: int              # 0 = scene switch flags, 1 = temp switch, 2 = clear flags...
    scene_id: int
    index: int              # u16, flag index within the group
    bitfield: int           # u32, which bits inside that group word changed
    value: int              # u32, the new value of those bits


@dataclass(slots=True)
class ItemGive(Message):
    TYPE: ClassVar[PacketType] = PacketType.ITEM_GIVE
    BODY: ClassVar[struct.Struct] = struct.Struct("<HBBIh")

    player_id: int
    item_id: int            # u8, the game's own ItemID
    slot: int               # u8, inventory slot it landed in (0xFF = none)
    flags: int              # u32, F_* of the give action (from chest, from NPC...)
    delta: int              # s16, ammo delta; 0 for non-ammo


@dataclass(slots=True)
class Ping(Message):
    """Latency probe. `client_ms` is milliseconds since the *client* started, not a
    wall clock: a u32 of epoch milliseconds overflows every 49.7 days and a wall
    clock is not monotonic anyway. The server echoes it unchanged."""

    TYPE: ClassVar[PacketType] = PacketType.PING
    BODY: ClassVar[struct.Struct] = struct.Struct("<II")

    nonce: int
    client_ms: int


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

_REGISTRY: dict[PacketType, type[Message]] = {}


def register(cls: type[Message]) -> type[Message]:
    _REGISTRY[cls.TYPE] = cls
    return cls


for _cls in (PlayerState, Hello, Welcome, PeerEvent, PeerJoin, FlagSet, ItemGive, Ping):
    register(_cls)


_FORMAT_CHARS = re.compile(r"\d*[A-Za-z?]")


def _format_chars(fmt: str) -> list[str]:
    """Format characters of a struct format string, one per packed value."""
    return [m.group(0).lstrip("0123456789") for m in _FORMAT_CHARS.finditer(fmt.lstrip("<>!="))]


class ProtocolError(ValueError):
    pass


def encode(msg: Message, flags: int = 0) -> bytes:
    """`flags` is transport metadata, not message data: bit 0 means "this went over
    a reliable channel". The receiver may ignore it; a UDP-only build does."""
    body = msg.pack_body()
    if len(body) > MAX_BODY:
        raise ProtocolError(f"body of {len(body)} bytes exceeds {MAX_BODY}")
    return HEADER.pack(MAGIC, VERSION, int(msg.TYPE), len(body), flags) + body


def decode(datagram: bytes) -> Message | None:
    """Returns the message, or None for a type this build does not know.

    Raises ProtocolError on anything malformed, so a caller can count bad packets
    instead of silently acting on a half-parsed one.
    """
    if len(datagram) < HEADER_SIZE:
        raise ProtocolError("short header")
    magic, version, ptype, size, _flags = HEADER.unpack_from(datagram)
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if version != VERSION:
        raise ProtocolError(f"protocol version {version}, this build speaks {VERSION}")
    if size > MAX_BODY or HEADER_SIZE + size > len(datagram):
        raise ProtocolError(f"declared body {size} does not fit in {len(datagram)} bytes")
    cls = _REGISTRY.get(PacketType(ptype)) if ptype in PacketType._value2member_map_ else None
    if cls is None:
        return None
    if cls.BODY.size != size:
        raise ProtocolError(
            f"{cls.__name__} expects {cls.BODY.size} body bytes, got {size}"
        )
    return cls.unpack_body(datagram[HEADER_SIZE : HEADER_SIZE + size])


def describe(msg: Message) -> dict[str, Any]:
    """Flat dict for logging and for the dashboard."""
    out: dict[str, Any] = {"type": msg.TYPE.name, "size": HEADER_SIZE + msg.BODY.size}
    for f in fields(msg):
        value = getattr(msg, f.name)
        if isinstance(value, bytes):
            value = value.rstrip(b"\x00").decode("utf-8", "replace")
        elif isinstance(value, float):
            value = round(value, 3)
        out[f.name] = value
    return out
