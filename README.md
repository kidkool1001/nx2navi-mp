# nx2navi-mp

An attempt at **Super Mario Odyssey Online, but for The Legend of Zelda: Ocarina
of Time on the Switch 2, running in [Lichii](https://github.com/zortos293/Lichii)**.

SMOO is four layers stacked: an Atmosphère/Starlight loader, a set of fixed-offset
code hooks, injected puppet actors that subclass the game's own `LiveActor`, and a
relay server with TCP for session data and UDP for state. Three of those need a
cracked-open game and a loader that a Switch 2 title does not have yet. The fourth
is just packets, and it is the layer you can build and test today, with no game
access at all.

So that is what lives here: **the transport, plus the tools to do the
reverse-engineering work that everything else depends on.**

- [`docs/ocarina-online-via-lichii.md`](docs/ocarina-online-via-lichii.md) — the
  full plan: what SMOO actually is, what Lichii's debug server can and cannot do,
  three paths (state bridge now, in-game puppets later, Ship of Harkinian if you
  just want to play), the mailbox design, milestones, and the one upstream change
  that would unlock the whole thing.
- `tools/nx2navi/net/` — relay server, wire protocol (one file, ~250 lines,
  stdlib), WebSocket dashboard.
- `tools/nx2navi/probe.py` — talks to Lichii's debug server: status, memory search
  for Link's position, a 20 Hz sampler that can publish to the relay, input
  injection.
- `tools/nx2navi/send_test.py` — a fake game client, so the relay is testable
  without the console.

## Try it

```sh
# Relay + dashboard, with three fake Links walking around
python3 tools/nx2navi/net/server.py --port 8099 --simulate 3
# open http://localhost:8099/

# A real client joining the room
python3 tools/nx2navi/send_test.py --players 2

# Tests
python3 -m unittest discover -s tools/nx2navi/tests -t . -v
```

With Lichii running a title and `EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545` set:

```sh
python3 tools/nx2navi/probe.py status
python3 tools/nx2navi/probe.py find --x 1234.5 --y 0.0 --z -567.0
python3 tools/nx2navi/probe.py stream --address 0x81234567 --hz 20 --relay 127.0.0.1:8099
```

That last one puts a read-only feed of your own game into the room, which is the
honest version of "multiplayer" until the in-game puppets exist.

Nothing in this repository contains game data, keys or firmware, and it never
will. You need your own dump to use it for real.
