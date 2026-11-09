# Ocarina of Time Online, for the Switch 2 remake, on Lichii

**Short answer.** Odyssey Online works because it is a *runtime code patch*: a
Starlight module is injected into the game process, it hooks about sixty
hardcoded offsets in `main`, replaces the actor factory so it can spawn puppet
actors, and talks to a relay server. Every one of those four layers is available
on Switch 1 with Atmosphère and none of them are available for a Switch 2 title
yet. Lichii does give you something SMOO never had: an opt-in debug server that
can read guest memory, inject controller input and take screenshots at runtime.
That is enough to build a *state bridge*, and it is exactly the reconnaissance
toolkit you need for the code patch. So the realistic plan is:

1. **Path A — state bridge (no code patching).** Read Link's position, animation
   and save flags out of guest memory, ship them to a relay, and give players an
   out-of-game view of each other plus shared progress. Doable now. This repo
   implements the transport half of it.
2. **Path B — in-game puppets (byte patch + a mailbox).** Fit a small ARM64 patch
   into the existing executable image that publishes Link's state to a mailbox in
   guest memory and consumes peer state from another mailbox; do all networking on
   the PC side. This needs one addition to Lichii (`write_memory`), plus symbols
   for the remake's `main`. This is the SMOO-equivalent result.
3. **Path C — play OoT co-op today.** Ship of Harkinian plus the Anchor co-op mod
   already does this, in source, for free, on PC. If the goal is "play with
   friends", stop reading and go do that.

The rest of this document explains each layer with enough detail to start, and
names the specific hard parts instead of hand-waving them.

---

## 1. What Super Mario Odyssey Online actually is

Read as a system, SMOO is four separable layers. It is worth naming them,
because they have wildly different portability.

| Layer | What it does | Portability to Switch 2 |
|---|---|---|
| **Loader** | Starlight installs as an Atmosphère IPS patch that relocates the game's entry point, then loads `subsdk1.elf` as an extra module. | **Blocked.** No Atmosphère equivalent, and Lichii's mod support is IPS/`.pchtxt` byte patches into existing ExeFS files — it cannot map a new segment or a new module. |
| **Hooks** | `patches/*.slpatch` rewrites call sites at fixed addresses, e.g. `4B5E30 B ProjectActorFactory` (replace the actor factory constructor with a custom one), `5359D8 BL tryInitSocket` (start the socket at `GameSystem::init`), `20CA78 BL startCounterHook` (HUD). | **Needs symbols.** Those offsets come from `OdysseyReversed`. There is no equivalent map for the OoT remake; you have to derive it. Byte patches can still hook the same way, but only inside the existing image. |
| **Gameplay injection** | New `PuppetActor`/`PuppetHackActor` classes subclass the game's `LiveActor`; the patched factory makes `Client::getPuppet(i)` spawnable; animation is applied by name (`startAction`, `setSklAnimBlendWeight`). | **Hardest part, and it is engine-shaped, not platform-shaped.** OoT's actor system is simpler than SMOO's (`Actor`/`ActorProfile`, an overlay table, one `Player` instance), so cloning a "puppet Link" is *more* tractable than in Odyssey — the 2019 OoT Online mod did it on N64. |
| **Netcode** | TCP for reliable session data, UDP with hole punching for the 20 Hz position stream; a relay server (`ooms-server`) with rooms; `nn::account::Uid` as identity; a `GameModeBase` abstraction for hide-and-seek etc. | **Fully portable.** It is just packets. This is what this repo builds first. |

The lesson: the netcode is the part you can build and test with no game access at
all, and the gameplay injection is the part that needs months of reverse
engineering. Build them independently, and make the boundary between them a
*dumb, stable buffer of bytes*, so the reverse-engineering effort never blocks on
the networking and vice versa.

---

## 2. What Lichii gives you, and what it deliberately does not

Lichii (`zortos293/Lichii`, branch `nx2`) is a yuzu/Eden descendant with an opt-in
debug server. Set `EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545` before launching a title
and it starts a loopback JSON-lines server once a game has loaded. From its own
`docs/nx2/debug-mcp.md`:

**Available:** `status`, `screenshot`, `frame_times`, `log_tail`, `log_filter`,
`capture_frame`, `capture_image`, `reject_shaders`, `read_memory`,
`search_memory`, `read_gpu_memory`, `input`, `input_sequence`, `speed_limit`,
`sample_threads`.

**Explicitly not available:** any memory *write*; calling into guest code; save
states (the doc says the core cannot serialise kernel, CPU and GPU state). Reads
are capped at 64 KiB per call and are read-only by construction — the server
walks the guest page table and copies out, it cannot map host memory or reach the
key area.

That shape dictates the design:

- **You can observe a running game precisely.** `search_memory` over the heap
  plus `read_memory` with `as: "f32"` is a poor man's Cheat Engine, and
  `input_sequence` means you can drive the game deterministically to reproduce a
  spot without touching the controller. Two 20 Hz samplers plus one `read_memory`
  call per tick is a real position feed.
- **You cannot make the game draw anything new.** No writes, no code execution.
  Any "another Link appears on screen" feature is impossible on Path A, full
  stop. Anything that pretends otherwise is a mockup.
- **One missing primitive is the whole difference.** A gated `write_memory` would
  turn a read-only observer into a full plugin API: a guest-side patch becomes a
  handful of instructions (publish here, read there) and all logic, UI and
  networking move to the PC where you can iterate in seconds. Section 6 sketches
  that patch.

Also relevant: Lichii applies per-title mods through the inherited yuzu
`PatchManager` — LayeredFS RomFS replacement, which it extended to Switch 2 NUP
containers (build 95), and ExeFS byte patches. A `.pchtxt` is a list of
`offset = bytes` rewrites, which is enough for hooking and for code caves inside
the existing image; it is *not* enough to load a new module the way Starlight
does. So Path B buys you the *same* hooks as SMOO minus its loader, and the
mailbox below is how you compensate for the missing part.

> Reality check on the game itself: the remake is a 24.9 GB Switch 2 exclusive
> that shipped 5 Nov 2026. Publicly there is no symbol map, no decompilation and
> no build-id list for it. Everything in Path B starts with you producing that
> yourself, from your own dump, on your own machine.

---

## 3. Path A — the state bridge (buildable now)

**Goal:** your friends can see where you are and what you have done, without the
game knowing anything about it. Useful on its own (a co-op companion screen, a
race tracker, a route checklist that ticks itself), and it is the lab in which you
find the addresses Path B will hardcode.

### 3.1 Find Link's position

OoT stores the player's position as three `f32`s, and the game keeps several
copies (world, room-relative, previous frame). The sequence with no debugger:

1. Stand still in a distinctive spot; note the coordinates from a save-state
   viewer of the N64 original, or from any coordinate display you trust.
2. `search_memory` for the float triple (little endian: `x`, `y`, `z` as 12 bytes).
3. Walk a few units, search again with the new triple.
4. Intersect. Three rounds usually leaves one or two addresses; step onto a
   platform to separate `y` from a duplicated copy.

`tools/nx2navi/probe.py find` does one round and prints the hits; the intersection
is three lines of set arithmetic you should do by hand the first time, because
you need to *learn* the memory layout, not just get an answer.

Then `probe.py stream --address 0x… --hz 20` samples it. If the numbers move
smoothly while you walk and the sample rate holds at 20 Hz with a > 60 fps
present rate, you have a viable feed.

### 3.2 What rides on top

- **Presence:** feed positions to the relay; the dashboard shows a top-down view,
  facing markers and a per-peer loss counter. `tools/nx2navi/net/dashboard.html`
  does this today.
- **Progress:** save flags are booleans in a big structure. Find the chest flag by
  flipping it (open a chest) and diffing `read_memory` dumps of the save region
  before and after. Once you have one, the offsets of its neighbours follow, and
  you can sync "who has which dungeon key" without understanding the game's code.
- **Deterministic replay:** `input_sequence` records a route as
  `[{right_stick:[…],frames:60}, …]`, which is both a test and a bug report.

### 3.3 Limits you must accept

No in-game avatars, no shared enemies, no anti-cheat-free PvP, a sampling rate
bounded by round-trip latency to the debug socket (fine at 20 Hz, bad at 120), and
an address table that breaks on every game update and on any build with different
memory layout. Version the address table by build id and refuse to run when it
does not match, or you will chase ghosts.

---

## 4. Path B — in-game puppets (the SMOO-equivalent)

Everything here needs a dumped executable you own and a weekend with Ghidra.
Treat it as the destination, not the starting line.

### 4.1 Prerequisites

- **The image.** Unpacked ExeFS or an extracted `main` from your own dump, plus
  the update's `main` if the game ships a code update (it does).
- **Symbols.** You are starting from zero. Lichii's repo carries `tools/nx2/`
  helpers (NSO symbol lookup, NPDM dumps, shader tools); use them to resolve
  addresses, and keep a symbol file keyed by build id from day one.
- **A way to run patches.** Lichii's patch manager applies per-title RomFS
  (LayeredFS) *and* ExeFS patches — the same `PatchManager` upstream yuzu uses, so
  IPS / `.pchtxt` byte patches are the mechanism. Confirm the exact mod folder
  layout for your build in its user docs before you write one; it has moved
  between builds and an issue is open about romfs mods failing to boot on 117.

### 4.2 The four things you must locate, in order

1. **The frame update that touches the player.** Find the function that writes the
   position triple you found in Path A. Every `str s0/Sx` that stores to those
   three consecutive words narrows it fast. That function is your publish hook.
2. **The actor spawn path.** In OoT the actor system allocates from a profile
   table and calls `Actor_Init`. You need to be able to create a second instance of
   the player actor with its input and camera disabled. Find `Actor_Spawn`'s
   callers and the profile table; the player profile is the one the game uses at
   file select and after every loading zone.
3. **The save flag writer.** One function sets scene switch flags. Hook it and you
   can both publish "chest opened" and apply a peer's "chest opened".
4. **Padding or a sacrificial function for a code cave.** `main` is large and has
   alignment padding, dead stubs and switch tables. You need a few hundred bytes of
   contiguous space, and everything you write must be position independent
   (PC-relative `adrp`/`add`/`b`), because you cannot add relocation entries.

### 4.3 The mailbox

Do not put a network stack in the guest. Switch 2 socket services are exactly the
kind of thing that is half-implemented in an emulator, and debugging it through
two layers of abstraction is miserable. Instead define two fixed buffers:

```
guest address 0xM+0x000   publish region:    [seq u32][PlayerState 34B][flags u32]
guest address 0xM+0x100   consume region:    [seq u32][peer count u32][peer[N] ...]
```

The patch's only jobs are: on the player update, `str` the current state into the
publish region; on the puppet update, read the consume region and drive `N`
puppet actors. Everything else — protocol, rooms, NAT, interpolation, persistence,
UI — lives in a Python process on the same PC, which can be changed without
restarting the game.

Filling the consume region requires a guest memory *write* from the host, which is
the one primitive Lichii's debug server lacks. Section 6 has the sketch.

### 4.4 Making puppets not be terrible

This is where the project lives or dies, and it is all polish work with clear
failure modes:

- **Interpolation.** Peers arrive at 20 Hz, you render at 60. Buffer two states,
  render ~100 ms behind, lerp position and slerp yaw. Without this, puppets
  teleport.
- **Animation.** Do not sync bone transforms. Sync a *named* animation id and a
  playback frame, then play it locally on the puppet's own skeleton. Reuse the
  game's own anim table; map ids by watching `read_memory` while triggering known
  states (walk, run, roll, climb, fall, swim, Z-target, ocarina).
- **Culling.** Puppets must only exist in the room the player is in. On a loading
  zone, destroy every puppet and rebuild from the peer's next state. Get this
  wrong and you will write to freed actor memory — the classic crash in every
  online mod.
- **Camera and input.** The player actor's update touches the camera and reads
  controller input. A puppet must be constructed in a "dumb" mode from the first
  frame, or it will fight the real Link for the camera.
- **World state.** Two players opening the same chest, completing the same
  cutscene, or warping at the same time must converge. Choose one owner per flag
  group (first writer wins, everyone else applies) and never apply a flag during a
  cutscene that is already running.
- **Age split.** Child and adult Link are different actor profiles in the same
  scene in OoT. Decide up front whether a session is single-age (simplest,
  honest) or mixed, and if mixed, whether the adult puppet is visible to the child
  (in the original, no).

### 4.5 Scope it like a release, not a demo

The reason SMOO feels complete is a `GameModeBase` abstraction: "explore together",
"hide and seek", "tag" are all the same transport with different rules about what
is shared. Do the same. Get "walk around Hyrule Field together, see each other,
open chests together" working and *stop*. Then add modes.

---

## 5. Path C — the boring correct answer

Ship of Harkinian (the OoT decompilation port) has had a co-op mod since 2026:
**Anchor**, a mod plus a small Deno relay server, now folded into SoH builds. It
syncs save state, item gives and flag sets, and it is open source — which makes it
the single best reference implementation for the *semantics* of OoT multiplayer,
even if you never run it. Read its packet protocol before you design yours; it is
a few hundred lines and it has already made the mistakes for you.

If your actual goal is playing OoT with a friend this weekend, use that. If your
goal is "the remake on Lichii", Path B is the only route, and Path A is its
scaffolding.

---

## 6. The one upstream change that unlocks everything

A `write_memory` tool in Lichii's debug server, opt-in and off by default, exactly
mirroring the existing `read_memory`:

- same page-table walk (`Memory::GetPointerSilent`), but writing through
  `RasterizerCachedMemory` so GPU-visible data stays coherent;
- same 64 KiB cap, same `EDEN_NX2_DEBUG_SERVER` gating, same loopback-only rule;
- refuses the key area, the NPDM and anything outside the guest address space —
  which the read path already does by construction, since an address has to
  resolve through a page the guest mapped;
- a separate switch, `EDEN_NX2_DEBUG_ALLOW_WRITE=1`, so nobody gets it by accident.

Sketch against `src/core/debugger/nx2_debug_tools.cpp`:

```cpp
// Same shape as ReadMemory; the only new thing is a store and a cache invalidation.
json WriteMemory(ToolState& state, const json& params) {
    if (!Settings::values.nx2_debug_allow_write.GetValue()) {
        throw std::runtime_error{"write_memory is off; set EDEN_NX2_DEBUG_ALLOW_WRITE=1"};
    }
    const u64 address = AddressParam(params, "address");
    const std::vector<u8> bytes = ParseHexBytes(params.at("data").get<std::string>());
    if (bytes.size() > MAX_READ) {
        throw std::runtime_error{fmt::format("write of {} bytes exceeds {}", bytes.size(), MAX_READ)};
    }
    Memory::Memory& memory = Application(state).GetMemory();
    if (const auto unmapped = WriteGuestMemory(memory, address, bytes)) {
        throw std::runtime_error{fmt::format("guest address {:#x} is not mapped", *unmapped)};
    }
    return {{"address", HexAddress(address)}, {"size", bytes.size()}};
}
```

With that in place, the whole mod becomes: a 200-byte `.pchtxt` patch that copies
state to a known address and reads it back, plus Python on the host. No in-guest
socket code, no in-guest UI, no rebuild to change a packet layout.

Two smaller asks that pay for themselves immediately: a `call_guest_function` (or
`read_memory` with a "follow pointer chain" form) for the address table, and a
way to mark a memory region as "watch and tell me what changed" so finding flag
offsets is a diff instead of a loop.

---

## 7. Milestones

| # | Milestone | Deliverable | Acceptance test | Effort |
|---|---|---|---|---|
| M0 | Transport | `tools/nx2navi/net/` (this repo) | Two fake clients see each other; late joiner receives the journal | done |
| M1 | Recon | `probe.py find/stream` | 20 Hz position feed, stable for 10 minutes, no dropped samples | 1–2 weeks of evenings |
| M2 | Mailbox | `write_memory` in Lichii + a `.pchtxt` that publishes state | Host reads exact position from the mailbox while the game runs | 1 week |
| M3 | Publish hook | patch at the player update | mailbox updates every frame, not every second | 1–3 weeks |
| M4 | Puppets | spawn/despawn a puppet on room change | a second instance of Lichii sees a moving Link, no crash across 20 loading zones | 1–3 months |
| M5 | State sync | flags/items | chest opened by A is open for B; save files stay valid | 2–4 weeks |
| M6 | Polish | interpolation, nametags, modes, NAT | playable with a friend over the internet | open-ended |

The estimates assume one experienced person who already knows ARM64 and the OoT
actor model, working evenings. Everything before M4 is a real project you can
finish; M4 is the part that eats people.

---

## 8. Testing without two consoles

- **Two instances of Lichii on one PC**, two user folders, two game copies, both
  pointed at `127.0.0.1:6545`/`:6546`. This is the main harness.
- **The simulator** (`python3 tools/nx2navi/net/server.py --simulate 3`) makes the relay
  and dashboard testable with no game at all, so the transport never becomes the
  thing you are debugging at 2 a.m.
- **`input_sequence`** reproduces a bug: record the route, replay it against both
  builds, compare `screenshot`s. Any diff is yours, not the tester's.
- **Desync detection as a feature:** hash the save-flag groups on both sides once
  a second and log a mismatch with the first differing group. Cheap, and it turns
  "the world went weird after an hour" into a line number.

---

## 9. The parts nobody can do for you

- **Legal.** This all assumes you dump your own game and keys. Ship patches, never
  game data. Nintendo's terms are hostile to modified consoles and to online
  play; the emulator's own contributors keep that boundary for a reason.
- **No public symbol map.** Until someone publishes one, every address in Path B
  is your own labour, and it breaks on the next update. Keep your symbol file in
  version control, keyed by build id.
- **The remake is not the N64 game.** Even with actors mapped, its save layout,
  its animation table and its cutscene system are the remake's own. Copy the
  *architecture* from OoT Online and Anchor, not their tables.
- **Do not promise in-game avatars before M4 is green.** A dashboard and a
  companion view are honest and useful; a video mockup of two Links in Hyrule
  Field is not.

---

## 10. If you want to start tonight

```sh
# Terminal 1: the relay, with three fake Links so the dashboard has content
python3 tools/nx2navi/net/server.py --port 8099 --simulate 3

# Terminal 2: a fake game client that joins the room for real
python3 tools/nx2navi/send_test.py --players 2

# Then open the dashboard, and with Lichii running:
python3 tools/nx2navi/probe.py status
python3 tools/nx2navi/probe.py find --x 1234.5 --y 0.0 --z -567.0
```

Read Anchor's protocol next, then go find the player update function. Nothing in
Path B is blocked on anything except that one address.

---

## Appendix: how confident is each claim?

Labelled so you can re-check the parts that matter to you instead of trusting a
document that has outlived its sources.

| Claim | Confidence | Basis |
|---|---|---|
| SMOO is a Starlight module + IPS patch, hooks ~60 fixed offsets, swaps the actor factory | **Read from source** | `patches/codehook.slpatch`, `source/main.cpp`, `include/puppets/*` |
| SMOO's netcode is TCP session + UDP state with hole punching | **Read from source** | `source/server/SocketClient.cpp` |
| Lichii's debug server exposes read/input/screenshot and **not** writes | **Read from its own docs** | `docs/nx2/debug-mcp.md` (phases 1–3, "read only", "cannot write either") |
| Lichii applies LayeredFS to Switch 2 NUP titles; ExeFS patches go through `PatchManager` | **Read from its PRs/issues** | PR #203, #686, issue #873 |
| The remake has no public symbols, decompilation or build-id list | **Believed, not proven** | absent from zeldaret/oot (N64), SoH (N64), and no search hit; verify before planning |
| The remake's internals resemble the N64 game closely enough to port OoT Online's actor work | **Unknown — the main risk** | It is a remake, not a port. Assume the actor model is similar and the tables are not |
| Build/dump details circulating on launch day | **Reported, unverified** | press coverage; do not build on them |
