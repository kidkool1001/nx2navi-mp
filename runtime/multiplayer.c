// Multiplayer (multiplayer.h): UDP host/relay and client play over LAN or the internet, local Link state and skeleton
// capture, remote player interpolation and Vulkan draw re-emission, plus the state the menu's Multiplayer tab and the
// in-world nametags read.
//
// Networking. One non-blocking socket per session (platform/platform.h plat_udp_*). The host opens MP_PORT and relays
// every state, skeleton and beacon packet it gets from one client to all the others, so a session is one star: clients
// talk to the host only. Packet types:
//   discover  a client's LAN broadcast ("which sessions are out there"), answered by an announce
//   announce  a host's answer: protocol, port, player count and name (the Multiplayer tab's session browser)
//   hello     a client's join request (protocol, name, color), repeated until welcomed
//   welcome   the host's answer: the player's id, slot and the roster; status CONNECTED
//   reject    the host is full, or the protocol differs
//   state     one player's position, velocity, orientation and action flags, MP_STATE_HZ times a second
//   skel      the animated skeleton (c5, gsys_skeleton) in chunks of MP_BONES_PER_PKT bones, MP_SKEL_HZ times a second
//   beacon    a 3D world marker that fades over MP_BEACON_S seconds
//   bye       leaving: the host drops the slot (from a client) or closes the session (from the host)
//   ping/pong round-trip measurement for the roster's latency column
// Numbers are little-endian.
//
// Local Link. The actor walk of frame_rate.c player_pos finds the actor named "Player"; this file reads its position
// (actor + 0x2b4) and orientation (3x3 at actor + 0x290) at MP_STATE_HZ and derives the velocity from consecutive
// positions (the game keeps no velocity this port reads). The skeleton comes from the uniform blocks local Link's own
// draws bind: vk_exec_draw.c hands every scene draw's c5 and c7 to mp_vk_link_draw, which recognises Link's parts by
// their c7 world translation and keeps the bones. Bones travel as int16: rows at 1/8192 of a unit length, translations
// at 1/16 of a unit relative to the capture position.
//
// Remote players. A received state is buffered with its arrival time; the session thread interpolates MP_INTERP_MS
// behind the newest packet, so 30 Hz motion plays smoothly at any frame rate. Rendering re-emits local Link's own scene
// draws with the remote player's skeleton (c5) and world matrix (c7) built into scratch memory
// (mp_vk_build_remote_ubos): the capture's reference frame is mapped onto the player's interpolated one, so the model
// stands where that player stands and plays the animation that arrived for it.
//
// Test modes: --probe-mp (probe_mp) checks the codecs, the transform math, a loopback host and client session, LAN
// discovery and the camera projection without game files; --mp-server runs a dedicated relay host.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "layout.h"
#include "multiplayer.h"
#include "platform/platform.h"

// vk_usnap.c: the uniform block's bytes as of the submit, from the snapshot's host copy, or NULL.
const uint8_t *vk_usnap_bytes(uint64_t addr, uint64_t n);

// ---- constants

#define MP_PROTOCOL 1
#define MP_MAGIC 0x504d324eu   // "N2MP"
#define MP_MAX_PKT 1400u
#define MP_TICK_MS 16u
#define MP_STATE_HZ 30
#define MP_SKEL_HZ 12
#define MP_HELLO_MS 400u
#define MP_JOIN_TIMEOUT_MS 6000u
#define MP_PEER_TIMEOUT_MS 4000u
#define MP_HIST 8              // buffered state snapshots per player (about a quarter second at 30 Hz)
#define MP_BEACON_S 12.0f
#define MP_LINK_RADIUS 4.0f    // world units: a part's c7 world translation this near local Link is one of its parts
#define MP_DEFAULT_INTERP_MS 100.0f

enum {
    MP_PKT_DISCOVER = 1,
    MP_PKT_ANNOUNCE = 2,
    MP_PKT_HELLO = 3,
    MP_PKT_WELCOME = 4,
    MP_PKT_REJECT = 5,
    MP_PKT_STATE = 6,
    MP_PKT_SKEL = 7,
    MP_PKT_BEACON = 8,
    MP_PKT_BYE = 9,
    MP_PKT_PING = 10,
    MP_PKT_PONG = 11,
};

enum { MP_REJECT_FULL = 1, MP_REJECT_PROTOCOL = 2 };

// ---- byte buffers, little-endian

typedef struct {
    uint8_t *p;
    size_t n, cap;
    int bad;   // a write past cap: the packet is dropped
} MpWr;

typedef struct {
    const uint8_t *p;
    size_t n, at;
    int bad;   // a read past n: the packet is rejected
} MpRd;

static void wr_init(MpWr *w, uint8_t *buf, size_t cap) { w->p = buf, w->n = 0, w->cap = cap, w->bad = 0; }

static void wr_bytes(MpWr *w, const void *src, size_t n) {
    if (w->bad || w->n + n > w->cap) return (void)(w->bad = 1);
    if (n) memcpy(w->p + w->n, src, n);
    w->n += n;
}

static void wr_u8(MpWr *w, uint32_t v) {
    uint8_t b = (uint8_t)v;
    wr_bytes(w, &b, 1);
}

static void wr_u16(MpWr *w, uint32_t v) {
    uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
    wr_bytes(w, b, sizeof b);
}

static void wr_u32(MpWr *w, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    wr_bytes(w, b, sizeof b);
}

static void wr_f32(MpWr *w, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    wr_u32(w, bits);
}

static void wr_str(MpWr *w, const char *s) {
    size_t n = s ? strlen(s) : 0;
    if (n > MP_NAME_MAX - 1) n = MP_NAME_MAX - 1;
    wr_u8(w, (uint32_t)n);
    wr_bytes(w, s, n);
}

static void rd_init(MpRd *r, const void *buf, size_t n) { r->p = buf, r->n = n, r->at = 0, r->bad = 0; }

static void rd_bytes(MpRd *r, void *out, size_t n) {
    if (r->bad || r->at + n > r->n) return (void)(r->bad = 1);
    if (n) memcpy(out, r->p + r->at, n);
    r->at += n;
}

static uint32_t rd_u8(MpRd *r) {
    uint8_t b = 0;
    rd_bytes(r, &b, 1);
    return b;
}

static uint32_t rd_u16(MpRd *r) {
    uint8_t b[2] = {0, 0};
    rd_bytes(r, b, sizeof b);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8;
}

static uint32_t rd_u32(MpRd *r) {
    uint8_t b[4] = {0, 0, 0, 0};
    rd_bytes(r, b, sizeof b);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static float rd_f32(MpRd *r) {
    uint32_t bits = rd_u32(r);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void rd_str(MpRd *r, char *out, size_t cap) {
    uint32_t n = rd_u8(r);
    if (!cap) return;
    memset(out, 0, cap);
    if (r->bad || n > cap - 1) return (void)(r->bad = 1);
    rd_bytes(r, out, n);
}

// ---- 3x3 rotations (row-major, as the game stores them at actor + 0x290) and 4x4 matrices

static void mat3_identity(float out[9]) {
    memset(out, 0, 9 * sizeof *out);
    out[0] = out[4] = out[8] = 1;
}

static void mat3_mul(const float a[9], const float b[9], float out[9]) {
    float t[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            t[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    memcpy(out, t, sizeof t);
}

static void mat3_transpose(const float m[9], float out[9]) {
    float t[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) t[r * 3 + c] = m[c * 3 + r];
    memcpy(out, t, sizeof t);
}

static void mat3_mul_vec(const float m[9], const float v[3], float out[3]) {
    float t[3] = {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
                  m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
    memcpy(out, t, sizeof t);
}

// v * m, the row-vector convention the game's 3x4 matrices are in (a point p maps to p * R + t).
static void mat3_mul_vec_row(const float m[9], const float v[3], float out[3]) {
    float t[3] = {v[0] * m[0] + v[1] * m[3] + v[2] * m[6], v[0] * m[1] + v[1] * m[4] + v[2] * m[7],
                  v[0] * m[2] + v[1] * m[5] + v[2] * m[8]};
    memcpy(out, t, sizeof t);
}

// Gram-Schmidt: a lerped rotation back to unit rows at right angles (interpolating rotations must not shrink them).
static void mat3_orthonormalize(const float in[9], float out[9]) {
    float r0[3] = {in[0], in[1], in[2]}, r1[3] = {in[3], in[4], in[5]}, r2[3];
    float n0 = sqrtf(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]);
    if (!(n0 > 1e-6f) || !isfinite(n0)) return mat3_identity(out);
    for (int i = 0; i < 3; i++) r0[i] /= n0;
    float d = r1[0] * r0[0] + r1[1] * r0[1] + r1[2] * r0[2];
    for (int i = 0; i < 3; i++) r1[i] -= d * r0[i];
    float n1 = sqrtf(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
    if (!(n1 > 1e-6f) || !isfinite(n1)) return mat3_identity(out);
    for (int i = 0; i < 3; i++) r1[i] /= n1;
    r2[0] = r0[1] * r1[2] - r0[2] * r1[1], r2[1] = r0[2] * r1[0] - r0[0] * r1[2], r2[2] = r0[0] * r1[1] - r0[1] * r1[0];
    memcpy(out, r0, sizeof r0);
    memcpy(out + 3, r1, sizeof r1);
    memcpy(out + 6, r2, sizeof r2);
}

// A plausible rotation: finite, right angles, one shared scale (the game's actor matrices are pure rotations).
static int mat3_rotation_like(const float m[9]) {
    float l[3];
    for (int i = 0; i < 9; i++)
        if (!isfinite(m[i]) || fabsf(m[i]) > 1000.0f) return 0;
    for (int r = 0; r < 3; r++) l[r] = sqrtf(m[r * 3] * m[r * 3] + m[r * 3 + 1] * m[r * 3 + 1] + m[r * 3 + 2] * m[r * 3 + 2]);
    for (int r = 0; r < 3; r++)
        if (l[r] < 1e-4f) return 0;
    for (int r = 1; r < 3; r++)
        if (fabsf(l[r] - l[0]) > 0.05f * l[0]) return 0;
    for (int a = 0; a < 3; a++)
        for (int b = a + 1; b < 3; b++) {
            float d = m[a * 3] * m[b * 3] + m[a * 3 + 1] * m[b * 3 + 1] + m[a * 3 + 2] * m[b * 3 + 2];
            if (fabsf(d) > 0.08f * l[a] * l[b]) return 0;
        }
    return 1;
}

// The heading a rotation gives, in radians: where its forward axis (row 2) points in the XZ plane.
static float mat3_yaw(const float m[9]) {
    float x = m[6], z = m[8];
    if (!isfinite(x) || !isfinite(z) || (!x && !z)) return 0;
    return atan2f(x, z);
}

static void mat3_from_yaw(float yaw, float out[9]) {
    float c = cosf(yaw), s = sinf(yaw), m[9] = {c, 0, -s, 0, 1, 0, s, 0, c};
    memcpy(out, m, sizeof m);
}

// ---- the frame transform: the core of remote rendering

// `in_mat` is a 3x4 row-major matrix of the local player's frame, in the game's row-vector convention (a point p maps
// to p * R + t). out_mat is the same geometry placed in the remote player's frame: the geometry turns by the heading
// between the two players (delta = local_rot * remote_rot^T, so that in_mat * delta^T = in_mat * remote_rot * local_rot^T)
// and its translation moves from local_pos to remote_pos. A bone recorded at the local player's hip therefore ends up at
// the remote player's hip, facing their way. local_pos zero places model-space geometry (whose translation is an offset
// from the model's origin) at the remote player's feet. probe_mp checks the identity, a translation and a rotation case.
void mp_transform_mat3x4(const float in_mat[12], const float local_pos[3], const float local_rot[9],
                         const float remote_pos[3], const float remote_rot[9], float out_mat[12]) {
    float lt[9], delta[9], t[3], rt[3];
    mat3_transpose(local_rot, lt);
    mat3_mul(lt, remote_rot, delta);   // local_rot^T * remote_rot, i.e. (remote_rot * local_rot^T)^T
    for (int r = 0; r < 3; r++) {
        float row[3] = {in_mat[r * 4], in_mat[r * 4 + 1], in_mat[r * 4 + 2]}, out[3];
        mat3_mul_vec_row(delta, row, out);
        out_mat[r * 4] = out[0], out_mat[r * 4 + 1] = out[1], out_mat[r * 4 + 2] = out[2];
    }
    t[0] = in_mat[3] - local_pos[0], t[1] = in_mat[7] - local_pos[1], t[2] = in_mat[11] - local_pos[2];
    mat3_mul_vec_row(delta, t, rt);
    out_mat[3] = remote_pos[0] + rt[0];
    out_mat[7] = remote_pos[1] + rt[1];
    out_mat[11] = remote_pos[2] + rt[2];
}

// Model-space geometry (bones whose translation is an offset from the model's origin): the rows already hold each bone's
// orientation in model space, so only the translation is placed.
static void mp_place_model_space(const float in_mat[12], const float remote_pos[3], const float remote_rot[9],
                                 float out_mat[12]) {
    float t[3] = {in_mat[3], in_mat[7], in_mat[11]}, rt[3];
    memcpy(out_mat, in_mat, 12 * sizeof *out_mat);
    mat3_mul_vec_row(remote_rot, t, rt);
    out_mat[3] = remote_pos[0] + rt[0];
    out_mat[7] = remote_pos[1] + rt[1];
    out_mat[11] = remote_pos[2] + rt[2];
}

// ---- session and player structures

typedef struct {
    float pos[3], rot[9], yaw;
    uint64_t t_ns;
} MpSnap;

typedef struct {
    int active;
    uint8_t slot, color;
    uint32_t id;
    char name[MP_NAME_MAX];
    PlatNetAddr addr;          // host side: where this player's packets come from
    uint64_t last_rx_ns;
    uint32_t rtt_ms;
    uint32_t ping_nonce;
    uint64_t ping_sent_ns;
    // newest state as received
    float pos[3], vel[3], rot[9], yaw;
    uint32_t action_flags, seq;
    // interpolation history: hist_head is the newest
    MpSnap hist[MP_HIST];
    int hist_n, hist_head;
    float ipos[3], irot[9], iyaw;   // interpolated: what the renderer, the nametags and the roster use
    // skeleton: the assembled block, and the one being received
    uint32_t skel_bones;
    int skel_model_space;
    float skel_ref_pos[3], skel_ref_rot[9];
    float skel[MP_MAX_BONES * 12];
    float skel_rx[MP_MAX_BONES * 12];
    uint32_t skel_rx_mask[(MP_MAX_BONES + 31) / 32];
    uint32_t skel_rx_have, skel_rx_total, skel_seq;
} MpPeer;

typedef struct {
    MpMode mode;
    MpStatus status;
    PlatUdp *sock;
    uint16_t port;
    PlatNetAddr server;        // client: the host
    uint32_t my_id;
    int my_slot;
    char my_name[MP_NAME_MAX];
    uint8_t my_color;
    char error[128];
    char host_text[64];
    MpPeer peers[MP_MAX_PLAYERS];
    MpLanSession lan[MP_MAX_LAN_HOSTS];
    int lan_n;
    MpBeaconView beacons[MP_MAX_BEACONS];
    uint64_t start_ns, last_state_ns, last_skel_ns, last_hello_ns, last_ping_ns, last_announce_ns;
    uint32_t tick;
    uint64_t tx_pkts, rx_pkts, tx_bytes, rx_bytes, dropped, bad;
    volatile uint64_t re_emitted;
    MpPeer local;              // the host's, or the client's, own player
    int verbose;
} MpSession;

// The local player's captured state: written under g_lock by the session thread (position, rotation) and the replay
// thread (skeleton), and published lock-free for the replay thread's per-draw proximity check.
static struct {
    int valid, actor_ok;
    uint64_t actor;
    float pos[3], vel[3], rot[9], yaw;
    uint32_t action_flags;
    uint32_t skel_bones;
    int skel_model_space;
    float skel_ref_pos[3], skel_ref_rot[9];
    float skel[MP_MAX_BONES * 12];
} L;

static struct {
    float vp[16];              // 4x4 row-major view-projection
    float view[16];
    int valid;
} C;    // the scene camera, for the nametags

static MpSession G;
static PlatMutex g_lock = PLAT_MUTEX_INIT;   // G and L
static volatile int32_t g_active;            // mp_active(): read for every draw of the replay
static volatile int32_t g_overlay;           // mp_overlay_wanted()
static volatile int32_t g_remote;            // the number of remote players (replication candidates)
static volatile uint32_t g_peer_mask;        // bit i: peers[i] is active (lock-free for the replay)
static volatile int32_t g_quit;
static volatile int32_t g_lpos_valid;        // L.pos is current (lock-free copy for the replay)
static volatile float g_lpos[3];
static float g_interp_ms = MP_DEFAULT_INTERP_MS;

static uint64_t now_ns(void) { return plat_now_ns(); }

static uint32_t rand_id(void) {
    static uint32_t n;
    uint64_t t = now_ns();
    n++;
    return (uint32_t)(t ^ (t >> 32) ^ (uint64_t)n * 2654435761u) | 1u;
}

// ---- settings

static uint8_t color_of_name(const char *s) {
    static const char *const names[MP_COLOR_COUNT] = {"green", "blue", "red", "violet", "gold", "cyan"};
    for (int i = 0; s && i < MP_COLOR_COUNT; i++)
        if (!strcmp(s, names[i])) return (uint8_t)i;
    return MP_COLOR_GREEN;
}

static MpMode mode_of(const char *s) {
    if (!s) return MP_MODE_OFF;
    if (!strcmp(s, "host")) return MP_MODE_HOST;
    if (!strcmp(s, "join")) return MP_MODE_JOIN;
    return MP_MODE_OFF;
}

// Re-reads the settings into s (the menu changes them live).
static void mp_apply_cfg(MpSession *s) {
    const char *name = cfg_value("NX2NAVI_MP_NAME"), *port = cfg_value("NX2NAVI_MP_PORT");
    snprintf(s->my_name, sizeof s->my_name, "%s", name && *name ? name : "Link");
    s->my_color = color_of_name(cfg_value("NX2NAVI_MP_COLOR"));
    int p = port ? atoi(port) : 0;
    s->port = (uint16_t)(p >= 1024 && p <= 65535 ? p : MP_DEFAULT_PORT);
    snprintf(s->host_text, sizeof s->host_text, "%s", cfg_value("NX2NAVI_MP_HOST"));
    int ip = cfg_int("NX2NAVI_MP_INTERP_MS");
    g_interp_ms = ip < 0 ? 0 : ip > 300 ? 300 : (float)ip;
    __atomic_store_n(&g_overlay, cfg_bool("NX2NAVI_MP_HUD") || cfg_bool("NX2NAVI_MP_NAMETAGS"), __ATOMIC_RELEASE);
}

// ---- peers

static MpPeer *peer_by_slot(MpSession *s, int slot) {
    if (slot < 0 || slot >= MP_MAX_PLAYERS) return NULL;
    return s->peers[slot].active ? &s->peers[slot] : NULL;
}

static MpPeer *peer_by_id(MpSession *s, uint32_t id) {
    for (int i = 0; i < MP_MAX_PLAYERS; i++)
        if (s->peers[i].active && s->peers[i].id == id) return &s->peers[i];
    return NULL;
}

static MpPeer *peer_by_addr(MpSession *s, PlatNetAddr a) {
    for (int i = 0; i < MP_MAX_PLAYERS; i++)
        if (s->peers[i].active && s->peers[i].addr.ip == a.ip && s->peers[i].addr.port == a.port) return &s->peers[i];
    return NULL;
}

static int peer_count(const MpSession *s) {
    int n = s->status == MP_STATUS_HOSTING || s->status == MP_STATUS_CONNECTED ? 1 : 0;
    for (int i = 0; i < MP_MAX_PLAYERS; i++) n += s->peers[i].active;
    return n;
}

// The host's side of a new connection, or an existing player's entry refreshed.
static void peer_reset(MpPeer *p, int slot, uint32_t id, const char *name, uint8_t color) {
    memset(p, 0, sizeof *p);
    p->active = 1;
    p->slot = (uint8_t)slot;
    p->id = id;
    p->color = color < MP_COLOR_COUNT ? color : MP_COLOR_GREEN;
    snprintf(p->name, sizeof p->name, "%s", name && *name ? name : "Link");
    mat3_identity(p->irot);
    mat3_identity(p->rot);
    mat3_identity(p->skel_ref_rot);
    p->last_rx_ns = now_ns();
}

// The replay thread's view of who is there, refreshed after every change.
static void publish_peers(MpSession *s) {
    uint32_t mask = 0;
    int n = 0;
    for (int i = 0; i < MP_MAX_PLAYERS; i++)
        if (s->peers[i].active) mask |= 1u << i, n++;
    __atomic_store_n(&g_peer_mask, mask, __ATOMIC_RELEASE);
    __atomic_store_n(&g_remote, n, __ATOMIC_RELEASE);
}

// The first free slot from 1 (slot 0 is the host's).
static int peer_alloc_slot(MpSession *s) {
    for (int i = 1; i < MP_MAX_PLAYERS; i++)
        if (!s->peers[i].active) return i;
    return -1;
}

// ---- sending

static int send_pkt(MpSession *s, PlatNetAddr to, const uint8_t *buf, size_t n) {
    if (!s->sock || !n) return 0;
    if (plat_udp_send(s->sock, to, buf, n)) {
        s->tx_pkts++, s->tx_bytes += n;
        return 1;
    }
    s->dropped++;
    return 0;
}

// The host's copy of a client's packet to every other client.
static void relay(MpSession *s, const uint8_t *buf, size_t n, PlatNetAddr from) {
    for (int i = 0; i < MP_MAX_PLAYERS; i++) {
        const MpPeer *p = &s->peers[i];
        if (!p->active) continue;
        if (p->addr.ip == from.ip && p->addr.port == from.port) continue;
        send_pkt(s, p->addr, buf, n);
    }
}

// The host's own state to every client (no source address to skip).
static void relay_own(MpSession *s, const uint8_t *buf, size_t n) {
    for (int i = 0; i < MP_MAX_PLAYERS; i++)
        if (s->peers[i].active) send_pkt(s, s->peers[i].addr, buf, n);
}

static uint64_t g_prev_pos_ns;
static float g_prev_pos[3];
static int g_prev_ok;

// The local player's state packet (also the host's own, which goes to every client).
static void send_state(MpSession *s) {
    if (!s->sock || !L.valid) return;
    uint8_t buf[128];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_STATE);
    wr_u32(&w, s->my_id);
    wr_u32(&w, ++s->local.seq);
    for (int i = 0; i < 3; i++) wr_f32(&w, L.pos[i]);
    for (int i = 0; i < 3; i++) wr_f32(&w, L.vel[i]);
    for (int i = 0; i < 9; i++) wr_f32(&w, L.rot[i]);
    wr_f32(&w, L.yaw);
    wr_u32(&w, L.action_flags);
    if (w.bad) return;
    memcpy(s->local.pos, L.pos, sizeof s->local.pos);
    memcpy(s->local.rot, L.rot, sizeof s->local.rot);
    s->local.yaw = L.yaw;
    s->local.action_flags = L.action_flags;
    if (s->status == MP_STATUS_HOSTING) relay_own(s, buf, w.n);
    else if (s->status == MP_STATUS_CONNECTED) send_pkt(s, s->server, buf, w.n);
}

// Quantize a bone's 3x4 matrix: rows as int16 at 1/8192 of a unit length, the translation as int16 at 1/16 of a unit
// relative to origin.
static void bone_pack(const float m[12], const float origin[3], int16_t out[12]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            float v = m[r * 4 + c] * 8192.0f;
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            out[r * 4 + c] = (int16_t)lrintf(v);
        }
    for (int r = 0; r < 3; r++) {
        float v = (m[r * 4 + 3] - origin[r]) * 16.0f;
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        out[r * 4 + 3] = (int16_t)lrintf(v);
    }
}

static void bone_unpack(const int16_t in[12], const float origin[3], float out[12]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) out[r * 4 + c] = in[r * 4 + c] / 8192.0f;
    for (int r = 0; r < 3; r++) out[r * 4 + 3] = origin[r] + in[r * 4 + 3] / 16.0f;
}

// The skeleton in chunks; a full set of chunks replaces the receiver's assembled skeleton.
static void send_skel(MpSession *s) {
    if (!s->sock) return;
    uint32_t total = L.skel_bones;
    if (total < 1) return;
    if (total > MP_MAX_BONES) total = MP_MAX_BONES;
    uint32_t chunks = (total + MP_BONES_PER_PKT - 1) / MP_BONES_PER_PKT;
    static uint32_t seq;
    seq++;
    for (uint32_t ci = 0; ci < chunks; ci++) {
        uint32_t first = ci * MP_BONES_PER_PKT;
        uint32_t n = total - first < MP_BONES_PER_PKT ? total - first : MP_BONES_PER_PKT;
        uint8_t buf[MP_MAX_PKT];
        MpWr w;
        wr_init(&w, buf, sizeof buf);
        wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_SKEL);
        wr_u32(&w, s->my_id);
        wr_u32(&w, seq);
        wr_u8(&w, total), wr_u8(&w, ci), wr_u8(&w, n);
        wr_u8(&w, L.skel_model_space ? 1 : 0);
        for (int i = 0; i < 3; i++) wr_f32(&w, L.skel_ref_pos[i]);
        for (int i = 0; i < 9; i++) wr_f32(&w, L.skel_ref_rot[i]);
        for (uint32_t k = 0; k < n; k++) {
            int16_t q[12];
            bone_pack(L.skel + (first + k) * 12, L.skel_ref_pos, q);
            for (int i = 0; i < 12; i++) wr_u16(&w, (uint16_t)q[i]);
        }
        if (w.bad) continue;
        if (s->status == MP_STATUS_HOSTING) relay_own(s, buf, w.n);
        else if (s->status == MP_STATUS_CONNECTED) send_pkt(s, s->server, buf, w.n);
    }
}

static uint32_t next_ping(MpPeer *p) {
    p->ping_nonce = rand_id();
    p->ping_sent_ns = now_ns();
    return p->ping_nonce;
}

static void send_ping(MpSession *s, PlatNetAddr to, uint32_t nonce) {
    uint8_t buf[16];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_PING), wr_u32(&w, nonce);
    if (!w.bad) send_pkt(s, to, buf, w.n);
}

static void send_hello(MpSession *s) {
    if (!s->sock) return;
    uint8_t buf[64];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_HELLO);
    wr_u16(&w, MP_PROTOCOL);
    wr_u8(&w, s->my_color);
    wr_str(&w, s->my_name);
    if (!w.bad) send_pkt(s, s->server, buf, w.n);
}

static void send_announce(MpSession *s, PlatNetAddr to) {
    uint8_t buf[64];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    int n = peer_count(s);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_ANNOUNCE);
    wr_u16(&w, MP_PROTOCOL);
    wr_u16(&w, s->port);
    wr_u8(&w, (uint32_t)(n > 255 ? 255 : n));
    wr_u8(&w, MP_MAX_PLAYERS);
    wr_str(&w, s->my_name);
    if (!w.bad) send_pkt(s, to, buf, w.n);
}

static void send_bye(MpSession *s) {
    uint8_t buf[16];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_BYE), wr_u32(&w, s->my_id);
    if (w.bad || !s->sock) return;
    if (s->status == MP_STATUS_HOSTING) relay_own(s, buf, w.n);
    else if (s->status == MP_STATUS_CONNECTED) send_pkt(s, s->server, buf, w.n);
}

static void send_beacon_pkt(MpSession *s, const float pos[3]) {
    if (!s->sock) return;
    uint8_t buf[64];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_BEACON);
    wr_u32(&w, s->my_id);
    wr_u8(&w, s->my_color);
    for (int i = 0; i < 3; i++) wr_f32(&w, pos[i]);
    wr_str(&w, s->my_name);
    if (w.bad) return;
    if (s->status == MP_STATUS_HOSTING) relay_own(s, buf, w.n);
    else if (s->status == MP_STATUS_CONNECTED) send_pkt(s, s->server, buf, w.n);
}

// ---- receiving

static void peer_push_snap(MpPeer *p, uint64_t t) {
    p->hist_head = (p->hist_head + 1) % MP_HIST;
    MpSnap *e = &p->hist[p->hist_head];
    memcpy(e->pos, p->pos, sizeof e->pos);
    memcpy(e->rot, p->rot, sizeof e->rot);
    e->yaw = p->yaw;
    e->t_ns = t;
    if (p->hist_n < MP_HIST) p->hist_n++;
}

static void on_state(MpSession *s, MpRd *r, uint64_t t, PlatNetAddr from) {
    uint32_t id = rd_u32(r), seq = rd_u32(r);
    float pos[3], vel[3], rot[9], yaw;
    for (int i = 0; i < 3; i++) pos[i] = rd_f32(r);
    for (int i = 0; i < 3; i++) vel[i] = rd_f32(r);
    for (int i = 0; i < 9; i++) rot[i] = rd_f32(r);
    yaw = rd_f32(r);
    uint32_t flags = rd_u32(r);
    if (r->bad || !isfinite(pos[0]) || !isfinite(pos[1]) || !isfinite(pos[2])) return (void)s->bad++;
    if (!mat3_rotation_like(rot)) mat3_from_yaw(isfinite(yaw) ? yaw : 0, rot);
    MpPeer *p = peer_by_id(s, id);
    if (!p) return (void)s->bad++;
    if (p->seq && seq <= p->seq && p->seq - seq < 0x40000000u) return;   // an older packet after a newer one
    memcpy(p->pos, pos, sizeof pos);
    memcpy(p->vel, isfinite(vel[0]) && isfinite(vel[2]) ? vel : (float[3]){0, 0, 0}, sizeof vel);
    memcpy(p->rot, rot, sizeof rot);
    p->yaw = isfinite(yaw) ? yaw : mat3_yaw(rot);
    p->action_flags = flags;
    p->seq = seq;
    p->last_rx_ns = t;
    (void)from;
    peer_push_snap(p, t);
}

static void on_skel(MpSession *s, MpRd *r, uint64_t t) {
    uint32_t id = rd_u32(r), seq = rd_u32(r);
    uint32_t total = rd_u8(r), ci = rd_u8(r), n = rd_u8(r), model = rd_u8(r);
    float ref_pos[3], ref_rot[9];
    for (int i = 0; i < 3; i++) ref_pos[i] = rd_f32(r);
    for (int i = 0; i < 9; i++) ref_rot[i] = rd_f32(r);
    if (r->bad || n > MP_BONES_PER_PKT || total < 1 || total > MP_MAX_BONES) return (void)s->bad++;
    uint32_t first = ci * MP_BONES_PER_PKT;
    if (first + n > total) return (void)s->bad++;
    MpPeer *p = peer_by_id(s, id);
    if (!p) return (void)s->bad++;
    p->last_rx_ns = t;
    if (seq != p->skel_seq) {   // a new set of chunks
        p->skel_seq = seq;
        p->skel_rx_total = total, p->skel_rx_have = 0;
        memset(p->skel_rx_mask, 0, sizeof p->skel_rx_mask);
        p->skel_model_space = model != 0;
        memcpy(p->skel_ref_pos, ref_pos, sizeof ref_pos);
        if (mat3_rotation_like(ref_rot)) memcpy(p->skel_ref_rot, ref_rot, sizeof ref_rot);
        else mat3_identity(p->skel_ref_rot);
    }
    for (uint32_t k = 0; k < n; k++) {
        int16_t q[12];
        for (int i = 0; i < 12; i++) q[i] = (int16_t)(uint16_t)rd_u16(r);
        if (r->bad || first + k >= MP_MAX_BONES) continue;
        bone_unpack(q, p->skel_ref_pos, p->skel_rx + (first + k) * 12);
        uint32_t bit = first + k;
        if (!(p->skel_rx_mask[bit / 32] & (1u << (bit % 32)))) p->skel_rx_mask[bit / 32] |= 1u << (bit % 32), p->skel_rx_have++;
    }
    if (r->bad) return (void)s->bad++;
    if (p->skel_rx_have >= p->skel_rx_total) {   // complete: it becomes the skeleton the renderer uses
        memcpy(p->skel, p->skel_rx, sizeof(float) * 12 * p->skel_rx_total);
        p->skel_bones = p->skel_rx_total;
    }
}

static void on_beacon(MpSession *s, MpRd *r, uint64_t t) {
    uint32_t id = rd_u32(r);
    uint8_t color = (uint8_t)rd_u8(r);
    float pos[3];
    for (int i = 0; i < 3; i++) pos[i] = rd_f32(r);
    char name[MP_NAME_MAX];
    rd_str(r, name, sizeof name);
    if (r->bad || !isfinite(pos[0]) || !isfinite(pos[2])) return (void)s->bad++;
    MpPeer *p = peer_by_id(s, id);
    for (int i = 0; i < MP_MAX_BEACONS; i++) {
        MpBeaconView *b = &s->beacons[i];
        if (b->active) continue;
        memset(b, 0, sizeof *b);
        b->active = 1;
        b->slot = p ? p->slot : 0;
        b->color_id = p ? p->color : (color < MP_COLOR_COUNT ? color : MP_COLOR_GREEN);
        snprintf(b->sender, sizeof b->sender, "%s", p ? p->name : name);
        memcpy(b->pos, pos, sizeof b->pos);
        b->duration_s = MP_BEACON_S;
        (void)t;
        return;
    }
}

// ---- the host's roster

static void send_welcome(MpSession *s, PlatNetAddr to, uint32_t id, int slot) {
    uint8_t buf[512];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    int n = 0;
    for (int i = 0; i < MP_MAX_PLAYERS; i++) n += s->peers[i].active && s->peers[i].id != id;
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_WELCOME);
    wr_u32(&w, id);
    wr_u8(&w, (uint32_t)slot);
    wr_u8(&w, MP_MAX_PLAYERS);
    wr_u8(&w, (uint32_t)(n + 1));
    wr_u32(&w, s->my_id), wr_u8(&w, 0), wr_u8(&w, s->my_color), wr_str(&w, s->my_name);
    for (int i = 0; i < MP_MAX_PLAYERS; i++) {
        const MpPeer *p = &s->peers[i];
        if (!p->active || p->id == id) continue;
        wr_u32(&w, p->id), wr_u8(&w, p->slot), wr_u8(&w, p->color), wr_str(&w, p->name);
    }
    if (!w.bad) send_pkt(s, to, buf, w.n);
}

static void send_reject(MpSession *s, PlatNetAddr to, int reason) {
    uint8_t buf[8];
    MpWr w;
    wr_init(&w, buf, sizeof buf);
    wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_REJECT), wr_u8(&w, (uint32_t)reason);
    if (!w.bad) send_pkt(s, to, buf, w.n);
}

// A client's hello: a new roster entry, or the welcome again while it waits for its first one.
static void on_hello(MpSession *s, MpRd *r, PlatNetAddr from, uint64_t t) {
    uint32_t proto = rd_u16(r);
    uint8_t color = (uint8_t)rd_u8(r);
    char name[MP_NAME_MAX];
    rd_str(r, name, sizeof name);
    if (r->bad) return (void)s->bad++;
    if (proto != MP_PROTOCOL) return send_reject(s, from, MP_REJECT_PROTOCOL);
    MpPeer *p = peer_by_addr(s, from);
    if (!p) {
        int slot = peer_alloc_slot(s);
        if (slot < 0) return send_reject(s, from, MP_REJECT_FULL);
        p = &s->peers[slot];
        peer_reset(p, slot, rand_id(), name, color);
        p->addr = from;
        rt_log("mp: '%s' joined slot %d (%d/%d players)", p->name, slot, peer_count(s), MP_MAX_PLAYERS);
        publish_peers(s);
    }
    p->last_rx_ns = t;
    send_welcome(s, from, p->id, p->slot);
}

static void on_welcome(MpSession *s, MpRd *r, uint64_t t) {
    uint32_t id = rd_u32(r);
    uint8_t slot = (uint8_t)rd_u8(r), max = (uint8_t)rd_u8(r), n = (uint8_t)rd_u8(r);
    if (r->bad) return (void)s->bad++;
    if (s->status != MP_STATUS_CONNECTED) {
        rt_log("mp: joined the session at %s as slot %u", s->host_text, (unsigned)slot);
        s->start_ns = t;
    }
    s->my_id = id;
    s->my_slot = slot;
    (void)max;
    s->status = MP_STATUS_CONNECTED;
    s->error[0] = 0;
    for (int k = 0; k < n; k++) {
        uint32_t pid = rd_u32(r);
        uint8_t pslot = (uint8_t)rd_u8(r), pcolor = (uint8_t)rd_u8(r);
        char pname[MP_NAME_MAX];
        rd_str(r, pname, sizeof pname);
        if (r->bad) return (void)s->bad++;
        if (pid == id) continue;
        MpPeer *p = peer_by_id(s, pid);
        if (!p && pslot < MP_MAX_PLAYERS && (pslot == 0 || !s->peers[pslot].active)) {
            p = &s->peers[pslot];
            peer_reset(p, pslot, pid, pname, pcolor);
        }
        if (!p) continue;
        p->last_rx_ns = t;
        snprintf(p->name, sizeof p->name, "%s", pname);
    }
    publish_peers(s);
}

// ---- packets in

static void handle_packet(MpSession *s, const uint8_t *buf, int n, PlatNetAddr from, uint64_t t) {
    if (n < 5) return (void)s->bad++;
    MpRd r;
    rd_init(&r, buf, (size_t)n);
    if (rd_u32(&r) != MP_MAGIC) return (void)s->bad++;
    uint32_t type = rd_u8(&r);
    s->rx_pkts++, s->rx_bytes += (uint64_t)n;
    switch (type) {
    case MP_PKT_STATE: on_state(s, &r, t, from); break;
    case MP_PKT_SKEL: on_skel(s, &r, t); break;
    case MP_PKT_BEACON: on_beacon(s, &r, t); break;
    case MP_PKT_HELLO:
        if (s->status == MP_STATUS_HOSTING) on_hello(s, &r, from, t);
        break;
    case MP_PKT_WELCOME:
        if (s->status == MP_STATUS_CONNECTING || s->status == MP_STATUS_CONNECTED) on_welcome(s, &r, t);
        break;
    case MP_PKT_REJECT: {
        uint32_t reason = rd_u8(&r);
        if (s->status != MP_STATUS_CONNECTING && s->status != MP_STATUS_CONNECTED) break;
        snprintf(s->error, sizeof s->error, "%s", reason == MP_REJECT_FULL ? "session full"
                                              : reason == MP_REJECT_PROTOCOL ? "protocol mismatch"
                                                                             : "the host is not hosting");
        rt_log("mp: the host refused the join: %s", s->error);
        s->status = MP_STATUS_ERROR;
        break;
    }
    case MP_PKT_PING: {
        uint32_t nonce = rd_u32(&r);
        uint8_t out[16];
        MpWr w;
        wr_init(&w, out, sizeof out);
        wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_PONG), wr_u32(&w, nonce);
        if (!w.bad) send_pkt(s, from, out, w.n);
        break;
    }
    case MP_PKT_PONG: {
        uint32_t nonce = rd_u32(&r);
        MpPeer *p = s->status == MP_STATUS_HOSTING ? peer_by_addr(s, from) : &s->local;
        if (p && p->ping_nonce == nonce && p->ping_sent_ns) {
            uint64_t dt = t > p->ping_sent_ns ? t - p->ping_sent_ns : 0;
            p->rtt_ms = dt > 4000000000ull ? 0 : (uint32_t)(dt / 1000000ull);
            p->ping_sent_ns = 0;
        }
        break;
    }
    case MP_PKT_DISCOVER:
        if (s->status == MP_STATUS_HOSTING) send_announce(s, from);
        break;
    case MP_PKT_ANNOUNCE: {
        uint32_t proto = rd_u16(&r), port = rd_u16(&r);
        uint32_t players = rd_u8(&r), max = rd_u8(&r);
        char name[MP_NAME_MAX];
        rd_str(&r, name, sizeof name);
        if (r.bad || proto != MP_PROTOCOL || !max || !port) return;
        char ip[64];
        PlatNetAddr a = from;
        a.port = (uint16_t)port;
        plat_net_format(a, ip, sizeof ip);
        for (int i = 0; i < s->lan_n; i++)
            if (!strcmp(s->lan[i].host_ip, ip)) {   // a known host: refresh it
                s->lan[i].player_count = (uint8_t)players;
                s->lan[i].max_players = (uint8_t)max;
                snprintf(s->lan[i].host_name, sizeof s->lan[i].host_name, "%s", name);
                return;
            }
        if (s->lan_n >= MP_MAX_LAN_HOSTS) return;
        MpLanSession *e = &s->lan[s->lan_n++];
        memset(e, 0, sizeof *e);
        snprintf(e->host_ip, sizeof e->host_ip, "%s", ip);
        e->port = (uint16_t)port;
        snprintf(e->host_name, sizeof e->host_name, "%s", name);
        e->player_count = (uint8_t)players;
        e->max_players = (uint8_t)max;
        break;
    }
    case MP_PKT_BYE: {
        uint32_t id = rd_u32(&r);
        if (s->status == MP_STATUS_HOSTING) {
            MpPeer *p = peer_by_addr(s, from);
            if (!p) p = peer_by_id(s, id);
            if (p) {
                rt_log("mp: '%s' left the session", p->name);
                memset(p, 0, sizeof *p);
                publish_peers(s);
            }
        } else {
            MpPeer *p = peer_by_id(s, id);
            if (p) memset(p, 0, sizeof *p);
            else {
                rt_log("mp: the host closed the session");
                snprintf(s->error, sizeof s->error, "host closed the session");
                s->status = MP_STATUS_ERROR;
            }
            publish_peers(s);
        }
        break;
    }
    default: s->bad++; break;
    }
}

// ---- interpolation

static void snap_lerp(const MpSnap *a, const MpSnap *b, float f, float pos[3], float rot[9], float *yaw) {
    for (int i = 0; i < 3; i++) pos[i] = a->pos[i] + (b->pos[i] - a->pos[i]) * f;
    float m[9];
    for (int i = 0; i < 9; i++) m[i] = a->rot[i] + (b->rot[i] - a->rot[i]) * f;
    mat3_orthonormalize(m, rot);
    *yaw = a->yaw + (b->yaw - a->yaw) * f;
}

// Moves every player's ipos/irot to `t` minus the interpolation window, and ages the beacons.
static void interp_tick(MpSession *s, uint64_t t) {
    double target = (double)t - (double)g_interp_ms * 1e6;
    for (int i = 0; i < MP_MAX_PLAYERS; i++) {
        MpPeer *p = &s->peers[i];
        if (!p->active) continue;
        if (p->hist_n < 2) {
            memcpy(p->ipos, p->pos, sizeof p->ipos);
            memcpy(p->irot, p->rot, sizeof p->irot);
            p->iyaw = p->yaw;
            continue;
        }
        const MpSnap *newest = &p->hist[p->hist_head];
        if (target >= (double)newest->t_ns) {   // newer than the window: the newest snapshot holds
            memcpy(p->ipos, newest->pos, sizeof p->ipos);
            memcpy(p->irot, newest->rot, sizeof p->irot);
            p->iyaw = newest->yaw;
            continue;
        }
        const MpSnap *a = NULL, *b = NULL;
        for (int k = 1; k < p->hist_n && !b; k++) {
            const MpSnap *o = &p->hist[(p->hist_head + MP_HIST - k) % MP_HIST];
            const MpSnap *n = &p->hist[(p->hist_head + MP_HIST - k + 1) % MP_HIST];
            if ((double)o->t_ns <= target && (double)n->t_ns >= target) a = o, b = n;
        }
        if (!b) {   // older than everything buffered: the oldest snapshot
            const MpSnap *old = &p->hist[(p->hist_head + MP_HIST - (p->hist_n - 1)) % MP_HIST];
            memcpy(p->ipos, old->pos, sizeof p->ipos);
            memcpy(p->irot, old->rot, sizeof p->irot);
            p->iyaw = old->yaw;
            continue;
        }
        double d = (double)(b->t_ns - a->t_ns);
        float f = d > 1 ? (float)(((double)target - (double)a->t_ns) / d) : 1.0f;
        if (!(f >= 0)) f = 0;
        if (f > 1) f = 1;
        snap_lerp(a, b, f, p->ipos, p->irot, &p->iyaw);
    }
    for (int i = 0; i < MP_MAX_BEACONS; i++) {
        MpBeaconView *b = &s->beacons[i];
        if (!b->active) continue;
        b->age_s += 1.0f / MP_STATE_HZ;
        if (b->age_s >= b->duration_s) b->active = 0;
    }
}

// ---- the local player, from guest memory
//
// The walk of frame_rate.c player_pos: [main + slot] + 0x1e8 scene, its component 13 lists the resident actors, and the
// one named "Player" holds Link's 3x3 rotation at +0x290 and position at +0x2b4.

static int guest_read(uint64_t a, void *out, size_t n) { return a && plat_mem_read(a, out, n); }

static uint64_t guest_ptr(uint64_t a) {
    uint64_t p = 0;
    return guest_read(a, &p, 8) ? p : 0;
}

static uint64_t player_actor(void) {
    uint64_t slot = MOD_MAIN_BASE + (RT_V143 ? 0x3add238 : RT_V110 ? 0x4630a68 : 0x4728538),
             module = guest_ptr(slot), scene = module ? guest_ptr(module + 0x1e8) : 0;
    int32_t count = 0;
    if (!scene || !guest_read(scene + 0x50, &count, 4) || count < 14 || count > 256) return 0;
    uint64_t comps = guest_ptr(scene + 0x58), resident = comps ? guest_ptr(comps + 13 * 8) : 0;
    if (!resident || !guest_read(resident + 0x20, &count, 4) || count < 1 || count > 64) return 0;
    uint64_t links = guest_ptr(resident + 0x28);
    for (int i = 0; links && i < count; i++) {
        uint64_t data = guest_ptr(links + (uint64_t)i * 0x70 + 8), actor = data ? guest_ptr(data + 0x40) : 0;
        uint64_t name = actor ? guest_ptr(actor + 0x218) : 0;
        char text[7];
        if (name && guest_read(name, text, 7) && !memcmp(text, "Player", 7)) return actor;
    }
    return 0;
}

// Reads Link's place and rotation into L, derives the velocity from the distance and time since the last read, and the
// action flags from it. 1 when the player was found.
static int local_capture(void) {
    uint64_t actor = player_actor();
    float pos[3], rot[9];
    int ok = actor && guest_read(actor + 0x2b4, pos, sizeof pos) && guest_read(actor + 0x290, rot, sizeof rot) &&
             isfinite(pos[0]) && isfinite(pos[1]) && isfinite(pos[2]) && mat3_rotation_like(rot);
    uint64_t t = now_ns();
    plat_mutex_lock(&g_lock);
    L.actor = actor;
    L.actor_ok = actor != 0;
    if (ok) {
        memcpy(L.pos, pos, sizeof L.pos);
        memcpy(L.rot, rot, sizeof L.rot);
        L.yaw = mat3_yaw(rot);
        if (g_prev_ok) {
            double dt = (double)(t - g_prev_pos_ns) / 1e9;
            if (dt > 1e-4 && dt < 1.0) {
                for (int i = 0; i < 3; i++) L.vel[i] = (float)((pos[i] - g_prev_pos[i]) / dt);
            }
        }
        memcpy(g_prev_pos, pos, sizeof g_prev_pos);
        g_prev_pos_ns = t, g_prev_ok = 1;
        float h = sqrtf(L.vel[0] * L.vel[0] + L.vel[2] * L.vel[2]);
        L.action_flags = h > 0.5f ? MP_ACTION_MOVING : MP_ACTION_GROUNDED;
        L.valid = 1;
        __atomic_store_n(&g_lpos_valid, 1, __ATOMIC_RELEASE);
        for (int i = 0; i < 3; i++) g_lpos[i] = pos[i];
    } else {
        L.valid = 0;
        g_prev_ok = 0;
        __atomic_store_n(&g_lpos_valid, 0, __ATOMIC_RELEASE);
    }
    int v = L.valid;
    plat_mutex_unlock(&g_lock);
    return v;
}

// ---- set-up and teardown

static void session_close(MpSession *s) {
    if (s->status == MP_STATUS_HOSTING || s->status == MP_STATUS_CONNECTED) send_bye(s);
    if (s->sock) plat_udp_close(s->sock), s->sock = NULL;
    memset(s->peers, 0, sizeof s->peers);
    memset(s->beacons, 0, sizeof s->beacons);
    s->lan_n = 0;
    s->local.active = 1;
    s->local.slot = 0;
    s->local.id = s->my_id;
    s->local.color = s->my_color;
    snprintf(s->local.name, sizeof s->local.name, "%s", s->my_name);
    s->status = MP_STATUS_OFF;
    publish_peers(s);
    __atomic_store_n(&g_active, 0, __ATOMIC_RELEASE);
}

static void session_stop(MpSession *s) {
    if (s->sock || s->status != MP_STATUS_OFF) session_close(s);
    s->mode = MP_MODE_OFF;
}

static int session_start(MpSession *s) {
    session_close(s);
    s->my_id = rand_id();
    s->local.active = 1;
    s->local.slot = 0;
    s->local.id = s->my_id;
    s->local.color = s->my_color;
    snprintf(s->local.name, sizeof s->local.name, "%s", s->my_name);
    s->local.seq = 0;
    s->error[0] = 0;
    s->start_ns = now_ns();
    s->last_hello_ns = s->last_ping_ns = s->last_announce_ns = 0;
    s->last_state_ns = s->last_skel_ns = 0;
    s->my_slot = 0;
    if (s->mode == MP_MODE_HOST) {
        s->sock = plat_udp_open(s->port, 1);
        if (!s->sock) {
            snprintf(s->error, sizeof s->error, "cannot open UDP port %u", (unsigned)s->port);
            rt_log("mp: %s", s->error);
            s->status = MP_STATUS_ERROR;
            return 0;
        }
        s->port = plat_udp_bound_port(s->sock);
        s->status = MP_STATUS_HOSTING;
        rt_log("mp: hosting on UDP port %u as '%s'", (unsigned)s->port, s->my_name);
        send_announce(s, (PlatNetAddr){0xffffffffu, s->port});
    } else {
        PlatNetAddr dst;
        s->sock = plat_udp_open(0, 1);
        if (!s->sock) {
            snprintf(s->error, sizeof s->error, "cannot open a UDP socket");
            rt_log("mp: %s", s->error);
            s->status = MP_STATUS_ERROR;
            return 0;
        }
        if (!plat_udp_resolve(s->host_text, s->port, &dst)) {
            snprintf(s->error, sizeof s->error, "cannot resolve '%s'", s->host_text);
            rt_log("mp: %s", s->error);
            plat_udp_close(s->sock), s->sock = NULL;
            s->status = MP_STATUS_ERROR;
            return 0;
        }
        s->server = dst;
        s->status = MP_STATUS_CONNECTING;
        rt_log("mp: joining %s on UDP port %u as '%s'", s->host_text, (unsigned)dst.port, s->my_name);
        send_hello(s);
        s->last_hello_ns = now_ns();
    }
    publish_peers(s);
    __atomic_store_n(&g_active, 1, __ATOMIC_RELEASE);
    return 1;
}

// ---- the session thread

static int saw_cfg;
static unsigned cfg_seen = ~0u;

static void pump_socket(MpSession *s, uint64_t t) {
    if (!s->sock) return;
    uint8_t buf[2048];
    for (int guard = 0; guard < 256; guard++) {
        PlatNetAddr from = {0, 0};
        int n = plat_udp_recv(s->sock, buf, sizeof buf, &from, 0);
        if (n <= 0) break;
        handle_packet(s, buf, n, from, t);
    }
    if (s->status == MP_STATUS_HOSTING) {   // peers that went quiet
        for (int i = 0; i < MP_MAX_PLAYERS; i++) {
            MpPeer *p = &s->peers[i];
            if (p->active && t - p->last_rx_ns > (uint64_t)MP_PEER_TIMEOUT_MS * 1000000ull) {
                rt_log("mp: '%s' timed out", p->name);
                memset(p, 0, sizeof *p);
                publish_peers(s);
            }
        }
    } else if (s->status == MP_STATUS_CONNECTED) {
        int heard = 0;
        for (int i = 0; i < MP_MAX_PLAYERS; i++)
            if (s->peers[i].active && t - s->peers[i].last_rx_ns < (uint64_t)MP_PEER_TIMEOUT_MS * 1000000ull) heard = 1;
        if (!heard && t - s->start_ns > (uint64_t)(MP_PEER_TIMEOUT_MS + 1000) * 1000000ull) {
            rt_log("mp: the host stopped answering");
            snprintf(s->error, sizeof s->error, "connection lost");
            s->status = MP_STATUS_ERROR;
        }
    }
}

static void session_tick(MpSession *s) {
    uint64_t t = now_ns();
    local_capture();
    pump_socket(s, t);
    if (s->status == MP_STATUS_OFF || s->status == MP_STATUS_ERROR) return;
    if (t - s->last_state_ns >= 1000000000ull / MP_STATE_HZ) {
        s->last_state_ns = t;
        send_state(s);
    }
    if (t - s->last_skel_ns >= 1000000000ull / MP_SKEL_HZ) {
        s->last_skel_ns = t;
        send_skel(s);
    }
    if (s->status == MP_STATUS_CONNECTING) {
        if (t - s->last_hello_ns >= (uint64_t)MP_HELLO_MS * 1000000ull) {
            s->last_hello_ns = t;
            send_hello(s);
        }
        if (t - s->start_ns > (uint64_t)MP_JOIN_TIMEOUT_MS * 1000000ull) {
            rt_log("mp: no answer from %s in %u ms", s->host_text, (unsigned)MP_JOIN_TIMEOUT_MS);
            snprintf(s->error, sizeof s->error, "no answer from the host");
            s->status = MP_STATUS_ERROR;
        }
    }
    if (t - s->last_ping_ns >= 1000000000ull) {   // latency
        s->last_ping_ns = t;
        if (s->status == MP_STATUS_HOSTING) {
            for (int i = 0; i < MP_MAX_PLAYERS; i++)
                if (s->peers[i].active) send_ping(s, s->peers[i].addr, next_ping(&s->peers[i]));
        } else if (s->status == MP_STATUS_CONNECTED) {
            send_ping(s, s->server, next_ping(&s->local));
        }
    }
    if (s->status == MP_STATUS_HOSTING && t - s->last_announce_ns >= 1000000000ull) {   // answers LAN scans
        s->last_announce_ns = t;
        send_announce(s, (PlatNetAddr){0xffffffffu, s->port});
    }
    interp_tick(s, t);
}

static void mp_thread_main(void *arg) {
    (void)arg;
    MpSession *s = &G;
    while (!__atomic_load_n(&g_quit, __ATOMIC_ACQUIRE)) {
        unsigned v = cfg_version();
        if (!saw_cfg || v != cfg_seen) {
            cfg_seen = v, saw_cfg = 1;
            MpMode want = mode_of(cfg_value("NX2NAVI_MP_MODE"));
            plat_mutex_lock(&g_lock);
            mp_apply_cfg(s);
            plat_mutex_unlock(&g_lock);
            if (want != s->mode) {
                plat_mutex_lock(&g_lock);
                if (want == MP_MODE_OFF) {
                    if (s->status != MP_STATUS_OFF) {
                        rt_log("mp: session closed (mode off)");
                        session_stop(s);
                    }
                    s->mode = MP_MODE_OFF;
                } else {
                    s->mode = want;
                    session_start(s);
                }
                plat_mutex_unlock(&g_lock);
            }
        }
        if (s->mode == MP_MODE_OFF) {
            plat_sleep_ms(50);
            continue;
        }
        plat_mutex_lock(&g_lock);
        session_tick(s);
        plat_mutex_unlock(&g_lock);
        s->tick++;
        plat_sleep_ms(MP_TICK_MS);
    }
    plat_mutex_lock(&g_lock);
    if (s->status != MP_STATUS_OFF) session_stop(s);
    plat_mutex_unlock(&g_lock);
}

// ---- public: lifecycle and state

void mp_init(void) {
    static int once;
    if (once) return;
    once = 1;
    memset(&G, 0, sizeof G);
    memset(&L, 0, sizeof L);
    memset(&C, 0, sizeof C);
    G.verbose = rt_knob("MP_VERBOSE", "0") != 0;
    G.mode = MP_MODE_OFF;
    G.status = MP_STATUS_OFF;
    mp_apply_cfg(&G);
    G.my_id = rand_id();
    G.local.active = 1;
    G.local.color = G.my_color;
    snprintf(G.local.name, sizeof G.local.name, "%s", G.my_name);
    mat3_identity(G.local.irot);
    mat3_identity(G.local.rot);
    plat_thread_spawn(mp_thread_main, NULL, "nx2navi-mp", 512u << 10);
    rt_log("mp: multiplayer ready (mode %s)", cfg_value("NX2NAVI_MP_MODE"));
}

int mp_active(void) { return __atomic_load_n(&g_active, __ATOMIC_ACQUIRE); }

int mp_has_remote_players(void) {
    return mp_active() && __atomic_load_n(&g_remote, __ATOMIC_RELAXED) > 0 && cfg_bool("NX2NAVI_MP_RENDER_PLAYERS");
}

int mp_overlay_wanted(void) { return __atomic_load_n(&g_overlay, __ATOMIC_RELAXED) && mp_active(); }

void mp_retry(void) {
    plat_mutex_lock(&g_lock);
    if (G.mode != MP_MODE_OFF && G.status == MP_STATUS_ERROR) session_start(&G);
    plat_mutex_unlock(&g_lock);
}

void mp_frame_tick(void) {
    // The session thread interpolates and ages the beacons; the present only notes that a frame happened (a hook for
    // per-frame work the UI may want, kept cheap: two atomic loads).
    (void)mp_active();
}

void mp_get_session_view(MpSessionView *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    plat_mutex_lock(&g_lock);
    out->mode = G.mode;
    out->status = G.status;
    out->port = G.port;
    snprintf(out->host_addr, sizeof out->host_addr, "%s", G.host_text);
    snprintf(out->local_name, sizeof out->local_name, "%s", G.my_name);
    snprintf(out->error, sizeof out->error, "%s", G.error);
    out->local_color = G.my_color;
    out->local_slot = (uint8_t)(G.my_slot >= 0 && G.my_slot < MP_MAX_PLAYERS ? G.my_slot : 0);
    out->rtt_ms = G.local.rtt_ms;
    out->tx_packets = G.tx_pkts;
    out->rx_packets = G.rx_pkts;
    out->tx_bytes = G.tx_bytes;
    out->rx_bytes = G.rx_bytes;
    out->re_emitted_draws = __atomic_load_n(&G.re_emitted, __ATOMIC_RELAXED);
    out->local_pos_valid = L.valid;
    memcpy(out->local_pos, L.pos, sizeof out->local_pos);
    out->local_yaw = L.yaw;
    int n = 0;
    MpPlayerView *v = &out->players[n++];   // the local player first
    v->active = 1;
    v->is_local = 1;
    v->slot = out->local_slot;
    v->color_id = G.my_color;
    v->player_id = G.my_id;
    snprintf(v->name, sizeof v->name, "%s", G.my_name);
    memcpy(v->pos, L.pos, sizeof v->pos);
    memcpy(v->rot, L.rot, sizeof v->rot);
    v->yaw = L.yaw;
    v->action_flags = L.action_flags;
    v->skel_bones = L.skel_bones;
    v->ping_ms = G.local.rtt_ms;
    for (int i = 0; i < MP_MAX_PLAYERS && n < MP_MAX_PLAYERS; i++) {
        const MpPeer *p = &G.peers[i];
        if (!p->active) continue;
        v = &out->players[n++];
        v->active = 1;
        v->slot = p->slot;
        v->color_id = p->color;
        v->player_id = p->id;
        snprintf(v->name, sizeof v->name, "%s", p->name);
        memcpy(v->pos, p->ipos, sizeof v->pos);
        memcpy(v->vel, p->vel, sizeof v->vel);
        memcpy(v->rot, p->irot, sizeof v->rot);
        v->yaw = p->iyaw;
        v->action_flags = p->action_flags;
        v->ping_ms = p->rtt_ms;
        v->skel_bones = p->skel_bones;
        v->skel_model_space = p->skel_model_space;
        memcpy(v->skel_ref_pos, p->skel_ref_pos, sizeof v->skel_ref_pos);
        memcpy(v->skel_ref_rot, p->skel_ref_rot, sizeof v->skel_ref_rot);
        if (p->skel_bones) memcpy(v->skel, p->skel, sizeof(float) * 12 * p->skel_bones);
        float dx = v->pos[0] - L.pos[0], dy = v->pos[1] - L.pos[1], dz = v->pos[2] - L.pos[2];
        v->distance_m = sqrtf(dx * dx + dy * dy + dz * dz);
    }
    out->player_count = n;
    out->remote_count = n - 1;
    for (int i = 0; i < MP_MAX_BEACONS; i++)
        if (G.beacons[i].active) out->beacons[out->beacon_count++] = G.beacons[i];
    out->lan_count = G.lan_n;
    memcpy(out->lan_sessions, G.lan, sizeof out->lan_sessions);
    plat_mutex_unlock(&g_lock);
}

void mp_color_rgba(uint8_t id, float rgba[4]) {
    static const float k[MP_COLOR_COUNT][3] = {
        {0.35f, 0.85f, 0.40f}, {0.36f, 0.62f, 0.98f}, {0.95f, 0.36f, 0.32f},
        {0.72f, 0.45f, 0.95f}, {0.96f, 0.80f, 0.30f}, {0.35f, 0.90f, 0.92f},
    };
    int i = id < MP_COLOR_COUNT ? id : 0;
    rgba[0] = k[i][0], rgba[1] = k[i][1], rgba[2] = k[i][2], rgba[3] = 1.0f;
}

// ---- public: actions

int mp_teleport_to_pos(const float xyz[3]) {
    uint64_t actor;
    plat_mutex_lock(&g_lock);
    actor = L.actor;
    plat_mutex_unlock(&g_lock);
    if (!actor || !isfinite(xyz[0]) || !isfinite(xyz[1]) || !isfinite(xyz[2])) return 0;
    if (!plat_mem_readable(actor + 0x2b4, 12)) return 0;
    memcpy((void *)(uintptr_t)(actor + 0x2b4), xyz, 12);
    rt_log("mp: teleported Link to %.2f, %.2f, %.2f", (double)xyz[0], (double)xyz[1], (double)xyz[2]);
    g_prev_ok = 0;
    return 1;
}

int mp_teleport_to_slot(int slot) {
    float target[3];
    int ok = 0;
    plat_mutex_lock(&g_lock);
    for (int i = 0; i < MP_MAX_PLAYERS; i++) {
        const MpPeer *p = &G.peers[i];
        if (!p->active || p->slot != slot) continue;
        memcpy(target, p->ipos, sizeof target);
        ok = 1;
        break;
    }
    plat_mutex_unlock(&g_lock);
    return ok && mp_teleport_to_pos(target);
}

int mp_send_beacon(void) {
    float pos[3];
    int ok;
    plat_mutex_lock(&g_lock);
    memcpy(pos, L.pos, sizeof pos);
    ok = L.valid && (G.status == MP_STATUS_HOSTING || G.status == MP_STATUS_CONNECTED);
    plat_mutex_unlock(&g_lock);
    if (!ok) return 0;
    plat_mutex_lock(&g_lock);
    for (int i = 0; i < MP_MAX_BEACONS; i++) {   // the sender sees its own beacon at once
        MpBeaconView *b = &G.beacons[i];
        if (b->active) continue;
        memset(b, 0, sizeof *b);
        b->active = 1;
        b->color_id = G.my_color;
        snprintf(b->sender, sizeof b->sender, "%s", G.my_name);
        memcpy(b->pos, pos, sizeof b->pos);
        b->duration_s = MP_BEACON_S;
        break;
    }
    send_beacon_pkt(&G, pos);
    plat_mutex_unlock(&g_lock);
    return 1;
}

void mp_request_lan_scan(void) {
    plat_mutex_lock(&g_lock);
    G.lan_n = 0;
    if (G.sock) {
        uint8_t buf[16];
        MpWr w;
        wr_init(&w, buf, sizeof buf);
        wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_DISCOVER), wr_u16(&w, MP_PROTOCOL);
        if (!w.bad) {
            PlatNetAddr bcast = {0xffffffffu, G.port};
            send_pkt(&G, bcast, buf, w.n);
            PlatNetAddr local = {0x7f000001u, G.port};
            send_pkt(&G, local, buf, w.n);
        }
    }
    plat_mutex_unlock(&g_lock);
}

// ---- the camera (nametags and off-screen arrows)

static void mat4_mul(const float a[16], const float b[16], float out[16]) {
    float t[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            t[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
    memcpy(out, t, sizeof t);
}

void mp_observe_camera_ubo(int slot, const void *data, uint32_t size) {
    (void)slot;
    if (!data) return;
    const float *f = data;
    float vp[16], view[16] = {0};
    if (size >= 0x940) {          // 1.4.3's deferred pass: the view at 0, the view-projection at 0x30
        for (int r = 0; r < 3; r++) memcpy(view + r * 4, f + r * 4, 4 * sizeof(float));
        memcpy(vp, f + 0x30 / 4, sizeof vp);
    } else if (size >= 0x3500) {  // Ocarina: the view at 0, the projection at 0x70
        for (int r = 0; r < 3; r++) memcpy(view + r * 4, f + r * 4, 4 * sizeof(float));
        float proj[16];
        memcpy(proj, f + 0x70 / 4, sizeof proj);
        if (!isfinite(proj[0]) || !isfinite(proj[5])) return;
        mat4_mul(proj, view, vp);
    } else {
        return;
    }
    if (!isfinite(vp[0]) || !isfinite(vp[5]) || !isfinite(vp[15])) return;
    view[15] = 1;
    plat_mutex_lock(&g_lock);
    memcpy(C.vp, vp, sizeof C.vp);
    memcpy(C.view, view, sizeof C.view);
    C.valid = 1;
    plat_mutex_unlock(&g_lock);
}

int mp_project_world_to_screen(const float world[3], float *u_out, float *v_out, float *clip_w_out) {
    float vp[16];
    plat_mutex_lock(&g_lock);
    memcpy(vp, C.vp, sizeof vp);
    int ok = C.valid;
    plat_mutex_unlock(&g_lock);
    if (!ok) return 0;
    float x = world[0], y = world[1], z = world[2];
    float cx = vp[0] * x + vp[1] * y + vp[2] * z + vp[3];
    float cy = vp[4] * x + vp[5] * y + vp[6] * z + vp[7];
    float cw = vp[12] * x + vp[13] * y + vp[14] * z + vp[15];
    if (!isfinite(cw) || fabsf(cw) < 1e-6f) return 0;
    float nx = cx / cw, ny = cy / cw;
    if (!isfinite(nx) || !isfinite(ny)) return 0;
    // Vulkan clip space has y pointing down, as the swapchain image does: no flip here
    if (u_out) *u_out = nx * 0.5f + 0.5f;
    if (v_out) *v_out = ny * 0.5f + 0.5f;
    if (clip_w_out) *clip_w_out = cw;
    return 1;
}

// ---- the renderer: recognising local Link's draws, and a remote player's blocks

// A uniform block as of the submit: the uniform snapshot's host copy when there is one, else guest memory.
static int read_block(uint64_t addr, uint32_t size, void *out, uint32_t want) {
    if (!addr || !out || !want || want > size) return 0;
    const uint8_t *p = vk_usnap_bytes(addr, want);
    if (p) return memcpy(out, p, want), 1;
    return plat_mem_read(addr, out, want);
}

// A c7 shape block's world translation, when the block is a rotation-like 3x4 matrix.
static int shape_translation(const void *data, uint32_t size, float t[3]) {
    if (!data || size < 48) return 0;
    const float *f = data;
    float m[9];
    for (int r = 0; r < 3; r++) memcpy(m + r * 3, f + r * 4, 3 * sizeof(float));
    if (!mat3_rotation_like(m)) return 0;
    t[0] = f[3], t[1] = f[7], t[2] = f[11];
    return isfinite(t[0]) && isfinite(t[1]) && isfinite(t[2]);
}

// The bones of a c5 block, and whether their translations are model-space offsets (the model sits at the origin) rather
// than world positions.
static uint32_t read_bones(const void *data, uint32_t size, uint32_t cap, float *out, int *model_space,
                           const float link_pos[3]) {
    if (!data || size < 48) return 0;
    uint32_t n = size / 48, good = 0;
    if (n > cap) n = cap;
    const float *f = data;
    for (uint32_t i = 0; i < n; i++) {
        const float *b = f + i * 12;
        float m[9];
        for (int r = 0; r < 3; r++) memcpy(m + r * 3, b + r * 4, 3 * sizeof(float));
        if (!mat3_rotation_like(m) || !isfinite(b[3]) || !isfinite(b[7]) || !isfinite(b[11])) break;
        memcpy(out + i * 12, b, 12 * sizeof(float));
        good++;
    }
    if (!good) return 0;
    float t[3] = {out[3], out[7], out[11]};
    float d0 = sqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    float dx = t[0] - link_pos[0], dy = t[1] - link_pos[1], dz = t[2] - link_pos[2];
    float dl = sqrtf(dx * dx + dy * dy + dz * dz);
    *model_space = d0 < MP_LINK_RADIUS && dl > MP_LINK_RADIUS;
    return good;
}

static volatile int32_t g_slot_pick[MP_MAX_PLAYERS];

int mp_vk_remote_slot(int i) { return i >= 0 && i < MP_MAX_PLAYERS ? g_slot_pick[i] : -1; }

int mp_vk_link_draw(uint64_t c5_addr, uint32_t c5_size, uint64_t c7_addr, uint32_t c7_size, int *c5_model_space) {
    if (c5_model_space) *c5_model_space = 0;
    if (!mp_active() || !__atomic_load_n(&g_lpos_valid, __ATOMIC_ACQUIRE)) return 0;
    float link_pos[3] = {g_lpos[0], g_lpos[1], g_lpos[2]};
    uint8_t shape[64];
    float st[3];
    if (!read_block(c7_addr, c7_size, shape, 48) || !shape_translation(shape, 48, st)) return 0;
    float dx = st[0] - link_pos[0], dy = st[1] - link_pos[1], dz = st[2] - link_pos[2];
    if (dx * dx + dy * dy + dz * dz > MP_LINK_RADIUS * MP_LINK_RADIUS) return 0;   // another object's part
    static uint8_t bones[48 * MP_MAX_BONES];   // the replay thread only
    uint32_t want = c5_size < sizeof bones ? c5_size : (uint32_t)sizeof bones;
    int model = 0;
    float local_bones[MP_MAX_BONES * 12];
    uint32_t nb = 0;
    if (want >= 48 && read_block(c5_addr, c5_size, bones, want))
        nb = read_bones(bones, want, MP_MAX_BONES, local_bones, &model, link_pos);
    plat_mutex_lock(&g_lock);
    if (nb) {   // this is what the session replicates
        L.skel_bones = nb;
        L.skel_model_space = model;
        memcpy(L.skel_ref_pos, link_pos, sizeof L.skel_ref_pos);
        memcpy(L.skel_ref_rot, L.rot, sizeof L.skel_ref_rot);
        memcpy(L.skel, local_bones, sizeof(float) * 12 * nb);
    }
    plat_mutex_unlock(&g_lock);
    if (c5_model_space) *c5_model_space = model;
    if (!cfg_bool("NX2NAVI_MP_RENDER_PLAYERS")) return 0;
    uint32_t mask = __atomic_load_n(&g_peer_mask, __ATOMIC_ACQUIRE);
    int n = 0;
    for (int i = 0; i < MP_MAX_PLAYERS; i++)
        if (mask & (1u << i)) g_slot_pick[n++] = i;
    return n;
}

// The blocks a re-emitted draw binds.
void mp_vk_build_remote_ubos(int slot, const void *local_c5, uint32_t c5_size, void *c5_out, const void *local_c7,
                             uint32_t c7_size, void *c7_out, int c5_model_space) {
    float pos[3], rot[9], rp[3], rr[9];
    int have_skel = 0;
    plat_mutex_lock(&g_lock);
    if (slot >= 0 && slot < MP_MAX_PLAYERS && G.peers[slot].active) {
        const MpPeer *p = &G.peers[slot];
        memcpy(pos, p->ipos, sizeof pos);
        memcpy(rot, p->irot, sizeof rot);
        have_skel = p->skel_bones > 0;
        memcpy(rp, p->skel_ref_pos, sizeof rp);
        memcpy(rr, p->skel_ref_rot, sizeof rr);
    }
    plat_mutex_unlock(&g_lock);
    if (!have_skel && (!isfinite(pos[0]) || !isfinite(pos[2]))) return;
    float ident[9], zero[3] = {0, 0, 0};
    mat3_identity(ident);
    if (c5_out && local_c5 && c5_size >= 48) {
        uint32_t n = c5_size / 48;
        const float *in = local_c5;
        float *out = c5_out;
        for (uint32_t i = 0; i < n; i++) {
            const float *src = in + i * 12;
            if (c5_model_space) mp_place_model_space(src, pos, rot, out + i * 12);
            else if (have_skel) mp_transform_mat3x4(src, rp, rr, pos, rot, out + i * 12);
            else mp_transform_mat3x4(src, zero, ident, pos, rot, out + i * 12);
        }
    }
    if (c7_out && local_c7 && c7_size >= 48) {
        const float *in = local_c7;
        if (have_skel) mp_transform_mat3x4(in, rp, rr, pos, rot, c7_out);
        else mp_transform_mat3x4(in, zero, ident, pos, rot, c7_out);
    }
}

void mp_vk_note_re_emitted(uint32_t n) {
    if (n) __atomic_add_fetch(&G.re_emitted, n, __ATOMIC_RELAXED);
}

// ---- the dedicated server (--mp-server)

int mp_run_dedicated_server(uint16_t port) {
    static MpSession s;
    memset(&s, 0, sizeof s);
    mp_apply_cfg(&s);
    if (port) s.port = port;
    s.mode = MP_MODE_HOST;
    s.verbose = 1;
    s.my_id = rand_id();
    snprintf(s.my_name, sizeof s.my_name, "%s", "Dedicated");
    s.local.active = 1, s.local.id = s.my_id, s.local.color = s.my_color;
    snprintf(s.local.name, sizeof s.local.name, "%s", s.my_name);
    if (!session_start(&s)) {
        fprintf(stderr, "mp-server: %s\n", s.error);
        return 1;
    }
    printf("mp-server: hosting on UDP port %u, %d players at most (Ctrl-C to stop)\n", (unsigned)s.port, MP_MAX_PLAYERS);
    fflush(stdout);
    rt_log("mp-server: hosting on UDP port %u", (unsigned)s.port);
    int last = -1;
    while (!__atomic_load_n(&g_quit, __ATOMIC_ACQUIRE)) {
        session_tick(&s);
        int n = peer_count(&s);
        if (n != last) {
            last = n;
            printf("mp-server: %d/%d players\n", n, MP_MAX_PLAYERS);
            fflush(stdout);
            rt_log("mp-server: %d/%d players", n, MP_MAX_PLAYERS);
        }
        plat_sleep_ms(MP_TICK_MS);
    }
    session_stop(&s);
    return 0;
}

// ---- the probe (--probe-mp)

static int probe_fail;

static void chk(int ok, const char *what) {
    printf("probe-mp: %-66s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) probe_fail++;
}

static int pump_once(MpSession *s, uint32_t wait_ms) {
    uint8_t buf[2048];
    PlatNetAddr from = {0, 0};
    int n = plat_udp_recv(s->sock, buf, sizeof buf, &from, wait_ms);
    if (n <= 0) return 0;
    handle_packet(s, buf, n, from, now_ns());
    return 1;
}

static void pump_pair(MpSession *a, MpSession *b, uint32_t ms) {
    uint64_t end = now_ns() + (uint64_t)ms * 1000000ull;
    while (now_ns() < end) {
        if (a->sock) pump_once(a, 2);
        if (b->sock) pump_once(b, 2);
    }
}

int probe_mp(void) {
    probe_fail = 0;
    // 1. the byte codecs
    {
        uint8_t buf[256];
        MpWr w;
        MpRd r;
        wr_init(&w, buf, sizeof buf);
        wr_u8(&w, 0x12), wr_u16(&w, 0x3456), wr_u32(&w, 0x789abcdeu), wr_f32(&w, -1.5f), wr_str(&w, "A Link");
        rd_init(&r, buf, w.n);
        uint32_t a = rd_u8(&r), b = rd_u16(&r), c = rd_u32(&r);
        float f = rd_f32(&r);
        char name[MP_NAME_MAX];
        rd_str(&r, name, sizeof name);
        chk(!w.bad && !r.bad && a == 0x12 && b == 0x3456 && c == 0x789abcdeu && f == -1.5f && !strcmp(name, "A Link"),
            "the little-endian codecs round trip");
        rd_init(&r, buf, 4);
        while (!r.bad) rd_u32(&r);
        chk(r.bad, "a short packet is rejected");
        wr_init(&w, buf, 4);
        wr_u32(&w, 1), wr_u32(&w, 2);
        chk(w.bad, "a write past the buffer is noticed");
    }
    // 2. the transform math and the bone quantization
    {
        float in[12] = {1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7}, out[12], ident[9], zero[3] = {0, 0, 0};
        mat3_identity(ident);
        mp_transform_mat3x4(in, zero, ident, zero, ident, out);
        int same = 1;
        for (int i = 0; i < 12; i++) same &= fabsf(out[i] - in[i]) < 1e-6f;
        chk(same, "the transform is the identity for one frame");
        float from[3] = {5, 6, 7}, to[3] = {105, 6, 7};
        mp_transform_mat3x4(in, from, ident, to, ident, out);
        chk(fabsf(out[3] - 105) < 1e-4f && fabsf(out[7] - 6) < 1e-4f && fabsf(out[11] - 7) < 1e-4f,
            "a bone at the capture position lands on the remote one");
        float a90[9], facez[9], out2[12], in2[12] = {0};
        mat3_from_yaw((float)M_PI / 2, a90);   // the local player faces +x
        mat3_from_yaw(0, facez);               // the remote one faces +z
        for (int r = 0; r < 3; r++) memcpy(in2 + r * 4, a90 + r * 3, 3 * sizeof(float));
        // the local player's own orientation, placed in the remote player's frame, is the remote player's: the rotation
        // part is remote_rot * local_rot^T applied to local_rot
        mp_transform_mat3x4(in2, zero, a90, zero, facez, out2);
        same = 1;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) same &= fabsf(out2[r * 4 + c] - facez[r * 3 + c]) < 1e-4f;
        chk(same, "the rotation part turns the local frame onto the remote one");
        float q[12];
        int16_t packed[12];
        bone_pack(in2, zero, packed);
        bone_unpack(packed, zero, q);
        same = 1;
        for (int i = 0; i < 12; i++) same &= fabsf(q[i] - in2[i]) < 1e-3f;
        chk(same, "the int16 bone quantization round trips");
        float half[9], om[9];
        for (int i = 0; i < 9; i++) half[i] = 0.5f * (a90[i] + facez[i]);
        mat3_orthonormalize(half, om);
        chk(mat3_rotation_like(om), "a lerped rotation is orthonormalized");
    }
    // 3. a loopback session: hello/welcome, state, skeleton chunks, a beacon, the ping and LAN discovery
    {
        static MpSession h, c;
        memset(&h, 0, sizeof h);
        memset(&c, 0, sizeof c);
        uint16_t base = (uint16_t)(30000 + (now_ns() / 1000000ull) % 20000);
        PlatUdp *hs = NULL;
        for (int tries = 0; tries < 24 && !hs; tries++) hs = plat_udp_open((uint16_t)(base + tries), 1);
        chk(hs != NULL, "a host socket opens");
        if (hs) {
            h.sock = hs;
            h.port = plat_udp_bound_port(hs);
            h.mode = MP_MODE_HOST;
            h.status = MP_STATUS_HOSTING;
            h.my_id = 0x1111u;
            h.my_color = MP_COLOR_GREEN;
            snprintf(h.my_name, sizeof h.my_name, "Host");
            h.local.active = 1, h.local.id = h.my_id;
            c.sock = plat_udp_open(0, 1);
            chk(c.sock != NULL, "a client socket opens");
            c.mode = MP_MODE_JOIN;
            c.status = MP_STATUS_CONNECTING;
            c.my_id = 0x2222u;
            c.my_color = MP_COLOR_BLUE;
            snprintf(c.my_name, sizeof c.my_name, "Client");
            c.port = h.port;
            snprintf(c.host_text, sizeof c.host_text, "127.0.0.1");
            c.server = (PlatNetAddr){0x7f000001u, h.port};
            send_hello(&c);
            pump_pair(&h, &c, 80);
            chk(c.status == MP_STATUS_CONNECTED, "the client is welcomed");
            chk(c.my_slot >= 1, "the client got a slot");
            int hn = 0;
            for (int i = 0; i < MP_MAX_PLAYERS; i++) hn += h.peers[i].active;
            chk(hn == 1, "the host lists the client");
            MpPeer *cl = NULL;
            for (int i = 0; i < MP_MAX_PLAYERS; i++)
                if (h.peers[i].active) cl = &h.peers[i];
            chk(cl && !strcmp(cl->name, "Client"), "the host has the client's name");
            chk(c.peers[0].active && !strcmp(c.peers[0].name, "Host"), "the client lists the host at slot 0");
            if (cl) {
                float pos[3] = {12.5f, 3.25f, -7.0f}, rot[9], vel[3] = {1, 0, 0};
                mat3_from_yaw(0.5f, rot);
                uint8_t buf[128];
                MpWr w;
                wr_init(&w, buf, sizeof buf);
                wr_u32(&w, MP_MAGIC), wr_u8(&w, MP_PKT_STATE), wr_u32(&w, c.my_id), wr_u32(&w, 7);
                for (int i = 0; i < 3; i++) wr_f32(&w, pos[i]);
                for (int i = 0; i < 3; i++) wr_f32(&w, vel[i]);
                for (int i = 0; i < 9; i++) wr_f32(&w, rot[i]);
                wr_f32(&w, 0.5f);
                wr_u32(&w, MP_ACTION_MOVING);
                send_pkt(&c, c.server, buf, w.n);
                pump_pair(&h, &c, 60);
                chk(fabsf(cl->pos[0] - 12.5f) < 1e-4f && fabsf(cl->pos[2] + 7.0f) < 1e-4f,
                    "the host got the client's position");
                chk(cl->hist_n >= 1, "the snapshot was buffered for interpolation");
                interp_tick(&h, now_ns());
                chk(fabsf(cl->ipos[0] - 12.5f) < 1e-4f, "one snapshot interpolates to itself");
                uint32_t total = MP_BONES_PER_PKT + 4, chunks = 2;
                float bones[MP_MAX_BONES * 12];
                memset(bones, 0, sizeof bones);
                for (uint32_t i = 0; i < total; i++) mat3_identity(bones + i * 12);
                for (uint32_t ci = 0; ci < chunks; ci++) {
                    uint32_t first = ci * MP_BONES_PER_PKT;
                    uint32_t n = total - first < MP_BONES_PER_PKT ? total - first : MP_BONES_PER_PKT;
                    uint8_t sbuf[MP_MAX_PKT];
                    MpWr sw;
                    float ident[9], zero[3] = {0, 0, 0};
                    mat3_identity(ident);
                    wr_init(&sw, sbuf, sizeof sbuf);
                    wr_u32(&sw, MP_MAGIC), wr_u8(&sw, MP_PKT_SKEL), wr_u32(&sw, c.my_id), wr_u32(&sw, 99);
                    wr_u8(&sw, total), wr_u8(&sw, ci), wr_u8(&sw, n), wr_u8(&sw, 0);
                    for (int i = 0; i < 3; i++) wr_f32(&sw, 0);
                    for (int i = 0; i < 9; i++) wr_f32(&sw, ident[i]);
                    for (uint32_t k = 0; k < n; k++) {
                        int16_t q[12];
                        bone_pack(bones + (first + k) * 12, zero, q);
                        for (int i = 0; i < 12; i++) wr_u16(&sw, (uint16_t)q[i]);
                    }
                    send_pkt(&c, c.server, sbuf, sw.n);
                    pump_pair(&h, &c, 30);
                }
                chk(cl->skel_bones == total, "a skeleton split in chunks reassembles");
                float bpos[3] = {1, 2, 3};
                send_beacon_pkt(&c, bpos);
                pump_pair(&h, &c, 40);
                int nb = 0;
                for (int i = 0; i < MP_MAX_BEACONS; i++) nb += h.beacons[i].active;
                chk(nb == 1, "the host got the beacon");
                send_ping(&c, c.server, next_ping(&c.local));
                pump_pair(&h, &c, 60);
                chk(c.local.ping_sent_ns == 0 && c.local.rtt_ms < 1000, "the ping round trip is measured");
                c.lan_n = 0;
                uint8_t dbuf[16];
                MpWr dw;
                wr_init(&dw, dbuf, sizeof dbuf);
                wr_u32(&dw, MP_MAGIC), wr_u8(&dw, MP_PKT_DISCOVER), wr_u16(&dw, MP_PROTOCOL);
                send_pkt(&c, (PlatNetAddr){0x7f000001u, h.port}, dbuf, dw.n);
                pump_pair(&h, &c, 80);
                chk(c.lan_n >= 1, "LAN discovery finds the host");
                if (c.lan_n >= 1)
                    chk(c.lan[0].port == h.port && c.lan[0].player_count >= 2,
                        "the announce names the host's port and player count");
                send_bye(&c);
                pump_pair(&h, &c, 40);
                int after = 0;
                for (int i = 0; i < MP_MAX_PLAYERS; i++) after += h.peers[i].active;
                chk(after == 0, "a goodbye drops the player");
            }
            if (c.sock) plat_udp_close(c.sock);
            plat_udp_close(hs);
            h.sock = c.sock = NULL;
        }
    }
    // 4. the camera projection the nametags use
    {
        float vp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, world[3] = {0.5f, -0.5f, 2.0f};
        uint8_t block[0x940];
        memset(block, 0, sizeof block);
        memcpy(block + 0x30, vp, sizeof vp);
        mp_observe_camera_ubo(1, block, sizeof block);
        float u = -1, v = -1, w = 0;
        int ok = mp_project_world_to_screen(world, &u, &v, &w);
        chk(ok && fabsf(u - 0.75f) < 1e-4f && fabsf(v - 0.25f) < 1e-4f && w > 0,
            "the camera projects a world point to the screen");
        plat_mutex_lock(&g_lock);
        C.valid = 0;
        plat_mutex_unlock(&g_lock);
        chk(!mp_project_world_to_screen(world, &u, &v, &w), "without a camera the projection declines");
    }
    printf("probe-mp: %s (%d failures)\n", probe_fail ? "FAILED" : "passed", probe_fail);
    return probe_fail ? 1 : 0;
}
