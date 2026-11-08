// Multiplayer (runtime/multiplayer.c): UDP host/relay and client play over LAN or the internet, local Link state and
// skeleton capture, remote player interpolation, Vulkan draw re-emission, 3D nametags and beacons.
//
// The session runs on its own thread (mp_init); the menu's Multiplayer tab (ui/mp_page.cpp) and the in-world overlay
// read consistent snapshots through mp_get_session_view, and the Vulkan replay asks mp_vk_* whether a scene draw is one
// of local Link's and, if so, what to draw for every remote player.
//
// Settings (cfg.c, section [multiplayer] of nx2navi.ini): MP_MODE (off | host | join), MP_HOST, MP_PORT, MP_NAME,
// MP_COLOR, MP_NAMETAGS, MP_HUD, MP_RENDER_PLAYERS, MP_INTERP_MS. All are live: changing one in the menu re-reads it on
// the session thread.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MP_MAX_PLAYERS   = 8,
    MP_MAX_BONES     = 96,
    MP_BONES_PER_PKT = 48,
    MP_NAME_MAX      = 24,
    MP_DEFAULT_PORT  = 27015,
    MP_MAX_LAN_HOSTS = 8,
    MP_MAX_BEACONS   = 8,
};

typedef enum {
    MP_MODE_OFF  = 0,
    MP_MODE_HOST = 1,
    MP_MODE_JOIN = 2,
} MpMode;

typedef enum {
    MP_STATUS_OFF        = 0,
    MP_STATUS_HOSTING    = 1,
    MP_STATUS_CONNECTING = 2,
    MP_STATUS_CONNECTED  = 3,
    MP_STATUS_ERROR      = 4,
} MpStatus;

typedef enum {
    MP_COLOR_GREEN  = 0,
    MP_COLOR_BLUE   = 1,
    MP_COLOR_RED    = 2,
    MP_COLOR_VIOLET = 3,
    MP_COLOR_GOLD   = 4,
    MP_COLOR_CYAN   = 5,
    MP_COLOR_COUNT  = 6,
} MpColorId;

enum {
    MP_ACTION_GROUNDED = 1u << 0,
    MP_ACTION_MOVING   = 1u << 1,
    MP_ACTION_AIRBORNE = 1u << 2,
    MP_ACTION_COMBAT   = 1u << 3,
    MP_ACTION_CROUCH   = 1u << 4,
};

// One player in the session, for the menu's roster, the 3D nametags and the renderer.
typedef struct {
    int      active;
    int      is_local;
    uint8_t  slot;
    uint8_t  color_id;                 // MpColorId of the player's accent colour
    uint32_t player_id;
    char     name[MP_NAME_MAX];
    float    pos[3];                   // interpolated world position
    float    vel[3];                   // world velocity
    float    rot[9];                   // 3x3 row-major orientation
    float    yaw;                      // heading in radians
    uint32_t action_flags;             // MP_ACTION_*
    uint32_t ping_ms;                  // round-trip latency
    float    distance_m;               // distance from local Link
    uint32_t skel_bones;               // bones synchronized (0: position only)
    int      skel_model_space;
    float    skel_ref_pos[3];          // the frame the bones were captured in
    float    skel_ref_rot[9];
    float    skel[MP_MAX_BONES * 12];  // 3x4 row-major bone matrices
} MpPlayerView;

// A 3D world beacon in the session.
typedef struct {
    int      active;
    uint8_t  slot;
    uint8_t  color_id;
    char     sender[MP_NAME_MAX];
    float    pos[3];
    float    age_s;                    // seconds since it arrived
    float    duration_s;               // its lifetime
} MpBeaconView;

// A session found on the LAN, for the Multiplayer tab's browser.
typedef struct {
    char     host_ip[64];
    uint16_t port;
    char     host_name[MP_NAME_MAX];
    uint8_t  player_count;
    uint8_t  max_players;
} MpLanSession;

// A consistent snapshot of the session, players, beacons and LAN list.
typedef struct {
    MpMode       mode;
    MpStatus     status;
    uint16_t     port;
    char         host_addr[64];
    char         local_name[MP_NAME_MAX];
    uint8_t      local_color;
    uint8_t      local_slot;
    char         error[128];           // why the last attempt failed ("")
    int          player_count;
    int          remote_count;
    uint32_t     rtt_ms;
    uint64_t     tx_packets, rx_packets, tx_bytes, rx_bytes;
    uint64_t     re_emitted_draws;     // Vulkan draws drawn again for remote players
    int          local_pos_valid;
    float        local_pos[3];
    float        local_yaw;
    MpPlayerView players[MP_MAX_PLAYERS];
    MpBeaconView beacons[MP_MAX_BEACONS];
    int          beacon_count;
    int          lan_count;
    MpLanSession lan_sessions[MP_MAX_LAN_HOSTS];
} MpSessionView;

// ---- lifecycle
void mp_init(void);                                   // main.c: starts the session thread (any time)
void mp_frame_tick(void);                             // per presented frame: advances the interpolation
int  mp_active(void);                                 // fast: a session is hosting or connected (replay thread)
int  mp_has_remote_players(void);                     // ... with remote players, and 3D rendering enabled
int  mp_overlay_wanted(void);                          // nametags or the mini-HUD is on
void mp_retry(void);                                  // starts a failed session again (the menu's Retry row)

// ---- state
void mp_get_session_view(MpSessionView *out);
void mp_color_rgba(uint8_t color_id, float rgba[4]);

// ---- actions (the Multiplayer tab and the overlay's keys)
int  mp_teleport_to_slot(int slot);
int  mp_teleport_to_pos(const float xyz[3]);
int  mp_send_beacon(void);                            // at local Link's position, to every player
void mp_request_lan_scan(void);                       // clears the LAN list and broadcasts a discovery

// ---- rendering: nametags and off-screen arrows
// The scene camera as the replay sees it: 1.4.3's c4 (0x940 bytes) or Ocarina's c3 (0x3500 bytes).
void mp_observe_camera_ubo(int slot, const void *data, uint32_t size);
// Projects a world position to normalized screen coordinates (u, v in [0, 1], top-left origin); *clip_w is positive in
// front of the camera. 1 when a camera is known.
int  mp_project_world_to_screen(const float world_xyz[3], float *u, float *v, float *clip_w);

// ---- rendering: remote players in the scene
// vk_exec_draw.c, for every scene draw once its vertex shader is known and its c5/c7 blocks are bound: if the draw is one
// of local Link's parts (its c7 world translation is within a few units of Link's position), the skeleton and shape are
// captured for replication and the draw is one to repeat for every remote player. Returns the number of players to
// draw again (0: not Link), and mp_vk_remote_slot(i) names each one.
int  mp_vk_link_draw(uint64_t c5_addr, uint32_t c5_size, uint64_t c7_addr, uint32_t c7_size, int *c5_model_space);
int  mp_vk_remote_slot(int i);
// Builds the blocks a re-emitted draw binds in place of local Link's: c5 (the skeleton) and c7 (the shape), copied from
// local_c5/local_c7 into c5_out/c7_out. c5_model_space: local_c5's translations are model-space offsets (from
// mp_vk_link_draw) rather than world positions.
void mp_vk_build_remote_ubos(int slot, const void *local_c5, uint32_t c5_size, void *c5_out, const void *local_c7,
                             uint32_t c7_size, void *c7_out, int c5_model_space);
void mp_vk_note_re_emitted(uint32_t n);

// ---- modes
int probe_mp(void);                                   // --probe-mp: codecs, math, a loopback session, projection
int mp_run_dedicated_server(uint16_t port);           // --mp-server: a relay host until the process ends

// ---- math (exposed for the probe)
// Places a 3x4 row-major matrix of the local player's frame in the remote player's frame (see multiplayer.c).
void mp_transform_mat3x4(const float in_mat[12], const float local_pos[3], const float local_rot[9],
                         const float remote_pos[3], const float remote_rot[9], float out_mat[12]);

#ifdef __cplusplus
}
#endif
