// The Multiplayer tab (ui/mp_page.cpp): the session's settings as rows of the cfg table (ui/option_rows.h), the live
// roster with each player's colour, distance, latency and bones, a Teleport and a Beacon for each remote player, and the
// LAN browser listing the hosts that answer the discovery broadcast.
//
// The page takes multiplayer.c's snapshot (mp_get_session_view) once a frame and draws from it; the session thread owns
// the socket and the players, so no row ever blocks on the network. What a row writes goes to cfg_set, and the session
// thread re-reads the settings when cfg_version changes.
//
// Text (the player's name, the host address) is typed in a window of its own (mp_page_after), opened by its row: the
// rows of option_rows.cpp show text, they take none. The window has every character a row's own keys do not cover —
// letters, digits and the punctuation of a host name — placed on the keyboard, so a session can be set up without one.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "imgui.h"
#include "imgui_internal.h"   // IsPopupOpen

#include "menu_sound.h"
#include "ui/menu_skin.h"
#include "ui/mp_page.h"
#include "ui/option_rows.h"

extern "C" {
#include "cfg.h"
#include "multiplayer.h"
#include "ui_text.h"
// host.h pulls in cpu.h, which is C only: the one function needed from it, as declared there.
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
int menu_is_open(void);   // ui/menu.cpp: the overlay goes under the menu's window while it is open
}

// ---- the session's snapshot, once a frame
//
// The snapshot carries every player's bone matrices (2.3 KiB each), so the rows, the overlay and the LAN list of a frame
// all read the one taken for it.
static MpSessionView g_view;
static unsigned g_view_frame = ~0u;
static int g_view_ok;

static const MpSessionView &session_view(void) {
    unsigned f = ImGui::GetFrameCount();
    if (!g_view_ok || f != g_view_frame) {
        mp_get_session_view(&g_view);
        g_view_frame = f, g_view_ok = 1;
    }
    return g_view;
}

static const char *status_text(MpStatus s) {
    switch (s) {
    case MP_STATUS_HOSTING: return menu_text(UI_MENU_MP_STATUS_HOSTING);
    case MP_STATUS_CONNECTING: return menu_text(UI_MENU_MP_STATUS_CONNECTING);
    case MP_STATUS_CONNECTED: return menu_text(UI_MENU_MP_STATUS_CONNECTED);
    case MP_STATUS_ERROR: return menu_text(UI_MENU_MP_STATUS_ERROR);
    case MP_STATUS_OFF:
    default: return menu_text(UI_MENU_MP_STATUS_OFF);
    }
}

static ImVec4 accent(uint8_t color) {
    float rgba[4];
    mp_color_rgba(color, rgba);
    return ImVec4(rgba[0], rgba[1], rgba[2], 1.0f);
}

// A world distance (about a metre to the unit at Link's scale), as metres under a kilometre.
static void format_distance(char *out, size_t cap, float m) {
    if (!(m >= 0) || m > 1e6f) snprintf(out, cap, "-");
    else if (m < 1000.0f) snprintf(out, cap, "%.1f m", (double)m);
    else snprintf(out, cap, "%.2f km", (double)m / 1000.0);
}

// A filled disc in a player's colour on the line the cursor is at, then the cursor past it.
static void color_dot(uint8_t color) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetTextLineHeight(), r = h * 0.24f;
    dl->AddCircleFilled(ImVec2(p.x + r * 1.7f, p.y + h * 0.5f), r, ImGui::GetColorU32(accent(color)));
    ImGui::Dummy(ImVec2(r * 3.6f, h));
    ImGui::SameLine(0, 0);
}

// ---- the roster: one line per player, and for a remote one the two things a player can do to it
static void roster_row(const MpPlayerView &p) {
    ImGui::PushID((int)p.slot);
    color_dot(p.color_id);
    ImGui::TextUnformatted(p.name);
    ImGui::SameLine();
    if (p.is_local) ImGui::TextDisabled("%s", menu_text(UI_MENU_MP_YOU));
    // the slot, the distance, the round trip and the bones in one column on the right, so none of them is cut off
    char right[128], dist[32], bones[48];
    format_distance(dist, sizeof dist, p.distance_m);
    if (p.skel_bones) snprintf(bones, sizeof bones, "%u%s", (unsigned)p.skel_bones, menu_text(UI_MENU_MP_BONES));
    else snprintf(bones, sizeof bones, "%s", menu_text(UI_MENU_MP_NO_SKELETON));
    if (p.is_local) snprintf(right, sizeof right, "#%u  %s", (unsigned)p.slot, bones);
    else if (p.ping_ms)
        snprintf(right, sizeof right, "#%u  %s  %u ms  %s", (unsigned)p.slot, dist, (unsigned)p.ping_ms, bones);
    else snprintf(right, sizeof right, "#%u  %s  %s", (unsigned)p.slot, dist, bones);
    ImGui::SameLine();
    float rx = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(right).x;
    if (rx > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + rx);
    ImGui::TextDisabled("%s", right);
    if (!p.is_local) {
        float h = ImGui::GetFrameHeight(), gap = ImGui::GetStyle().ItemSpacing.x;
        float w = fmaxf((ImGui::GetContentRegionAvail().x - gap) * 0.5f, 1.0f);
        if (ImGui::Button(menu_text(UI_MENU_MP_TELEPORT), ImVec2(w, h)))
            if (mp_teleport_to_slot(p.slot)) ImGui::SetItemTooltip("%s", menu_text(UI_MENU_MP_TELEPORTED));
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
            ImGui::SetItemTooltip("%s", menu_text(UI_MENU_MP_TELEPORT_DESC));
        ImGui::SameLine(0, gap);
        if (ImGui::Button(menu_text(UI_MENU_MP_BEACON), ImVec2(w, h))) mp_send_beacon();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
            ImGui::SetItemTooltip("%s", menu_text(UI_MENU_MP_PING_DESC));
    }
    ImGui::Separator();
    ImGui::PopID();
}

// The status line over the roster: the state, then the address or the port, the players and the round trip.
static void status_line(const MpSessionView &v) {
    ImVec4 col = v.status == MP_STATUS_ERROR      ? ImVec4(0.95f, 0.55f, 0.40f, 1)
                 : v.status == MP_STATUS_HOSTING || v.status == MP_STATUS_CONNECTED ? ImVec4(0.58f, 0.88f, 0.54f, 1)
                                                                                   : ImVec4(0.78f, 0.74f, 0.66f, 1);
    ImGui::TextColored(col, "%s", status_text(v.status));
    ImGui::SameLine();
    if (v.status == MP_STATUS_HOSTING)
        ImGui::TextDisabled("%s %u  -  %d/%d  -  %s", menu_text(UI_MENU_MP_PORT), (unsigned)v.port, v.player_count,
                            MP_MAX_PLAYERS, v.local_name);
    else if (v.status == MP_STATUS_CONNECTED)
        ImGui::TextDisabled("%s  -  %d/%d  -  %u ms", v.host_addr, v.player_count, MP_MAX_PLAYERS, (unsigned)v.rtt_ms);
    else if (v.status == MP_STATUS_ERROR && v.error[0]) ImGui::TextDisabled("%s", v.error);
    else ImGui::TextDisabled("%s %u", menu_text(UI_MENU_MP_PORT), (unsigned)v.port);
}

static void roster_block(const MpSessionView &v) {
    ImGui::BeginChild("##mp_roster", ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    status_line(v);
    if (v.status == MP_STATUS_OFF || v.status == MP_STATUS_ERROR) {
        ImGui::TextWrapped("%s", menu_text(UI_MENU_MP_MODE_DESC));
        if (v.status == MP_STATUS_ERROR && ImGui::Button(menu_text(UI_MENU_MP_RETRY))) mp_retry();
    } else {
        for (int i = 0; i < v.player_count; i++) roster_row(v.players[i]);
        if (v.status == MP_STATUS_CONNECTING) {   // the join is still being answered: one line, until a slot arrives
            ImGui::TextDisabled("%s...", menu_text(UI_MENU_MP_STATUS_CONNECTING));
        }
        if (v.re_emitted_draws)   // the draws repeated for the remote players, the price of seeing them
            ImGui::TextDisabled("%s: %llu", menu_text(UI_MENU_MP_STATS), (unsigned long long)v.re_emitted_draws);
    }
    ImGui::EndChild();
}

// The LAN list: what the last discovery found, each host a press away from joining it.
static void lan_block(const MpSessionView &v) {
    if (ImGui::Button(menu_text(UI_MENU_MP_LAN_SCAN))) mp_request_lan_scan();
    ImGui::SameLine();
    if (v.lan_count <= 0) {
        ImGui::TextWrapped("%s", menu_text(UI_MENU_MP_LAN_NONE));
        return;
    }
    for (int i = 0; i < v.lan_count; i++) {
        const MpLanSession &s = v.lan_sessions[i];
        ImGui::PushID(i);
        ImGui::Text("%s", s.host_name);
        ImGui::SameLine();
        ImGui::TextDisabled("%s  %u/%u", s.host_ip, (unsigned)s.player_count, (unsigned)s.max_players);
        ImGui::SameLine();
        if (ImGui::SmallButton(menu_text(UI_MENU_MP_LAN_JOIN))) {   // the address and the port, then join it
            char port[16];
            snprintf(port, sizeof port, "%u", (unsigned)s.port);
            opt_apply("NX2NAVI_MP_HOST", s.host_ip);
            opt_apply("NX2NAVI_MP_PORT", port);
            opt_apply("NX2NAVI_MP_MODE", "join");
            mp_retry();
        }
        ImGui::PopID();
    }
}

// ---- typing a text setting (the player's name, the host address)

enum { FIELD_NONE = 0, FIELD_NAME, FIELD_HOST };

static struct {
    int field = FIELD_NONE;   // the window open, 0 none
    bool open;                // the popup is to be opened by the next frame (OpenPopup, once)
    bool focus;               // the text field takes the keyboard at the next frame (the window opening, a press on it)
    char text[64];            // what it holds
    char label[96];           // its title, kept while it is open (the menu language can change under it)
} g_edit;

// What a character button of the window writes into the text.
static void type_char(unsigned c) {
    size_t n = strlen(g_edit.text), cap = sizeof g_edit.text - 1;
    if (n + 1 > cap) return;
    if (c == 8) {   // backspace
        if (n) g_edit.text[n - 1] = 0;
        return;
    }
    if ((c == ' ') && (n == 0 || g_edit.text[n - 1] == ' ')) return;   // no run of spaces, no leading one
    g_edit.text[n] = (char)c, g_edit.text[n + 1] = 0;
}

static void open_edit(int field, const char *value) {
    g_edit.field = field;
    g_edit.open = true;
    g_edit.focus = true;
    snprintf(g_edit.text, sizeof g_edit.text, "%s", value ? value : "");
    snprintf(g_edit.label, sizeof g_edit.label, "%s",
             menu_text(field == FIELD_NAME ? UI_MENU_MP_NAME : UI_MENU_MP_HOST));
}

bool mp_modal(void) { return g_edit.field != FIELD_NONE; }

void mp_open_field(int field) {
    const MpSessionView &v = session_view();
    if (field == FIELD_NAME) open_edit(FIELD_NAME, v.local_name);
    else if (field == FIELD_HOST) open_edit(FIELD_HOST, v.host_addr);
}

void mp_cancel(void) { g_edit.field = FIELD_NONE; }

// The row that opens the window: a text row (opt_text_row) whose A / click answer is 2.
static void text_row(const char *id, UiTextId label, UiTextId desc, const char *value, int field) {
    const char *shown = value && *value ? value : "-";
    char note[128];
    snprintf(note, sizeof note, "%s...", menu_text(UI_MENU_MP_TYPE));
    if (opt_text_row(id, menu_text(label), shown, note, menu_text(desc), false) == 2) open_edit(field, value);
}

// The window itself: the title, the text as it stands, the keyboard, and Cancel / OK.
static void edit_window(void) {
    if (g_edit.field == FIELD_NONE) return;
    const bool name = g_edit.field == FIELD_NAME;
    if (g_edit.open) {   // opened once: a repeated OpenPopup would reset the window (its size, its text field)
        g_edit.open = false;
        ImGui::OpenPopup("##mp_edit");
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    if (!ImGui::BeginPopupModal("##mp_edit", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                                          ImGuiWindowFlags_NoSavedSettings)) {
        g_edit.field = FIELD_NONE;   // closed from elsewhere (the menu closing): nothing stays open
        return;
    }
    const float fs = ImGui::GetFontSize(), st = ImGui::GetStyle().ItemSpacing.x;
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.38f, 1), "%s", g_edit.label);
    // The keyboard: ten to a row, every row full, so a row's characters are where the player expects them. A host name
    // takes letters, digits, '.', '-' and ':' (an IPv6 address goes into nx2navi.ini); a player's name letters, digits, a
    // space and the two marks a name often has. Each button is a nav item and its press inserts its character.
    static const char *const kLetters[3] = {"ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZ0123"};
    static const char *const kMarks[2] = {"456789.-:", "456789 .-'"};   // for a host name, for a player's name
    const float bw = fs * 1.55f, bh = fs * 1.5f, key_w = bw * 10 + st * 0.5f * 9;
    // The field: as wide as the keys, so every line of the window ends in the same column. The longest text it takes, at
    // the field's own padding, is what sets that width (the keys are wider either way).
    char room[64];
    int units = name ? MP_NAME_MAX - 2 : 40;
    memset(room, 'W', (size_t)units), room[units] = 0;
    ImGui::PushItemWidth(fmaxf(ImGui::CalcTextSize(room).x + ImGui::GetStyle().FramePadding.x * 2, key_w));
    // The field takes the keyboard when the window opens (Enter is OK, EnterReturnsTrue) and again with A / a click on
    // it once the player has moved on to the keys; typing into it goes through the menu's characters (menu_event_char).
    if (g_edit.focus) {
        ImGui::SetKeyboardFocusHere();
        g_edit.focus = false;
    }
    const bool enter = ImGui::InputText("##mp_edit_text", g_edit.text, sizeof g_edit.text,
                                        ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
    ImGui::PopItemWidth();
    // the hint wraps to the keyboard's width, so the window is as wide as the keys and no wider
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + key_w);
    ImGui::TextDisabled("%s", menu_text(UI_MENU_MP_TYPE_HINT));
    ImGui::PopTextWrapPos();
    for (int r = 0; r < 4; r++) {
        const char *row = r < 3 ? kLetters[r] : kMarks[name ? 1 : 0];
        for (int c = 0; c < 10; c++) {
            char ch = row[c];   // every row is ten characters, spaces included
            ImGui::PushID(r * 10 + c);
            char label[8];
            if (ch == ' ') snprintf(label, sizeof label, "  ###k%d", c);
            else snprintf(label, sizeof label, "%c###k%c", ch, ch);
            if (ImGui::Button(label, ImVec2(bw, bh))) type_char((unsigned char)ch);
            ImGui::PopID();
            if (c < 9) ImGui::SameLine(0, st * 0.5f);
        }
    }
    const float half = (key_w - st) * 0.5f;
    if (ImGui::Button(menu_text(UI_MENU_MP_ERASE), ImVec2(half, bh))) type_char(8);
    ImGui::SameLine(0, st);
    if (ImGui::Button(menu_text(UI_MENU_MP_CLEAR), ImVec2(half, bh))) g_edit.text[0] = 0;
    ImGui::Spacing();
    bool cancel = ImGui::Button(menu_text(UI_SWKBD_CANCEL), ImVec2(half, 0));
    ImGui::SameLine(0, st);
    bool ok = ImGui::Button(menu_text(UI_SWKBD_OK), ImVec2(half, 0)) || enter;
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false))
        cancel = true, menu_sound_play(MENU_SOUND_CANCEL);
    if (ok) {
        // a host name is not trimmed: what was typed is what is resolved; an empty field unsets its key
        if (name) {
            char cut[MP_NAME_MAX];
            snprintf(cut, sizeof cut, "%s", g_edit.text);
            opt_apply("NX2NAVI_MP_NAME", cut);
        } else {
            opt_apply("NX2NAVI_MP_HOST", g_edit.text);
        }
        rt_log("mp: %s on the Multiplayer tab set to \"%s\"", name ? "name" : "host", g_edit.text);
    }
    if (ok || cancel) {
        ImGui::CloseCurrentPopup();
        g_edit.field = FIELD_NONE;
        opt_save_soon();
    }
    ImGui::EndPopup();
}

void mp_page_after(void) { edit_window(); }

// ---- the page

void tab_mp(float scale) {
    (void)scale;   // the rows take ImGui's own sizes, already scaled with the font
    const MpSessionView &v = session_view();
    opt_begin("##mp", scale);
    opt_group(UI_MENU_MP_SESSION);
    static const UiTextId kModes[3] = {UI_MENU_MP_MODE_OFF, UI_MENU_MP_MODE_HOST, UI_MENU_MP_MODE_JOIN};
    opt_choice("NX2NAVI_MP_MODE", UI_MENU_MP_MODE, kModes, 3, UI_MENU_MP_MODE_DESC);
    text_row("NX2NAVI_MP_HOST", UI_MENU_MP_HOST, UI_MENU_MP_HOST_DESC, v.host_addr, FIELD_HOST);
    opt_slider("NX2NAVI_MP_PORT", UI_MENU_MP_PORT, 1024, 65535, UI_MENU_MP_PORT_DESC, 1, "%.0f");
    if (v.status == MP_STATUS_ERROR && opt_button("##mp_retry", UI_MENU_MP_RETRY, UI_MENU_MP_MODE_DESC)) mp_retry();
    opt_text(UI_MENU_MP_ROSTER);
    roster_block(v);
    opt_group(UI_MENU_MP_LAN);
    lan_block(v);

    opt_group(UI_MENU_MP_PLAYER);
    text_row("NX2NAVI_MP_NAME", UI_MENU_MP_NAME, UI_MENU_MP_NAME_DESC, v.local_name, FIELD_NAME);
    static const UiTextId kColors[MP_COLOR_COUNT] = {UI_MENU_MP_COLOR_GREEN, UI_MENU_MP_COLOR_BLUE, UI_MENU_MP_COLOR_RED,
                                                     UI_MENU_MP_COLOR_VIOLET, UI_MENU_MP_COLOR_GOLD,
                                                     UI_MENU_MP_COLOR_CYAN};
    opt_choice("NX2NAVI_MP_COLOR", UI_MENU_MP_COLOR, kColors, MP_COLOR_COUNT, UI_MENU_MP_COLOR_DESC);
    opt_toggle("NX2NAVI_MP_RENDER_PLAYERS", UI_MENU_MP_RENDER, UI_MENU_MP_RENDER_DESC);
    opt_toggle("NX2NAVI_MP_NAMETAGS", UI_MENU_MP_NAMETAGS, UI_MENU_MP_NAMETAGS_DESC);
    opt_toggle("NX2NAVI_MP_HUD", UI_MENU_MP_HUD, UI_MENU_MP_HUD_DESC);
    opt_slider("NX2NAVI_MP_INTERP_MS", UI_MENU_MP_INTERP, 0, 300, UI_MENU_MP_INTERP_DESC, 10, "%.0f ms");
    if (opt_button("##mp_beacon", UI_MENU_MP_PING, UI_MENU_MP_PING_DESC)) mp_send_beacon();
    opt_end();
}

// ---- the in-world overlay (mp_overlay_draw): nametags, off-screen arrows, beacons, the mini-HUD

// A rounded pill with a player's colour along its left edge: the nametag's background.
static void pill(ImDrawList *dl, ImVec2 a, ImVec2 b, uint8_t color, float alpha) {
    float r = (b.y - a.y) * 0.5f;
    dl->AddRectFilled(a, b, IM_COL32(12, 8, 4, (int)(170 * alpha)), r);
    dl->AddRectFilled(a, ImVec2(a.x + fmaxf(r * 0.7f, 2.0f), b.y), ImGui::GetColorU32(accent(color)), r);
    dl->AddRect(a, b, IM_COL32(150, 122, 70, (int)(190 * alpha)), r, 0, 1.0f);
}

void mp_overlay_draw(float w, float h, float scale, const float pic[4]) {
    if (!mp_active() || w < 2 || h < 2) return;
    const bool tags = cfg_bool("NX2NAVI_MP_NAMETAGS"), hud = cfg_bool("NX2NAVI_MP_HUD");
    if (!tags && !hud) return;
    const MpSessionView &v = session_view();
    // under the menu's window while it is open (as the performance overlay is), over the game otherwise
    ImDrawList *dl = menu_is_open() ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
    dl->PushClipRectFullScreen();
    // The overlay follows the game's picture: where the last present put it in the image (the window's bars are black and
    // hold nothing of the scene), or the whole image before the first present.
    const bool have_pic = pic && pic[2] > 0 && pic[3] > 0;
    const float px = have_pic ? pic[0] * w : 0, py = have_pic ? pic[1] * h : 0;
    const float pw = have_pic ? pic[2] * w : w, ph = have_pic ? pic[3] * h : h;
    const float fs = fmaxf(ImGui::GetFontSize() * 0.95f, 13.0f), th = fs * 1.5f, pad = fs * 0.45f;
    ImFont *font = ImGui::GetFont();
    if (tags) {
        for (int i = 0; i < v.player_count; i++) {
            const MpPlayerView &p = v.players[i];
            if (p.is_local) continue;
            char label[MP_NAME_MAX + 32], dist[32];
            format_distance(dist, sizeof dist, p.distance_m);
            snprintf(label, sizeof label, "%s  %s", p.name, dist);
            ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, label);
            // above the player's head: a little over its own height in world units
            float head[3] = {p.pos[0], p.pos[1] + 1.85f, p.pos[2]};
            float u = 0, vv = 0, cw = 0;
            bool front = mp_project_world_to_screen(head, &u, &vv, &cw) && cw > 0.05f;
            if (front && u > -0.05f && u < 1.05f && vv > -0.05f && vv < 1.05f) {   // over the head, in the picture
                float sx = px + u * pw, sy = py + vv * ph;
                ImVec2 a(sx - ts.x * 0.5f - pad, sy - th - pad * 0.5f), b(a.x + ts.x + pad * 2, a.y + th);
                pill(dl, a, b, p.color_id, 1.0f);
                dl->AddText(font, fs, ImVec2(a.x + pad * 1.5f, a.y + (th - ts.y) * 0.5f), SKIN_TEXT, label);
            } else {   // off screen or behind the camera: an arrow at the picture's edge, pointing at the player
                float dx = u - 0.5f, dy = vv - 0.5f;
                if (!isfinite(dx) || !isfinite(dy) || (dx == 0 && dy == 0)) dx = 0, dy = -1;
                float len = sqrtf(dx * dx + dy * dy);
                dx /= len, dy /= len;
                float m = fs * 1.8f, ex = px + pw * 0.5f, ey = py + ph * 0.5f;
                // the edge of the inset rectangle along that direction
                float k = fminf((pw * 0.5f - m) / fmaxf(fabsf(dx), 1e-4f), (ph * 0.5f - m) / fmaxf(fabsf(dy), 1e-4f));
                if (k < 0) k = 0;
                ex += dx * k, ey += dy * k;
                float s2 = fs * 0.5f, nx = -dy, ny = dx;
                ImVec2 tip(ex + dx * s2, ey + dy * s2);
                dl->AddTriangleFilled(tip, ImVec2(ex - dx * s2 + nx * s2, ey - dy * s2 + ny * s2),
                                      ImVec2(ex - dx * s2 - nx * s2, ey - dy * s2 - ny * s2),
                                      ImGui::GetColorU32(accent(p.color_id)));
                float lw = ts.x * 0.9f + pad * 2, lx = ex - lw * 0.5f;
                // inside the picture: the bars at its sides are black, and a label half way over one would be cut
                if (lx < px + pad * 0.5f) lx = px + pad * 0.5f;
                if (lx + lw > px + pw - pad * 0.5f) lx = fmaxf(px + pad * 0.5f, px + pw - pad * 0.5f - lw);
                ImVec2 a(lx, ey + s2 * 1.6f), b(lx + lw, a.y + th * 0.8f);
                pill(dl, a, b, p.color_id, 0.85f);
                dl->AddText(font, fs * 0.8f, ImVec2(a.x + pad, a.y + th * 0.15f), SKIN_TEXT, label);
            }
        }
        // the beacons: a diamond in the sender's colour that fades with its age, and the sender's name under it
        for (int i = 0; i < v.beacon_count; i++) {
            const MpBeaconView &b = v.beacons[i];
            float u = 0, vv = 0, cw = 0;
            if (!mp_project_world_to_screen(b.pos, &u, &vv, &cw) || cw <= 0.05f) continue;
            if (u < -0.02f || u > 1.02f || vv < -0.02f || vv > 1.02f) continue;
            float life = b.duration_s > 0 ? b.duration_s : 1.0f, fade = 1.0f - b.age_s / life;
            if (fade < 0) fade = 0;
            if (fade > 1) fade = 1;
            float sx = px + u * pw, sy = py + vv * ph, r = fs * (0.55f + 0.2f * sinf(b.age_s * 4.0f));
            ImVec4 c = accent(b.color_id);
            ImU32 col = ImGui::GetColorU32(ImVec4(c.x, c.y, c.z, fade));
            dl->AddQuadFilled(ImVec2(sx, sy - r), ImVec2(sx + r, sy), ImVec2(sx, sy + r), ImVec2(sx - r, sy), col);
            ImVec2 ts = font->CalcTextSizeA(fs * 0.85f, FLT_MAX, 0, b.sender);
            dl->AddText(font, fs * 0.85f, ImVec2(sx - ts.x * 0.5f, sy + r * 1.25f), col, b.sender);
        }
    }
    if (hud) {   // the session's line and its players, in the picture's top left corner
        char line[160], names[MP_MAX_PLAYERS][MP_NAME_MAX + 16];
        snprintf(line, sizeof line, "%s  -  %s", menu_text(UI_MENU_TAB_MP), status_text(v.status));
        float lw = font->CalcTextSizeA(fs, FLT_MAX, 0, line).x;
        for (int i = 0; i < v.player_count && i < MP_MAX_PLAYERS; i++) {
            snprintf(names[i], sizeof names[i], "%s%s%s", v.players[i].name, v.players[i].is_local ? " " : "",
                     v.players[i].is_local ? menu_text(UI_MENU_MP_YOU) : "");
            float nw = font->CalcTextSizeA(fs * 0.92f, FLT_MAX, 0, names[i]).x + fs * 7;
            if (nw > lw) lw = nw;
        }
        float bw = lw + pad * 3.4f, bh = th + (float)v.player_count * fs * 1.45f + pad * 3.0f;
        ImVec2 a(px + pad * 1.5f, py + pad * 1.5f), b(a.x + bw, a.y + bh);
        skin_panel(dl, a, b, scale * 0.75f, 0.82f);
        dl->AddText(font, fs, ImVec2(a.x + pad * 1.3f, a.y + pad * 1.1f), SKIN_GOLD, line);
        for (int i = 0; i < v.player_count && i < MP_MAX_PLAYERS; i++) {
            const MpPlayerView &p = v.players[i];
            float y = a.y + pad * 1.1f + th + (float)i * fs * 1.45f;
            dl->AddCircleFilled(ImVec2(a.x + pad * 1.7f, y + fs * 0.45f), fs * 0.22f,
                                ImGui::GetColorU32(accent(p.color_id)));
            dl->AddText(font, fs * 0.92f, ImVec2(a.x + pad * 2.5f, y), SKIN_TEXT, names[i]);
            if (!p.is_local) {
                char right[64], dist[32];
                format_distance(dist, sizeof dist, p.distance_m);
                if (p.ping_ms) snprintf(right, sizeof right, "%s  %u ms", dist, (unsigned)p.ping_ms);
                else snprintf(right, sizeof right, "%s", dist);
                float rw = font->CalcTextSizeA(fs * 0.86f, FLT_MAX, 0, right).x;
                dl->AddText(font, fs * 0.86f, ImVec2(b.x - pad * 1.4f - rw, y + fs * 0.08f), SKIN_TEXT_DIM, right);
            }
        }
    }
    dl->PopClipRect();
}
