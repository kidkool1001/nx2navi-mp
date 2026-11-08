// Draws the in-game settings menu and the performance overlay without the game, for looking at their design: the
// menu's own row code (runtime/ui/option_rows.cpp), its shapes (ui/menu_skin.cpp), the overlay panel
// (ui/perf_panel.cpp) and the text catalog (ui_text.c), with the few runtime calls they make answered here, laid out as
// menu.cpp layout_menu lays out the window. Dear ImGui's draw lists are rasterised on the CPU into an RGBA image.
// Without an art folder the window has its plain border in place of the game's ornamented frame and every box is plain
// shapes; with one (tools/menu_preview.py --romfs) the boxes use the game's textures as in the game. The given font
// stands in for the game's either way.
//
// tools/menu_preview.py builds and runs it:
//   menu_preview <font.ttf> <scene: menu | mp | text | hud | overlay | boot> <width> <height> <out.rgba> [art folder]
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "ui/menu_art.h"
#include "ui/mp_page.h"
#include "ui/menu_skin.h"
#include "ui/option_rows.h"
#include "ui/perf_panel.h"

extern "C" {
#include "cfg.h"
#include "menu_sound.h"
#include "multiplayer.h"
#include "ui_text.h"
}

// ---- what the rows and the catalog ask of the runtime: a few settings with the menu's values and defaults

static const CfgMeta kMeta[] = {
    {"FULLSCREEN", "1", "0|1", 1},       {"WINDOW_SIZE", "", NULL, 1},  {"VK_PRESENT", "mailbox", "mailbox|fifo|immediate", 1},
    {"GPU_INDEX", "", NULL, 0},          {"PERF_OVERLAY", "off", "off|fps|full", 1},
    {"GAME_ASPECT", "16:9", "16:9|window|21:9|32:9", 0}, {"RENDER_SCALE", "", NULL, 0},
    {"UPSCALER", "bilinear", "bilinear|fsr", 0},         {"SHARPNESS", "50", "0..100", 1},
    {"ASPECT", "fit", "fit|stretch", 1}, {"MENU_LANG", "", NULL, 1},
    {"MP_MODE", "host", "off|host|join", 1}, {"MP_HOST", "127.0.0.1", NULL, 1},
    {"MP_PORT", "27015", "1024..65535", 1},  {"MP_NAME", "Link", NULL, 1},
    {"MP_COLOR", "green", "green|blue|red|violet|gold|cyan", 1}, {"MP_NAMETAGS", "1", "0|1", 1},
    {"MP_HUD", "1", "0|1", 1}, {"MP_RENDER_PLAYERS", "1", "0|1", 1}, {"MP_INTERP_MS", "100", "0..300", 1},
};
static const struct { const char *key, *value; } kSet[] = {{"VK_PRESENT", "mailbox"}, {"PERF_OVERLAY", "full"}};

static const char *strip(const char *name) { return strncmp(name, "NX2NAVI_", 8) ? name : name + 8; }

extern "C" {
const CfgMeta *cfg_meta(const char *key) {
    for (const CfgMeta &m : kMeta)
        if (!strcasecmp(m.key, strip(key))) return &m;
    return NULL;
}
const CfgMeta *cfg_meta_at(int i) { return i >= 0 && i < (int)(sizeof kMeta / sizeof *kMeta) ? &kMeta[i] : NULL; }
const char *cfg_get(const char *name) {
    for (auto &s : kSet)
        if (!strcasecmp(s.key, strip(name))) return s.value;
    return NULL;
}
const char *cfg_value(const char *name) {
    const char *v = cfg_get(name);
    const CfgMeta *m = cfg_meta(name);
    return v ? v : m ? m->def : "";
}
const char *cfg_default(const char *key) {   // no device profile here: the row's own default
    const CfgMeta *m = cfg_meta(key);
    return m ? m->def : NULL;
}
int cfg_int(const char *name) { return atoi(cfg_value(name)); }
double cfg_double(const char *name) { return atof(cfg_value(name)); }
int cfg_bool(const char *name) { return strcmp(cfg_value(name), "0") != 0; }
int cfg_from_env(const char *) { return 0; }
void cfg_set(const char *, const char *) {}
int cfg_save(void) { return 1; }
int cfg_reset(const char *const *, int) { return 0; }
unsigned cfg_version(void) { return 1; }
void menu_sound_play(MenuSound) {}
const char *ov_language(void) { return "en-US"; }
const char *plat_ui_language(void) { return "en"; }
uint64_t plat_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
void rt_log(const char *, ...) {}

// multiplayer.c's snapshot (multiplayer.h): a session of three, for the Multiplayer tab and the overlay
void mp_get_session_view(MpSessionView *out) {
    memset(out, 0, sizeof *out);
    out->mode = MP_MODE_HOST, out->status = MP_STATUS_HOSTING, out->port = 27015;
    snprintf(out->host_addr, sizeof out->host_addr, "192.168.1.24");
    snprintf(out->local_name, sizeof out->local_name, "Link");
    out->local_color = MP_COLOR_GREEN, out->local_slot = 0, out->rtt_ms = 24;
    out->player_count = 3, out->remote_count = 2;
    out->tx_bytes = 1284 * 1024, out->rx_bytes = 931 * 1024, out->re_emitted_draws = 18422;
    out->local_pos_valid = 1;
    MpPlayerView *p = &out->players[0];
    p->active = 1, p->is_local = 1, p->slot = 0, p->color_id = MP_COLOR_GREEN;
    snprintf(p->name, sizeof p->name, "Link");
    p->skel_bones = 96;
    p = &out->players[1];
    p->active = 1, p->slot = 1, p->color_id = MP_COLOR_CYAN, p->ping_ms = 31, p->distance_m = 12.4f, p->skel_bones = 96;
    snprintf(p->name, sizeof p->name, "Navi");
    p->pos[0] = 3.0f, p->pos[1] = 0.2f, p->pos[2] = -11.0f;
    p = &out->players[2];
    p->active = 1, p->slot = 2, p->color_id = MP_COLOR_VIOLET, p->ping_ms = 58, p->distance_m = 240.7f, p->skel_bones = 48;
    snprintf(p->name, sizeof p->name, "Saria");
    p->pos[0] = -90.0f, p->pos[1] = 1.5f, p->pos[2] = 220.0f;
    out->beacon_count = 1;
    MpBeaconView *b = &out->beacons[0];
    b->active = 1, b->slot = 1, b->color_id = MP_COLOR_CYAN, b->age_s = 1.5f, b->duration_s = 8.0f;
    snprintf(b->sender, sizeof b->sender, "Navi");
    out->lan_count = 2;
    snprintf(out->lan_sessions[0].host_ip, 64, "192.168.1.24"), out->lan_sessions[0].port = 27015;
    snprintf(out->lan_sessions[0].host_name, MP_NAME_MAX, "Zelda");
    out->lan_sessions[0].player_count = 3, out->lan_sessions[0].max_players = 8;
    snprintf(out->lan_sessions[1].host_ip, 64, "192.168.1.31"), out->lan_sessions[1].port = 27015;
    snprintf(out->lan_sessions[1].host_name, MP_NAME_MAX, "Impa");
    out->lan_sessions[1].player_count = 1, out->lan_sessions[1].max_players = 4;
}
int mp_active(void) { return 1; }
int menu_is_open(void) { return 0; }
void mp_retry(void) {}
void mp_request_lan_scan(void) {}
int mp_send_beacon(void) { return 1; }
int mp_teleport_to_slot(int) { return 1; }
int mp_teleport_to_pos(const float *) { return 1; }
int mp_overlay_wanted(void) { return 1; }
void mp_color_rgba(uint8_t id, float rgba[4]) {
    static const float k[6][3] = {{0.35f, 0.85f, 0.40f}, {0.36f, 0.62f, 0.98f}, {0.95f, 0.36f, 0.32f},
                                  {0.72f, 0.45f, 0.95f}, {0.96f, 0.80f, 0.30f}, {0.35f, 0.90f, 0.92f}};
    int i = id < 6 ? id : 0;
    rgba[0] = k[i][0], rgba[1] = k[i][1], rgba[2] = k[i][2], rgba[3] = 1;
}
// the scene camera: straight down -Z, looking at the origin (the players of the snapshot are around it)
int mp_project_world_to_screen(const float world[3], float *u, float *v, float *cw) {
    float z = 24.0f - world[2];   // in front of the camera when positive
    if (z < 0.1f) {   // behind it: off screen, the arrow's direction
        if (u) *u = world[0] < 0 ? 0.02f : 0.98f;
        if (v) *v = 0.5f;
        if (cw) *cw = z;
        return 1;
    }
    if (u) *u = 0.5f + world[0] * 0.06f;
    if (v) *v = 0.55f - world[1] * 0.10f;
    if (cw) *cw = z;
    return 1;
}
}

// ---- the game's art, from files menu_preview.py decoded: NN.art, a width and a height (32-bit) and the RGBA pixels of
// the MenuArt NN, with the smaller levels the game uploads

struct Level {
    int w, h;
    std::vector<uint8_t> px;
};
static std::vector<Level> g_art[MENU_ART_COUNT];

static void art_load(const char *dir) {
    for (int i = 0; i < MENU_ART_COUNT; i++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/%02d.art", dir, i);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        uint32_t size[2];
        if (fread(size, 4, 2, f) == 2) {
            Level top{(int)size[0], (int)size[1], std::vector<uint8_t>((size_t)size[0] * size[1] * 4)};
            if (fread(top.px.data(), 1, top.px.size(), f) == top.px.size()) {
                skin_art_prepare(i, top.px.data(), size[0], size[1]);
                g_art[i].push_back(std::move(top));
                while (g_art[i].size() < 6 && (g_art[i].back().w > 1 || g_art[i].back().h > 1)) {
                    const Level &last = g_art[i].back();
                    Level next{last.w > 1 ? last.w / 2 : 1, last.h > 1 ? last.h / 2 : 1, {}};
                    next.px.resize((size_t)next.w * next.h * 4);
                    skin_art_halve(last.px.data(), last.w, last.h, next.px.data(), next.w, next.h);
                    g_art[i].push_back(std::move(next));
                }
            }
        }
        fclose(f);
    }
}

// Texture ids in the draw lists: 1 is the font atlas, 2 and up the art.
extern "C" int menu_art(MenuArt id, ImTextureID *tex, float *w, float *h) {
    if ((unsigned)id >= MENU_ART_COUNT || g_art[id].empty()) return 0;
    *tex = (ImTextureID)(intptr_t)(2 + id);
    if (w) *w = (float)g_art[id][0].w;
    if (h) *h = (float)g_art[id][0].h;
    return 1;
}

// ---- the scenes

static void backdrop(float w, float h) {
    // a dim field and sky, as the game shows behind the menu; then the menu's own dimming (menu.cpp layout_menu)
    ImDrawList *bg = ImGui::GetBackgroundDrawList();
    bg->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(w, h * 0.55f), IM_COL32(62, 70, 74, 255), IM_COL32(70, 76, 78, 255),
                                IM_COL32(52, 64, 52, 255), IM_COL32(48, 60, 50, 255));
    bg->AddRectFilledMultiColor(ImVec2(0, h * 0.55f), ImVec2(w, h), IM_COL32(52, 64, 52, 255), IM_COL32(48, 60, 50, 255),
                                IM_COL32(30, 38, 30, 255), IM_COL32(28, 34, 28, 255));
}

static void triforce(ImDrawList *dl, float x, float y, float h, ImU32 col) {
    float w = h * 1.1547f;
    ImVec2 t(x, y), l(x - w / 2, y + h), r(x + w / 2, y + h), ml(x - w / 4, y + h / 2), mr(x + w / 4, y + h / 2), mb(x, y + h);
    dl->AddTriangleFilled(t, mr, ml, col);
    dl->AddTriangleFilled(ml, mb, l, col);
    dl->AddTriangleFilled(mr, r, mb, col);
}

static void scene_menu(float w, float h, float scale, ImFont *title, bool first) {
    backdrop(w, h);
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), IM_COL32(10, 6, 2, 150));
    float ww = fminf(w * 0.88f, 1400 * scale), wh = fminf(h * 0.86f, 820 * scale);
    ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(ww, wh), ImGuiCond_Always);
    ImTextureID frame_tex;
    const bool framed = menu_art(MENU_ART_FRAME_DECO, &frame_tex, nullptr, nullptr);
    if (framed) ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("##menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    if (framed) ImGui::PopStyleVar();
    const ImVec2 win_p = ImGui::GetWindowPos(), win_s = ImGui::GetWindowSize();
    if (framed) {   // as menu.cpp layout_menu draws it
        ImDrawList *fdl = ImGui::GetWindowDrawList();
        const float ox = 15 * scale, oy = 30 * scale;
        fdl->PushClipRectFullScreen();
        for (int turned = 0; turned < 2; turned++)
            menu_art_draw9_turned(fdl, MENU_ART_FRAME_DECO, ImVec2(win_p.x - ox, win_p.y - oy),
                                  ImVec2(win_p.x + win_s.x + ox, win_p.y + win_s.y + oy), 96 * scale, 0.2f,
                                  IM_COL32(236, 196, 112, 235), 0, turned);
        fdl->PopClipRect();
    }
    const ImVec4 gold(0.93f, 0.76f, 0.38f, 1);
    ImGui::PushFont(title);
    ImVec2 at = ImGui::GetCursorScreenPos();
    float th = ImGui::GetTextLineHeight();
    triforce(ImGui::GetWindowDrawList(), at.x + th * 0.6f, at.y + th * 0.02f, th * 0.96f, ImGui::GetColorU32(gold));
    ImGui::SetCursorScreenPos(ImVec2(at.x + th * 1.5f, at.y));
    ImGui::TextColored(gold, "Ocarina of Time");
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextUnformatted(menu_text(UI_MENU_TITLE));
    const float title_end = ImGui::GetItemRectMax().x;
    ImGui::PopFont();
    const char *version = "NX2navi 0.5.0  -  game 1.1.1";
    float version_x = ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x - ImGui::CalcTextSize(version).x;
    if (ImGui::GetWindowPos().x + version_x >= title_end + ImGui::GetStyle().ItemSpacing.x * 2) {
        ImGui::SameLine(version_x);
        ImGui::TextDisabled("%s", version);
    }
    const ImGuiStyle &st = ImGui::GetStyle();
    float bottom = ImGui::GetFrameHeight() + st.ItemSpacing.y * 2;
    static const UiTextId tabs[] = {UI_MENU_TAB_SCREEN, UI_MENU_TAB_GAME, UI_MENU_TAB_SOUND, UI_MENU_TAB_CONTROLS};
    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_NoTabListScrollingButtons)) {
        for (int k = 0; k < 4; k++) {
            char label[96];
            snprintf(label, sizeof label, "  %s  ###tab%d", menu_text(tabs[k]), k);
            bool shown = skin_tab_item(label, menu_text(tabs[k]), 0, scale, title);
            if (!shown) continue;
            ImGui::Spacing();
            opt_reserve_bottom(bottom);
            if (first) opt_focus_first();
            // the Screen tab's first rows, as menu.cpp tab_screen shows them
            opt_begin("##screen", scale);
            opt_toggle("NX2NAVI_FULLSCREEN", UI_MENU_FULLSCREEN, UI_MENU_FULLSCREEN_DESC);
            static const char *const sizes[] = {"", "1920x1080"};
            static const char *const size_names[] = {"1920 \xC3\x97 1080", "1920 \xC3\x97 1080"};
            opt_choice_list("NX2NAVI_WINDOW_SIZE", UI_MENU_WINDOW_SIZE, sizes, size_names, 2, UI_MENU_WINDOW_SIZE_DESC, 0);
            opt_toggle_value("NX2NAVI_VK_PRESENT", UI_MENU_PRESENT_VSYNC, UI_MENU_PRESENT_DESC, "mailbox", "immediate");
            static const char *const gpus[] = {""};
            static const char *const gpu_names[] = {"NVIDIA GeForce RTX 3050 Ti Laptop GPU"};   // a long name
            opt_choice_list("NX2NAVI_GPU_INDEX", UI_MENU_GPU, gpus, gpu_names, 1, UI_MENU_GPU_DESC, 0);
            static const UiTextId perf[] = {UI_MENU_PERF_OFF, UI_MENU_PERF_FPS, UI_MENU_PERF_FULL};
            opt_choice("NX2NAVI_PERF_OVERLAY", UI_MENU_PERF_OVERLAY, perf, 3, UI_MENU_PERF_OVERLAY_DESC);
            opt_group(UI_MENU_GRAPHICS);
            static const UiTextId wide[] = {UI_MENU_GAME_ASPECT_16_9, UI_MENU_GAME_ASPECT_WINDOW, UI_MENU_GAME_ASPECT_21_9,
                                            UI_MENU_GAME_ASPECT_32_9};
            opt_choice("NX2NAVI_GAME_ASPECT", UI_MENU_GAME_ASPECT, wide, 4, UI_MENU_GAME_ASPECT_DESC);
            opt_slider("NX2NAVI_SHARPNESS", UI_MENU_SHARPNESS, 0, 100, UI_MENU_SHARPNESS_DESC, 5, "%.0f %%");
            opt_end();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::Spacing();
    float y = ImGui::GetCursorPosY();
    {
        float sy = win_p.y + y, fh = ImGui::GetFrameHeight();
        menu_art_draw(ImGui::GetWindowDrawList(), MENU_ART_GUIDE_BAR, ImVec2(win_p.x + st.WindowPadding.x, sy),
                      ImVec2(win_p.x + win_s.x - st.WindowPadding.x, sy + fh), IM_COL32(96, 70, 36, 170));
    }
    const char *cont = menu_text(UI_MENU_CONTINUE), *restart = menu_text(UI_MENU_RESTART_YES), *quit = menu_text(UI_MENU_QUIT);
    float bw = fmaxf(fmaxf(ImGui::CalcTextSize(cont).x, ImGui::CalcTextSize(restart).x), ImGui::CalcTextSize(quit).x) + st.FramePadding.x * 4;
    float gap = st.ItemSpacing.x + skin_cursor_reach(scale);
    ImGui::SetCursorPos(ImVec2(fmaxf(st.WindowPadding.x, ImGui::GetWindowWidth() - st.WindowPadding.x - bw * 3 - gap * 2), y));
    skin_widget_button(cont, ImVec2(bw, 0), scale);
    ImGui::SameLine(0, gap);
    skin_widget_button(restart, ImVec2(bw, 0), scale);
    ImGui::SameLine(0, gap);
    skin_widget_button(quit, ImVec2(bw, 0), scale);
    ImGui::End();
}

// The Multiplayer tab (ui/mp_page.cpp), laid out as menu.cpp lays out a tab: the tab bar, the page's rows and its
// window-less blocks, the bottom row.
static void scene_mp(float w, float h, float scale, ImFont *title) {
    backdrop(w, h);
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), IM_COL32(10, 6, 2, 150));
    float ww = fminf(w * 0.88f, 1400 * scale), wh = fminf(h * 0.86f, 820 * scale);
    ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(ww, wh), ImGuiCond_Always);
    ImTextureID frame_tex;
    const bool framed = menu_art(MENU_ART_FRAME_DECO, &frame_tex, nullptr, nullptr);
    if (framed) ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("##menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    if (framed) ImGui::PopStyleVar();
    const ImVec2 win_p = ImGui::GetWindowPos(), win_s = ImGui::GetWindowSize();
    if (framed) {
        ImDrawList *fdl = ImGui::GetWindowDrawList();
        const float ox = 15 * scale, oy = 30 * scale;
        fdl->PushClipRectFullScreen();
        for (int turned = 0; turned < 2; turned++)
            menu_art_draw9_turned(fdl, MENU_ART_FRAME_DECO, ImVec2(win_p.x - ox, win_p.y - oy),
                                  ImVec2(win_p.x + win_s.x + ox, win_p.y + win_s.y + oy), 96 * scale, 0.2f,
                                  IM_COL32(236, 196, 112, 235), 0, turned);
        fdl->PopClipRect();
    }
    const ImVec4 gold(0.93f, 0.76f, 0.38f, 1);
    ImGui::PushFont(title);
    ImGui::TextColored(gold, "Ocarina of Time");
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextUnformatted(menu_text(UI_MENU_TITLE));
    ImGui::PopFont();
    const ImGuiStyle &st = ImGui::GetStyle();
    float bottom = ImGui::GetFrameHeight() + st.ItemSpacing.y * 2;
    static const UiTextId tabs[] = {UI_MENU_TAB_SCREEN, UI_MENU_TAB_GAME, UI_MENU_TAB_SOUND, UI_MENU_TAB_CONTROLS,
                                    UI_MENU_TAB_MP};
    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_NoTabListScrollingButtons)) {
        for (int k = 0; k < 5; k++) {
            char label[96];
            snprintf(label, sizeof label, "  %s  ###tab%d", menu_text(tabs[k]), k);
            bool shown = skin_tab_item(label, menu_text(tabs[k]), k == 4 ? ImGuiTabItemFlags_SetSelected : 0, scale, title);
            if (!shown) continue;
            ImGui::Spacing();
            opt_reserve_bottom(bottom);
            tab_mp(scale);
            mp_page_after();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    const char *cont = menu_text(UI_MENU_CONTINUE), *restart = menu_text(UI_MENU_RESTART_YES), *quit = menu_text(UI_MENU_QUIT);
    float bw = fmaxf(fmaxf(ImGui::CalcTextSize(cont).x, ImGui::CalcTextSize(restart).x), ImGui::CalcTextSize(quit).x) + st.FramePadding.x * 4;
    float gap = st.ItemSpacing.x + skin_cursor_reach(scale), y = ImGui::GetCursorPosY();
    ImGui::SetCursorPos(ImVec2(fmaxf(st.WindowPadding.x, ImGui::GetWindowWidth() - st.WindowPadding.x - bw * 3 - gap * 2), y));
    skin_widget_button(cont, ImVec2(bw, 0), scale);
    ImGui::SameLine(0, gap);
    skin_widget_button(restart, ImVec2(bw, 0), scale);
    ImGui::SameLine(0, gap);
    skin_widget_button(quit, ImVec2(bw, 0), scale);
    ImGui::End();
}

// The Multiplayer tab's text window (ui/mp_page.cpp edit_window), open on the player's name.
static void scene_text(float w, float h, float scale, ImFont *title) {
    scene_mp(w, h, scale, title);
    // the window is a popup of the page's own window in the menu; here a silent host window stands in for it
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(1, 1));
    ImGui::Begin("##host", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs |
                                        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
    mp_open_field(1);
    mp_page_after();
    ImGui::End();
}

// The multiplayer overlay over a game picture (ui/mp_page.cpp mp_overlay_draw): nametags, an off-screen arrow, a beacon
// and the mini-HUD.
static void scene_hud(float w, float h, float scale) {
    backdrop(w, h);
    const float pic[4] = {0.05f, 0.08f, 0.9f, 0.84f};   // letterboxed, as a 16:10 window shows a 16:9 picture
    ImDrawList *bg = ImGui::GetBackgroundDrawList();
    bg->AddRectFilled(ImVec2(pic[0] * w, pic[1] * h), ImVec2((pic[0] + pic[2]) * w, (pic[1] + pic[3]) * h),
                      IM_COL32(70, 96, 66, 255));
    mp_overlay_draw(w, h, scale, pic);
}

static void scene_overlay(float w, float h, float scale) {
    backdrop(w, h);
    static PerfPanel p;   // the numbers of the screenshot the overlay was redesigned from
    PerfStats &s = p.s;
    s.fps = 60, s.frame_ms_avg = 16.7f, s.frame_ms_max = 20.2f, s.low1_fps = 47, s.low01_fps = 8;
    for (int i = 0; i < PERF_HISTORY; i++) s.frame_ms[i] = 16.7f + 1.4f * sinf(i * 0.7f) + (i % 17 == 0 ? 3.5f : 0);
    snprintf(s.cpu_name, sizeof s.cpu_name, "AMD Ryzen 7 5800X3D 8-Core Processor");
    s.cpu_total_pct = 50, s.cpu_game_pct = 44, s.cpu_temp_c = -1;
    s.ram_used_mb = 23.0f * 1024, s.ram_total_mb = 31.9f * 1024, s.ram_game_mb = 8.6f * 1024;
    snprintf(s.gpu_name, sizeof s.gpu_name, "AMD Radeon RX 7900 XT");
    s.gpu_pct = 56, s.gpu_temp_c = 69, s.gpu_power_w = 107, s.gpu_clock_mhz = 1312, s.vram_used_mb = 5.3f * 1024,
    s.vram_total_mb = 20.0f * 1024;
    s.replay_ms = 0.60f, s.gpu_ms = 1.28f, s.pipe_wait_ms = 0, s.xlate_ms = 0, s.tex_uploads = 0;
    p.version = "NX2navi 0.5.0, game 1.1.1";
    snprintf(p.picture[0], 64, "1920x1080");
    snprintf(p.picture[1], 64, "2560x1440");
    snprintf(p.picture[3], 64, "bilinear");
    snprintf(p.picture[4], 64, "60 FPS");
    snprintf(p.picture[5], 64, "VSync");
    p.scene_known = false, p.fg_fps = 60, p.under_menu = false;
    perf_panel_draw(&p, w, scale);
}

static void scene_boot(float w, float h, float scale, ImFont *title) {
    // the boot screen's box over the black the window shows then (menu.cpp menu_boot_frame), half way through the
    // pipelines
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), IM_COL32(0, 0, 0, 255));
    char step[160];
    snprintf(step, sizeof step, ui_text(UI_BOOT_PIPELINES), 1603LL, 3235LL);
    skin_boot_panel(ImGui::GetForegroundDrawList(), title, title->FontSize, ImGui::GetFont(), ImGui::GetFontSize(),
                    ui_text(UI_BOOT_TITLE), step, ui_text(UI_BOOT_HINT), 1603.0f / 3235.0f, 0, w, h, scale);
}

// ---- the rasteriser: every triangle of the draw lists, its colour times the font atlas, blended over the image

struct Image {
    int w, h;
    std::vector<float> px;   // straight RGBA, 0..1
};

// Twice the signed area of (e0, e1, p), computed with the edge's ends in one fixed order, so the two triangles that share
// an edge get exactly opposite values for a pixel and none is shaded twice or left out along it.
static float edge(ImVec2 e0, ImVec2 e1, float px, float py) {
    bool swap = e0.x > e1.x || (e0.x == e1.x && e0.y > e1.y);
    if (swap) {
        ImVec2 t = e0;
        e0 = e1, e1 = t;
    }
    float v = (e0.x - px) * (e1.y - py) - (e0.y - py) * (e1.x - px);
    return swap ? -v : v;
}

// The atlas at (u, v), bilinear, clamped to its edges; 0..1.
static void sample(const uint8_t *atlas, int aw, int ah, float u, float v, float out[4]) {
    float x = u * aw - 0.5f, y = v * ah - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    for (int k = 0; k < 4; k++) out[k] = 0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            int xx = x0 + i < 0 ? 0 : x0 + i >= aw ? aw - 1 : x0 + i, yy = y0 + j < 0 ? 0 : y0 + j >= ah ? ah - 1 : y0 + j;
            float wgt = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
            const uint8_t *t = atlas + 4 * ((size_t)yy * aw + xx);
            for (int k = 0; k < 4; k++) out[k] += wgt * t[k] / 255.0f;
        }
}

// The art at (u, v) at level of detail lod, trilinear between its levels, as the game's sampler reads it.
static void sample_art(const std::vector<Level> &levels, float u, float v, float lod, float out[4]) {
    const float top = (float)levels.size() - 1;
    lod = lod < 0 ? 0 : lod > top ? top : lod;
    int l0 = (int)floorf(lod), l1 = l0 + 1 < (int)levels.size() ? l0 + 1 : l0;
    float a[4], b[4], f = lod - l0;
    sample(levels[l0].px.data(), levels[l0].w, levels[l0].h, u, v, a);
    sample(levels[l1].px.data(), levels[l1].w, levels[l1].h, u, v, b);
    for (int k = 0; k < 4; k++) out[k] = a[k] + (b[k] - a[k]) * f;
}

static void raster(const ImDrawData *dd, Image &img, const uint8_t *atlas, int aw, int ah) {
    for (int l = 0; l < dd->CmdListsCount; l++) {
        const ImDrawList *dl = dd->CmdLists[l];
        const ImDrawVert *vb = dl->VtxBuffer.Data;
        const ImDrawIdx *ib = dl->IdxBuffer.Data;
        for (const ImDrawCmd &cmd : dl->CmdBuffer) {
            if (cmd.UserCallback) continue;
            int cx0 = (int)fmaxf(0, cmd.ClipRect.x), cy0 = (int)fmaxf(0, cmd.ClipRect.y);
            int cx1 = (int)fminf((float)img.w, cmd.ClipRect.z), cy1 = (int)fminf((float)img.h, cmd.ClipRect.w);
            const intptr_t tex_id = (intptr_t)cmd.GetTexID();
            const std::vector<Level> *art = tex_id >= 2 && tex_id < 2 + MENU_ART_COUNT ? &g_art[tex_id - 2] : nullptr;
            for (unsigned i = 0; i + 2 < cmd.ElemCount; i += 3) {
                const ImDrawVert &a = vb[cmd.VtxOffset + ib[cmd.IdxOffset + i]], &b = vb[cmd.VtxOffset + ib[cmd.IdxOffset + i + 1]],
                                 &c = vb[cmd.VtxOffset + ib[cmd.IdxOffset + i + 2]];
                float area = (b.pos.x - a.pos.x) * (c.pos.y - a.pos.y) - (b.pos.y - a.pos.y) * (c.pos.x - a.pos.x);
                if (fabsf(area) < 1e-6f) continue;
                // the top-left rule: a pixel centre on an edge two triangles share is shaded by one of them only
                // (edge k is the one opposite vertex k, run in the triangle's own winding)
                const ImVec2 *vtx[3] = {&a.pos, &b.pos, &c.pos};
                bool top_left[3];
                for (int k = 0; k < 3; k++) {
                    ImVec2 e0 = *vtx[(k + 1) % 3], e1 = *vtx[(k + 2) % 3];
                    float dx = (e1.x - e0.x) * (area > 0 ? 1 : -1), dy = (e1.y - e0.y) * (area > 0 ? 1 : -1);
                    top_left[k] = (dy == 0 && dx < 0) || dy > 0;
                }
                int x0 = (int)fmaxf((float)cx0, floorf(fminf(a.pos.x, fminf(b.pos.x, c.pos.x))));
                int x1 = (int)fminf((float)cx1, ceilf(fmaxf(a.pos.x, fmaxf(b.pos.x, c.pos.x))));
                int y0 = (int)fmaxf((float)cy0, floorf(fminf(a.pos.y, fminf(b.pos.y, c.pos.y))));
                int y1 = (int)fminf((float)cy1, ceilf(fmaxf(a.pos.y, fmaxf(b.pos.y, c.pos.y))));
                float lod = 0;
                if (art && !art->empty()) {   // texels per screen pixel along the steeper axis
                    const float tw = (float)(*art)[0].w, th = (float)(*art)[0].h;
                    const float e1x = b.pos.x - a.pos.x, e1y = b.pos.y - a.pos.y, e2x = c.pos.x - a.pos.x, e2y = c.pos.y - a.pos.y;
                    const float du1 = (b.uv.x - a.uv.x) * tw, dv1 = (b.uv.y - a.uv.y) * th, du2 = (c.uv.x - a.uv.x) * tw, dv2 = (c.uv.y - a.uv.y) * th;
                    const float dudx = (du1 * e2y - du2 * e1y) / area, dvdx = (dv1 * e2y - dv2 * e1y) / area;
                    const float dudy = (du2 * e1x - du1 * e2x) / area, dvdy = (dv2 * e1x - dv1 * e2x) / area;
                    const float rho = fmaxf(sqrtf(dudx * dudx + dvdx * dvdx), sqrtf(dudy * dudy + dvdy * dvdy));
                    lod = rho > 0 ? log2f(rho) : 0;
                }
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) {
                        float px = x + 0.5f, py = y + 0.5f;
                        float w0 = edge(b.pos, c.pos, px, py) / area, w1 = edge(c.pos, a.pos, px, py) / area,
                              w2 = edge(a.pos, b.pos, px, py) / area;
                        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                        if ((w0 == 0 && !top_left[0]) || (w1 == 0 && !top_left[1]) || (w2 == 0 && !top_left[2])) continue;
                        float u = w0 * a.uv.x + w1 * b.uv.x + w2 * c.uv.x, v = w0 * a.uv.y + w1 * b.uv.y + w2 * c.uv.y;
                        float col[4];
                        for (int k = 0; k < 4; k++) {
                            float ca = (a.col >> (8 * k)) & 255, cb = (b.col >> (8 * k)) & 255, cc = (c.col >> (8 * k)) & 255;
                            col[k] = (w0 * ca + w1 * cb + w2 * cc) / 255.0f;
                        }
                        float t[4];
                        if (art && !art->empty()) sample_art(*art, u, v, lod, t);
                        else sample(atlas, aw, ah, u, v, t);   // filtered, as the GPU samples the atlas's baked lines
                        for (int k = 0; k < 4; k++) col[k] *= t[k];
                        float *d = &img.px[4 * ((size_t)y * img.w + x)], sa = col[3];
                        for (int k = 0; k < 3; k++) d[k] = col[k] * sa + d[k] * (1 - sa);
                        d[3] = sa + d[3] * (1 - sa);
                    }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: %s <font.ttf> <menu|mp|text|hud|overlay|boot> <width> <height> <out.rgba> [art folder]\n", argv[0]);
        return 2;
    }
    const char *font = argv[1], *scene = argv[2], *out = argv[5];
    if (argc == 7) art_load(argv[6]);
    int w = atoi(argv[3]), h = atoi(argv[4]);
    float scale = fmaxf(fminf((float)h, w * 0.68f), 720.0f) / 1080.0f;   // as menu.cpp menu_scale at 100 % display scale
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2((float)w, (float)h);
    io.DeltaTime = 1.0f / 60;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.Fonts->AddFontFromFileTTF(font, 26.0f * scale);
    ImFont *title = io.Fonts->AddFontFromFileTTF(font, 31.0f * scale);
    skin_atlas_add(io.Fonts);
    skin_atlas_fill(io.Fonts);
    uint8_t *atlas;
    int aw, ah;
    io.Fonts->GetTexDataAsRGBA32(&atlas, &aw, &ah);
    io.Fonts->SetTexID((ImTextureID)1);
    ImGuiStyle &st = ImGui::GetStyle();
    skin_style(st);
    st.ScaleAllSizes(scale);
    ImGui::SetCurrentFont(io.Fonts->Fonts[0]);
    for (int frame = 0; frame < 4; frame++) {   // a few frames: sizes settle and the first row takes the cursor
        ImGui::NewFrame();
        if (!strcmp(scene, "menu")) scene_menu((float)w, (float)h, scale, title, frame == 0);
        else if (!strcmp(scene, "boot")) scene_boot((float)w, (float)h, scale, title);
        else if (!strcmp(scene, "mp")) scene_mp((float)w, (float)h, scale, title);
        else if (!strcmp(scene, "hud")) scene_hud((float)w, (float)h, scale);
        else if (!strcmp(scene, "text")) scene_text((float)w, (float)h, scale, title);
        else scene_overlay((float)w, (float)h, scale);
        ImGui::Render();
    }
    Image img{w, h, std::vector<float>((size_t)w * h * 4, 0.0f)};
    raster(ImGui::GetDrawData(), img, atlas, aw, ah);
    FILE *f = fopen(out, "wb");
    if (!f) return 1;
    std::vector<uint8_t> bytes((size_t)w * h * 4);
    for (size_t i = 0; i < bytes.size(); i++) bytes[i] = (uint8_t)(fminf(1, fmaxf(0, img.px[i])) * 255 + 0.5f);
    fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
    ImGui::DestroyContext();
    return 0;
}
