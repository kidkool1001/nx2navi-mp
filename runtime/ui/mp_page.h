// Multiplayer menu page (ui/mp_page.cpp): the settings of the session, the live roster and the LAN browser, the text
// fields for the player name and the host address, and the in-world overlay (nametags, off-screen arrows, beacons, the
// session mini-HUD).
#pragma once

// The tab's rows: inside the page's opt_begin / opt_end, as every page of the menu.
void tab_mp(float scale);
// After opt_end: the window that types the player name or the host address (the rows' text fields have no text entry of
// their own, as the remapping rows open a window for their press).
void mp_page_after(void);
// Such a window is open: B, Esc, L / R and the tab keys stay on the page (menu.cpp, as binds_modal).
bool mp_modal(void);
// The menu opened or closed: a window still open is dropped, unchanged.
void mp_cancel(void);
// The text window opened on a field, as its row does (tools/menu_preview.cpp draws it): 1 the player's name, 2 the host
// address.
void mp_open_field(int field);

// The in-world overlay while the menu is closed (menu.cpp menu_overlay_frame): a nametag with the player's name, distance
// and latency over each remote player, an arrow at the picture's edge for one off screen, the beacons, and the session's
// mini-HUD in the top left corner. pic: where the game's picture sits in the swapchain image, fractions of w, h (x, y,
// width, height); its width 0 means the whole image.
void mp_overlay_draw(float w, float h, float scale, const float pic[4]);
