// Win32 adapter of the in-game menu: window messages to menu events (menu.h). The only Win32 code of the menu; the SDL3
// window has its own (ui/menu_sdl.cpp). Window thread.
//
// F1 opens and closes the menu, Esc opens it (with the menu open, Esc goes to the menu, which closes). Esc also releases
// the cursor (input_window_msg runs first; closing the menu gives the capture back). While the menu is open, its keys,
// the mouse and focus changes go to it; the game polls the keyboard on its own (input_win.c) and gets nothing while the
// menu is open (input_menu).
#include <windows.h>
#include <windowsx.h>

#include "imgui.h"

extern "C" {
#include "input.h"
#include "input_win.h"
#include "ui/menu.h"
}

static ImGuiKey key_of(WPARAM vk) {
    switch (vk) {
    case VK_UP: return ImGuiKey_UpArrow;
    case VK_DOWN: return ImGuiKey_DownArrow;
    case VK_LEFT: return ImGuiKey_LeftArrow;
    case VK_RIGHT: return ImGuiKey_RightArrow;
    case VK_RETURN: return ImGuiKey_Enter;
    case VK_SPACE: return ImGuiKey_Space;
    case VK_ESCAPE: return ImGuiKey_Escape;
    case VK_TAB: return ImGuiKey_Tab;
    case VK_BACK: return ImGuiKey_Backspace;
    case VK_DELETE: return ImGuiKey_Delete;   // clears a cell of the remapping rows
    case VK_HOME: return ImGuiKey_Home;
    case VK_END: return ImGuiKey_End;
    case VK_PRIOR: return ImGuiKey_PageUp;
    case VK_NEXT: return ImGuiKey_PageDown;
    case VK_SHIFT: return ImGuiKey_LeftShift;
    case VK_CONTROL: return ImGuiKey_LeftCtrl;
    case 'Q': return ImGuiKey_Q;   // previous tab
    case 'E': return ImGuiKey_E;   // next tab
    default: return ImGuiKey_None;
    }
}

// The mouse buttons pressed on the open menu and not yet released, one bit each. While any is held the window has the
// mouse capture, so the release reaches the menu where the pointer went outside the window (else ImGui would hold the
// button down and miss the next click). The game itself only clips the cursor (input_win.c), it takes no capture.
static unsigned g_held;

extern "C" int menu_window_msg(HWND h, UINT m, WPARAM w, LPARAM l, LRESULT *res) {
    switch (m) {
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        // bit 30: auto-repeat
        // Esc with the menu open (opened by F1 too) is not a toggle here: it goes to the menu, which closes itself
        if (!(l & (1 << 30)) && (w == VK_F1 || (w == VK_ESCAPE && !menu_is_open()))) {
            menu_toggle();
            *res = 0;
            return 1;
        }
        if (w == VK_F10) {   // also keeps F10 from opening the window menu
            if (!(l & (1 << 30))) menu_perf_cycle();
            *res = 0;
            return 1;
        }
        // a held Esc's repeats: the press that opened the menu is not queued, so its first repeat would close it
        if (w == VK_ESCAPE && (l & (1 << 30))) return 0;
        [[fallthrough]];
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (!menu_is_open()) return 0;
        if (ImGuiKey k = key_of(w)) menu_event_key(k, m == WM_KEYDOWN || m == WM_SYSKEYDOWN);
        return 0;   // F11 and the system keys keep working
    case WM_CHAR:   // a typed character (WM_SYSCHAR's Alt combinations are the window menu's)
        if (!menu_is_open()) return 0;
        menu_event_char((unsigned)w, 1);
        return 0;
    case WM_MOUSEMOVE:
        if (menu_is_open()) menu_event_mouse_pos((float)GET_X_LPARAM(l), (float)GET_Y_LPARAM(l));
        return 0;
    case WM_LBUTTONDOWN: case WM_LBUTTONUP:
    case WM_RBUTTONDOWN: case WM_RBUTTONUP:
    case WM_MBUTTONDOWN: case WM_MBUTTONUP: {
        int b = m == WM_LBUTTONDOWN || m == WM_LBUTTONUP ? 0 : m == WM_RBUTTONDOWN || m == WM_RBUTTONUP ? 1 : 2;
        int down = m == WM_LBUTTONDOWN || m == WM_RBUTTONDOWN || m == WM_MBUTTONDOWN;
        if (!down && (g_held & 1u << b)) {   // also when the menu closed meanwhile: the capture goes either way
            g_held &= ~(1u << b);
            if (!g_held && GetCapture() == h) ReleaseCapture();
        }
        if (!menu_is_open()) return 0;
        if (down) {
            if (!g_held) SetCapture(h);
            g_held |= 1u << b;
        }
        menu_event_mouse_pos((float)GET_X_LPARAM(l), (float)GET_Y_LPARAM(l));
        menu_event_mouse_button(b, down);
        return 0;
    }
    case WM_MOUSEWHEEL:
        if (menu_is_open()) menu_event_wheel(0, (float)GET_WHEEL_DELTA_WPARAM(w) / WHEEL_DELTA);
        return 0;
    case WM_CAPTURECHANGED:   // another window took the capture (ReleaseCapture above finds g_held empty)
        if ((HWND)l != h && g_held) {
            for (int b = 0; b < 3; b++)
                if (g_held & 1u << b) menu_event_mouse_button(b, 0);
            g_held = 0;
        }
        return 0;
    case WM_SETFOCUS:
    case WM_KILLFOCUS:
        menu_event_focus(m == WM_SETFOCUS);
        return 0;
    }
    return 0;
}
