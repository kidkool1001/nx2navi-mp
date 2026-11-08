"""Draw the in-game settings menu and the performance overlay without the game (tools/menu_preview.cpp).

    python tools/menu_preview.py [--out build/menu-preview] [--font FILE] [--size 1920x1080] [--romfs FOLDER]

Builds tools/menu_preview.cpp with the menu's own row, shape, multiplayer page, overlay and text code and Dear ImGui (host clang++, into
build/menu-preview), runs it for each scene and writes menu.png, mp.png (the Multiplayer tab), text.png (its text window), hud.png (the multiplayer overlay),
overlay.png (the performance overlay) and boot.png. Without --romfs no game
files are read: the window has its plain border in place of the game's ornamented frame and the boxes are plain
shapes. With --romfs, an extracted RomFS folder, the textures listed in runtime/ui/menu_art.cpp are decoded into --out
and the boxes look as they do in the game (needs what tools/ui_textures.py needs). The font given (default: DejaVu
Sans, else the first TrueType font found) stands in for the game's either way.
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / 'runtime'
SOURCES = [ROOT / 'tools' / 'menu_preview.cpp', RT / 'ui' / 'option_rows.cpp', RT / 'ui' / 'menu_skin.cpp',
           RT / 'ui' / 'mp_page.cpp', RT / 'ui' / 'perf_panel.cpp', RT / 'ui_text.c'] + \
          sorted((RT / 'third_party' / 'imgui').glob('imgui*.cpp'))
FONTS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf']


def png(path, w, h, rgba):
    """An 8-bit RGBA PNG of the raw pixels."""
    rows = b''.join(b'\0' + rgba[y * w * 4:(y + 1) * w * 4] for y in range(h))

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)) +
                           chunk(b'IDAT', zlib.compress(rows, 6)) + chunk(b'IEND', b''))


def build(out):
    cxx = shutil.which('clang++') or shutil.which('g++')
    if not cxx:
        sys.exit('needs clang++ or g++ on PATH')
    exe = out / ('menu_preview.exe' if os.name == 'nt' else 'menu_preview')
    objs = []
    for src in SOURCES:
        obj = out / (src.stem + '.o')
        if not obj.is_file() or obj.stat().st_mtime < max(src.stat().st_mtime, (RT / 'ui' / 'menu_skin.h').stat().st_mtime,
                                                          (RT / 'ui' / 'menu_art.h').stat().st_mtime):
            lang = ['-x', 'c', '-std=gnu11'] if src.suffix == '.c' else ['-x', 'c++', '-std=gnu++17']
            cmd = [cxx, *lang, '-c', '-O2', '-w', '-I', str(RT), '-I', str(RT / 'third_party' / 'imgui'), str(src), '-o',
                   str(obj)]
            if subprocess.run(cmd).returncode:
                sys.exit(f'{src.name} did not compile')
        objs.append(str(obj))
    if subprocess.run([cxx, *objs, '-o', str(exe), '-lm']).returncode:
        sys.exit('the link failed')
    return exe


def decode_art(romfs, out):
    """The textures of runtime/ui/menu_art.cpp kNames as out/art/NN.art for menu_preview.cpp; that folder."""
    sys.path.insert(0, str(ROOT / 'tools'))
    import ui_textures
    source = (RT / 'ui' / 'menu_art.cpp').read_text(encoding='utf-8')
    names = re.findall(r'"([^"]+)"', re.search(r'kNames\[MENU_ART_COUNT\] = \{(.*?)\};', source, re.S).group(1))
    data = ui_textures.Romfs(romfs).read(f'{ui_textures.ARCHIVES}/Common.Product.100.Nin_Ounce_NVN2.bntx.zs')
    textures = {t['name']: t for t in ui_textures.bntx_textures(data)}
    art = out / 'art'
    art.mkdir(parents=True, exist_ok=True)
    lib = ui_textures.astc_library(out)
    for i, name in enumerate(names):
        rgba = ui_textures.decode(textures[name], data, lib)
        (art / f'{i:02}.art').write_bytes(struct.pack('<II', rgba.shape[1], rgba.shape[0]) + rgba.tobytes())
    return art


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'menu-preview')
    ap.add_argument('--font', help='a TrueType font for the text')
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--romfs', type=Path, help="an extracted RomFS folder: draw the game's textures")
    a = ap.parse_args()
    font = a.font or next((f for f in FONTS if Path(f).is_file()), None)
    if not font:
        sys.exit('no font found: give one with --font')
    w, h = (int(x) for x in a.size.lower().split('x'))
    a.out.mkdir(parents=True, exist_ok=True)
    exe = build(a.out)
    art = [str(decode_art(a.romfs, a.out))] if a.romfs else []
    for scene in ('menu', 'mp', 'text', 'hud', 'overlay', 'boot'):
        raw = a.out / f'{scene}.rgba'
        if subprocess.run([str(exe), font, scene, str(w), str(h), str(raw), *art]).returncode:
            sys.exit(f'the {scene} scene failed')
        png(a.out / f'{scene}.png', w, h, raw.read_bytes())
        raw.unlink()
        print(a.out / f'{scene}.png')


if __name__ == '__main__':
    main()
