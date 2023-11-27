import math
import time
import random
import contextlib
import subprocess
import functools
import struct
import array
import zlib
import os
import io
import timeit
import zipfile

import PIL.Image

# import colorsys

MIN_DARKNESS = 0
MAX_DARKNESS = 5
TILE_SIZES = (1, 2, 3, 6, 8, 12, 16)
MIN_TILE_SIZE = min(TILE_SIZES)
MAX_TILE_SIZE = max(TILE_SIZES)
CHAR_SIZES = (6, 7, 8)
MIN_CHAR_SIZE = min(CHAR_SIZES)
MAX_CHAR_SIZE = max(CHAR_SIZES)
EGA2RGB = [
    (0x00, 0x00, 0x00),
    (0x00, 0x00, 0xAA),
    (0x00, 0xAA, 0x00),
    (0x00, 0xAA, 0xAA),
    (0xAA, 0x00, 0x00),
    (0xAA, 0x00, 0xAA),
    (0xAA, 0x55, 0x00),
    (0xAA, 0xAA, 0xAA),
    (0x55, 0x55, 0x55),
    (0x55, 0x55, 0xFF),
    (0x55, 0xFF, 0x55),
    (0x55, 0xFF, 0xFF),
    (0xFF, 0x55, 0x55),
    (0xFF, 0x55, 0xFF),
    (0xFF, 0xFF, 0x55),
    (0xFF, 0xFF, 0xFF),
]


# XXX Chafa is great, but we need to pre-render *everything* so that we can avoid it
# as a dependency if we want to distribute this with 'pip' ..
CHAFA_BIN = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "chafa", "tools", "chafa", "chafa"
)
CHAFA_TRIM_START = len("\x1b[?25l\x1b[0m")
CHAFA_EXTRA_ARGS = ["-w", "1", "-O", "1", "--font-ratio=9/16"]

TILESET_CACHE_ZIP = os.path.join(os.path.dirname(__file__), "tileset_cache.zip")
TILESET_FP = zipfile.ZipFile(TILESET_CACHE_ZIP, 'a')

# todo: make a Shapes class, of course!
# Just init a new Shapes class for each tileset, and then call get_tile() on it.
def apply_darkness(image: PIL.Image, darkness: int):
    if darkness == 0 or image is None:
        return image
    tmp_image = PIL.Image.new(image.mode, image.size)
    for y in range(image.size[1]):
        y_even = y % 2 == 0
        for x in range(image.size[0]):
            x_even = x % 2 == 0
            x2 = x + x_even
            if darkness == 1 and y_even and x2 % 4 == 0:
                tmp_image.putpixel((x, y), (0, 0, 0))

            elif darkness == 2 and y_even and x2 % 3 == 0:
                tmp_image.putpixel((x, y), (0, 0, 0))
            elif darkness == 2 and not y_even and x2 % 4 == 0:
                tmp_image.putpixel((x, y), (0, 0, 0))

            elif darkness == 3 and y_even and x2 % 2 == 0:
                tmp_image.putpixel((x, y), (0, 0, 0))
            elif darkness == 3 and not y_even and x2 % 3 == 0:
                tmp_image.putpixel((x, y), (0, 0, 0))

            elif darkness == 4 and y_even and x2 % 3:
                tmp_image.putpixel((x, y), (0, 0, 0))
            elif darkness == 4 and not y_even and x2 % 2:
                tmp_image.putpixel((x, y), (0, 0, 0))

            elif darkness == 5 and y_even and x2 % 4:
                tmp_image.putpixel((x, y), (0, 0, 0))
            elif darkness == 5 and not y_even and x2 % 3:
                tmp_image.putpixel((x, y), (0, 0, 0))

            elif darkness == 5 and y_even and x2 % 6:
                tmp_image.putpixel((x, y), (0, 0, 0))
            elif darkness == 5 and not y_even and x2 % 4:
                tmp_image.putpixel((x, y), (0, 0, 0))

            else:
                tmp_image.putpixel((x, y), image.getpixel((x, y)))
    return tmp_image

def apply_offsets(ref_image, x_offset, y_offset):
    if ref_image and (x_offset or y_offset):
        tmp_img = PIL.Image.new(ref_image.mode, ref_image.size)
        for y in range(tmp_img.size[1]):
            new_y = (y + y_offset) % tmp_img.size[1]
            for x in range(tmp_img.size[0]):
                new_x = (x + x_offset) % tmp_img.size[0]
                tmp_img.putpixel((new_x, new_y), ref_image.getpixel((x, y)))
        return tmp_img
    return ref_image

def apply_inverse(ref_image):
    if ref_image:
        # sets all pixels to their inverted color (spell effects)
        tmp_image = PIL.Image.new(ref_image.mode, ref_image.size)
        for y in range(ref_image.size[1]):
            for x in range(ref_image.size[0]):
                r, g, b, a = ref_image.getpixel((x, y))
                tmp_image.putpixel((x, y), (255 - r, 255 - g, 255 - b, a))
        return tmp_image
    return ref_image

def apply_composite(bg_image, fg_image):
    # Convert to RGBA mode, set black pixels as alpha transprancy layer
    # tmp_fg_image = PIL.Image.new(fg_image.mode, fg_image.size)
    # copy fg_image to tmp_fg_image
    tmp_fg_image = fg_image.copy()
    # TODO try putting this higher level

    # TODO: apply a kind of border mask, is that posssible, to "grow" by 1 pixel
    # any mask, to ensure black pixels betwixt? esp. if we double our tile
    # resolution, but add 1 pixel of alpha, it should give a more distinct border
    tmp_fg_image.putalpha(
        tmp_fg_image.split()[0].point(lambda p: 0 if p == 0 else 255).convert("L")
    )

    # convert background image to RGBA mode, merge with foreground image
    tmp_bg_image = bg_image.copy()
    tmp_bg_image.alpha_composite(tmp_fg_image)
    return tmp_bg_image


class TileService:
    def __init__(self, Tilesets, tile_filename, tile_size, Charsets, char_filename, char_size):
        self.Tilesets = Tilesets
        self.Charsets = Charsets
        self.tile_data = self.init_tiles(tile_filename, tile_size)
        self.char_data = self.init_chars(char_filename, char_size)

    def init_tiles(self, tile_filename, tile_size):
        for ts in self.Tilesets:
            if ts["filename"] == tile_filename:
                tile_size = min(max(tile_size, MIN_TILE_SIZE), MAX_TILE_SIZE)
                # update tile_height by rendering to ansi and measuring the result height
                tile_data = load_tileset(tileset_record=ts)
                self._make_ansi_tile.cache_clear()
                tile_ff_ansi_txt = self.make_ansi_text_from_image(
                    ref_image=make_image_from_pixels(pixels=tile_data[0xFF]),
                    tile_width=tile_size, tile_height=max(1, tile_size))
                self.tile_size = tile_size
                self.tile_width = tile_size
                self.tile_height = len(tile_ff_ansi_txt)
                self.tile_filename = tile_filename
                return tile_data
        raise ValueError(f"No matching records by filename={self.tile_filename!r}, Tilesets={self.Tilesets}")

    def init_chars(self, char_filename, char_size):
        for cs in self.Charsets:
            if cs["filename"] == char_filename:
                char_size = min(max(char_size, MIN_CHAR_SIZE), MAX_CHAR_SIZE)
                # update tile_height by rendering to ansi and measuring the result height
                char_data = load_charset(charset_record=cs)
                self._make_ansi_tile.cache_clear()
                tile_03_ansi_txt = self.make_ansi_text_from_image(
                    ref_image=make_image_from_pixels(pixels=char_data[0x03]),
                    tile_width=char_size, tile_height=max(1, char_size - 1))
                self.char_size = char_size
                self.char_width = char_size
                self.char_height = len(tile_03_ansi_txt)
                self.char_filename = char_filename
                return char_data
        raise ValueError(f"No matching records by filename={self.char_filename!r}, Charsets={self.Charsets}")

    def cycle_tileset(self):
        # given the current self.tile_filename, cycle to the next
        by_filename = {ts["filename"]: ts for ts in self.Tilesets}
        idx = self.Tilesets.index(by_filename[self.tile_filename])
        idx += 1
        if idx == len(self.Tilesets):
            idx = 0
        next_tile_filename = self.Tilesets[idx]["filename"]
        self.tile_data = self.init_tiles(next_tile_filename, self.tile_size)

    def cycle_charset(self):
        # given the current self.char_filename, cycle to the next
        by_filename = {cs["filename"]: cs for cs in self.Charsets}
        idx = self.Charsets.index(by_filename[self.char_filename])
        idx += 1
        if idx == len(self.Charsets):
            idx = 0
        next_char_filename = self.Charsets[idx]["filename"]
        self.char_data = self.init_chars(next_char_filename, self.char_size)

    def darkness(self, item, player_pos, world_darkness):
        # TODO: we should be able to cast light from fire pits,
        #       fireballs etc. But that would mean some kind of
        #       high-level "scanner" of items in our viewport,
        #       first to apply darkness layer depending on
        #       world_darkness,
        if item.brightness:
            return 0
        distance = item.distance(player_pos)
        # 1/24 chance of rounding error of distance provides "candlelight effect"
        fn_trim = math.ceil if not random.randrange(24) else math.floor
        return fn_trim(min(max(0, (world_darkness - 2) + distance), MAX_DARKNESS))

    def make_character_tile(self, character: str, darkness=0, x_offset=0, y_offset=0, inverse=False):
        return self._make_ansi_tile(tile_id=ord(character),
                                    bg_tile_id=None,
                                    tile_filename=self.char_filename,
                                    tile_width=self.char_width,
                                    tile_height=self.char_height,
                                    tile_darkness=darkness,
                                    x_offset_fg=x_offset,
                                    y_offset_fg=y_offset,
                                    inverse=inverse,
                                    data_source='char')

    def make_ansi_tile(self, items, player_pos, darkness):
        # TODO: move all this logic into an ItemCollection.render_ansi()
        # and Maps should be [Y,X] array of ItemCollections
        if len(items) < 1 or any(i.is_void for i in items) or darkness > MAX_DARKNESS:
            # speedy performance hack for blank tiles
            return [f"\x1b[0m{(" " * self.tile_width)}"] * self.tile_height

        # it would be better if items were always in sorted order !!
        items.sort(key=lambda i: i.sort_value, reverse=True)
        tile_id = items[0].animation_tile_id(tick=int(time.monotonic() * 4))
        bg_tile_id = None
        if items[0].composite and len(items) > 1:
            bg_tile_id = items[-1].tile_id
        y_offset_fg, y_offset_bg = 0, 0
        if items[0].is_field:
            y_offset_fg = items[0].get_y_offset(tick=int(time.monotonic() * 4))
        if items[-1].is_water:
            if bg_tile_id is not None:
                y_offset_bg = items[-1].get_y_offset(tick=int(time.monotonic() * 4))
            else:
                y_offset_fg = items[-1].get_y_offset(tick=int(time.monotonic() * 4))

        # TODO: may require 'ui', or maybe just party.effects,
        # and work.tick/time_of_dat for game state, to implement,
        # - darkness, inverse, hallucinations, poison, confusion
        tile_darkness = self.darkness(item=items[-1], player_pos=player_pos, world_darkness=darkness)
        # TODO: allow many layers (player over boat over water), this would allow better attacks, like
        # a bright fireball over creature over animated water, for example.
        return self._make_ansi_tile(tile_id=tile_id,
                                   bg_tile_id=bg_tile_id,
                                   tile_filename=self.tile_filename,
                                   tile_width=self.tile_width,
                                   tile_height=self.tile_height,
                                   tile_darkness=tile_darkness,
                                   y_offset_bg=y_offset_bg,
                                   y_offset_fg=y_offset_fg,
                                   inverse=False)


    @functools.lru_cache(maxsize=256)
    def get_pixel_cache(self, tile_filename, tile_id, data_source):
        assert data_source in ('tile', 'char')
        if tile_id is not None:
            pixel_storage = self.tile_data if data_source == 'tile' else self.char_data
            return make_image_from_pixels(pixels=pixel_storage[tile_id])

    @functools.lru_cache(maxsize=1024)
    def _make_ansi_tile(self, tile_id, bg_tile_id, tile_filename, tile_width, tile_height,
                              tile_darkness=0, x_offset_bg=0, y_offset_bg=0,
                              x_offset_fg=0, y_offset_fg=0, inverse=False, data_source='tile'):
        val = self.load_disk_cache(tile_id, bg_tile_id, tile_filename, tile_width, tile_height,
                                       tile_darkness, x_offset_bg, y_offset_bg,
                                       x_offset_fg, y_offset_fg, inverse)
        if val is not None:
            return val
        fg_image = self.get_pixel_cache(tile_filename, tile_id, data_source)
        bg_image = self.get_pixel_cache(tile_filename, bg_tile_id, data_source)
        # apply darkness to both layers
        bg_image = apply_darkness(bg_image, tile_darkness)
        fg_image = apply_darkness(fg_image, tile_darkness)
        # apply y & x offsets
        bg_image = apply_offsets(bg_image, x_offset_bg, y_offset_bg)
        fg_image = apply_offsets(fg_image, x_offset_fg, y_offset_fg)
        ref_image = fg_image
        if bg_image and fg_image:
            # apply fg over background image
            ref_image = apply_composite(bg_image, fg_image)
        if inverse:
            ref_image = apply_inverse(ref_image)
        ref_image = ref_image
        val = self.make_ansi_text_from_image(ref_image, tile_width, tile_height)
        self.save_disk_cache(tile_id, bg_tile_id, tile_filename, tile_width, tile_height,
                             tile_darkness, x_offset_bg, y_offset_bg,
                             x_offset_fg, y_offset_fg, inverse, val)
        return val

    def load_disk_cache(self, tile_id, bg_tile_id, tile_filename, tile_width, tile_height,
                        tile_darkness, x_offset_bg, y_offset_bg,
                        x_offset_fg, y_offset_fg, inverse):
        diskcache_fname = os.path.join(
            f'{tile_filename}',
            f'{tile_id}',
            f'{bg_tile_id}_{tile_width}_{tile_height}_{tile_darkness}_{x_offset_bg}_{y_offset_bg}_{x_offset_fg}_{y_offset_fg}_{inverse}.txt')
        global TILESET_FP
        if TILESET_FP.mode != 'r':
            TILESET_FP.close()
            TILESET_FP = zipfile.ZipFile(TILESET_CACHE_ZIP, 'r')
        try:
            return TILESET_FP.read(diskcache_fname).decode().split('\n')
        except KeyError:
            return None

    def save_disk_cache(self, tile_id, bg_tile_id, tile_filename, tile_width, tile_height,
                        tile_darkness, x_offset_bg, y_offset_bg,
                        x_offset_fg, y_offset_fg, inverse, val):
        global TILESET_FP
        if TILESET_FP.mode != 'a':
            TILESET_FP.close()
            TILESET_FP = zipfile.ZipFile(TILESET_CACHE_ZIP, 'a')
        diskcache_fname = os.path.join(
            f'{tile_filename}',
            f'{tile_id}',
            f'{bg_tile_id}_{tile_width}_{tile_height}_{tile_darkness}_{x_offset_bg}_{y_offset_bg}_{x_offset_fg}_{y_offset_fg}_{inverse}.txt')
        TILESET_FP.writestr(diskcache_fname, '\n'.join(val))

    @staticmethod
    def make_ansi_text_from_image(ref_image, tile_width, tile_height):
        img_byte_arr = io.BytesIO()
        ref_image.save(img_byte_arr, format="PNG")

        chafa_cmd_args = [
            CHAFA_BIN,
            *CHAFA_EXTRA_ARGS,
            "--size",
            f"{tile_width}x{tile_height + 1}",
            "-",
        ]
        ans = subprocess.check_output(
            chafa_cmd_args, input=img_byte_arr.getvalue()
        ).decode()
        lines = ans.splitlines()

        # remove preceeding and trailing hide/show cursor attributes
        return_list = [lines[0][CHAFA_TRIM_START:]] + lines[1:-1]
        return return_list

def scale(tile: list[tuple], scale_factor):
    # given 'tile' as an array of 16 by 16 pixels, return a new array
    # of width*scale_factor by height width * scale_factor pixels,
    # with each pixel repeated as necessary to fill
    result = []
    height, width = 16, 16
    for y in range(height):
        for _ in range(scale_factor):
            row = []
            for x in range(width):
                row.extend([tile[(y * width) + x]] * scale_factor)
            result.extend(row)
    return result


def load_shapes_ega(filename, width, height):
    # from https://github.com/jtauber/ultima4 a bit outdated project
    shapes = []
    shape_bytes = open(
        os.path.join(os.path.dirname(__file__), "dat", filename), "rb"
    ).read()
    for i in range(256):
        shape = []
        for j in range(width):
            for k in range(height):
                d = shape_bytes[k + height * j + (width * height) * i]
                a, b = divmod(d, 16)
                shape.append(EGA2RGB[a])
                shape.append(EGA2RGB[b])
        shapes.append(shape)
    return shapes


def load_shapes_vga(filename, width, height):
    # loads the VGA set, from http://www.moongates.com/u4/upgrade/files/u4upgrad.zip
    # or, from https://github.com/jahshuwaa/u4graphics
    shapes = []
    shape_bytes = open(
        os.path.join(os.path.dirname(__file__), "dat", filename), "rb"
    ).read()
    shape_pal = open(
        os.path.join(os.path.dirname(__file__), "dat", "U4VGA.pal"), "rb"
    ).read()
    for tile_idx in range(0, len(shape_bytes), width * height):
        shape = []
        for pixel_idx in range(width * height):
            idx = shape_bytes[tile_idx + pixel_idx]
            r = shape_pal[idx * 3] * 4
            g = shape_pal[(idx * 3) + 1] * 4
            b = shape_pal[(idx * 3) + 2] * 4
            shape.append((r, g, b))
        shapes.append(shape)
    return shapes


def output_chunk(out, chunk_type, data):
    out.write(struct.pack("!I", len(data)))
    out.write(bytes(chunk_type, "utf-8"))
    out.write(data)
    checksum = zlib.crc32(data, zlib.crc32(bytes(chunk_type, "utf-8")))
    out.write(struct.pack("!I", checksum))


def get_data(width, height, pixels):
    compressor = zlib.compressobj()
    data = array.array("B")
    for y in range(height):
        data.append(0)
        for x in range(width):
            data.extend(pixels[y * width + x])
    compressed = compressor.compress(data.tobytes())
    flushed = compressor.flush()
    return compressed + flushed


def make_png_bytes(width, height, pixels):
    out = io.BytesIO()
    out.write(struct.pack("8B", 137, 80, 78, 71, 13, 10, 26, 10))
    output_chunk(out, "IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
    output_chunk(out, "IDAT", get_data(width, height, pixels))
    output_chunk(out, "IEND", b"")
    return out.getvalue()


def make_image_from_pixels(pixels):
    if pixels:
        import PIL.Image

        height = width = (
            8 if len(pixels) == (8 * 8) else
            16 if len(pixels) == (16 * 16) else
            32 if len(pixels) == (32 * 32) else -1
        )
        assert -1 not in (
            height,
            width,
        ), f"Invalid pixel count, cannot determine HxW: {len(pixels)}"
        pbdata = make_png_bytes(width, height, pixels)
        image = PIL.Image.open(io.BytesIO(pbdata))
        return image.convert("RGBA")


def load_tileset(tileset_record):
    if tileset_record["mode"].upper() == "EGA":
        load_fn = functools.partial(load_shapes_ega, height=16, width=8)
    else:
        assert tileset_record["mode"].upper() == "VGA"
        load_fn = functools.partial(load_shapes_vga, height=16, width=16)
    return load_fn(tileset_record["filename"])


def load_charset(charset_record):
    if charset_record["mode"].upper() == "EGA":
        load_fn = functools.partial(load_shapes_ega, height=8, width=4)
    else:
        assert charset_record["mode"].upper() == "VGA"
        load_fn = functools.partial(load_shapes_vga, height=8, width=8)
    return load_fn(charset_record["filename"])