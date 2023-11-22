import subprocess
import functools
import struct
import array
import zlib
import os
import io

# 3rd party
import PIL.Image

# import colorsys

CHAFA_BIN = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "chafa", "tools", "chafa", "chafa"
)
CHAFA_TRIM_START = len("\x1b[?25l\x1b[0m")
CHAFA_EXTRA_ARGS = ["-w", "1", "-O", "1"]


# todo: make a Shapes class, of course!
# Just init a new Shapes class for each tileset, and then call get_tile() on it.
def apply_darkness(image, darkness, max_darkness_level):
    # for each darkness level, create an increasingly darker mosaic of black
    # this takes into account these tilesets are doubled (16x16 -> 32x32), and
    # so it steps and stamps 2 pixels at a time.
    black = (0, 0, 0)
    for y in range(0, image.size[1], 2):
        y_even = y % 4
        for x in range(
            y_even,
            image.size[0] - 1,
            (5 - y_even) % (max_darkness_level - darkness) + 3,
        ):
            image.putpixel((x, y), black)
            image.putpixel((x, y + 1), black)
            image.putpixel((x + 1, y), black)
            image.putpixel((x + 1, y + 1), black)
    return image
        

def apply_over_background(bg_image, fg_image):
    # Convert to RGBA mode, set black pixels as alpha transprancy layer
    fg_image = fg_image.convert("RGBA")
    fg_image.putalpha(
        fg_image.split()[0].point(lambda p: 0 if p == 0 else 255).convert("L")
    )

    # convert background image to RGBA mode, merge with foreground image
    bg_image = bg_image.convert("RGBA")
    bg_image.alpha_composite(fg_image)
    return bg_image



#@functools.lru_cache(maxsize=256 * 16)
def make_tile(
    shape_data,
    tile_id,
    tile_width,
    tile_height,
    # effects,
    darkness=0,
    x_offset=0,
    y_offset=0,
    bg_tile_id=None,
    max_darkness_level=1,   # todo repeated in main.py
):
    # special tile_id -1 is "Void" and cannot display, or so dark
    # that it cannot be displayed, fast black tile in any case
    if tile_id == -1 or darkness >= max_darkness_level:
        return [" " * tile_width] * tile_height

    bg_image = None
    applied_image = None
    fg_image = make_image_from_pixels(shape_data[tile_id])
    if bg_tile_id is not None:
        bg_image = make_image_from_pixels(shape_data[bg_tile_id])

    if darkness:
        # apply darkness to both layers
        fg_image = apply_darkness(fg_image, darkness, max_darkness_level)
        if bg_image:
            bg_image = apply_darkness(bg_image, darkness, max_darkness_level)
    if y_offset or x_offset:
        # apply y & x offsets to the background tile or only visible tile
        if bg_image is not None:
            bg_image = apply_offsets(x_offset, y_offset, bg_image)
        else:
            fg_image = apply_offsets(x_offset, y_offset, fg_image)
    if bg_image is not None and fg_image:
        # apply fg over background image
        ref_image = apply_over_background(bg_image, fg_image)
        img_byte_arr = io.BytesIO()
        ref_image.save(img_byte_arr, format="PNG")
    else:
        ref_image = fg_image or bg_image

    img_byte_arr = io.BytesIO()
    ref_image.save(img_byte_arr, format="PNG")

    chafa_cmd_args = [
        CHAFA_BIN,
        *CHAFA_EXTRA_ARGS,
        "--size",
        f"{tile_width}x{tile_width}",
        "-",
    ]
    ans = subprocess.check_output(chafa_cmd_args, input=img_byte_arr.getvalue()).decode()
    lines = ans.splitlines()

    # remove preceeding and trailing hide/show cursor attributes
    return_list = [lines[0][CHAFA_TRIM_START:]] + lines[1:-1]
    return return_list

def apply_offsets(x_offset, y_offset, ref_image):
    tmp_img = PIL.Image.new(ref_image.mode, ref_image.size)

    # Loop through each pixel and shift themn by given offset
    for y in range(tmp_img.size[1]):
        for x in range(tmp_img.size[0]):
            new_x = (x + x_offset) % tmp_img.size[0]
            new_y = (y + y_offset) % tmp_img.size[1]
            tmp_img.putpixel((new_x, new_y), ref_image.getpixel((x, y)))
    return tmp_img


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


def load_shapes_ega():
    shapes = []
    shape_bytes = open("ULT/SHAPES.EGA", "rb").read()

    for i in range(256):
        shape = []
        for j in range(16):
            for k in range(8):
                d = shape_bytes[k + 8 * j + 128 * i]
                a, b = divmod(d, 16)
                shape.append(EGA2RGB[a])
                shape.append(EGA2RGB[b])
        shapes.append(shape)
    return shapes


def load_shapes_vga():
    # loads the VGA set, from http://www.moongates.com/u4/upgrade/files/u4upgrad.zip
    # or, from https://github.com/jahshuwaa/u4graphics
    shapes = []
    shape_bytes = open(
        os.path.join(os.path.dirname(__file__), "dat", "SHAPES.VGA"), "rb"
    ).read()
    shape_pal = open(
        os.path.join(os.path.dirname(__file__), "dat", "U4VGA.pal"), "rb"
    ).read()
    for tile_idx in range(0, len(shape_bytes), 16 * 16):
        shape = []
        for pixel_idx in range(16 * 16):
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
    height = width = (16 if len(pixels) == (16 * 16) else 32 if len(pixels) == (32 * 32) else -1)
    assert -1 not in (height, width), f"Invalid pixel count, cannot determine HxW: {len(pixels)}"
    pbdata = make_png_bytes(width, height, pixels)
    return PIL.Image.open(io.BytesIO(pbdata))


# if __name__ == "__main__":
#    from png import write_png
#
#    shapes = load_shapes_vga()
#    scale_factor = 4
#    for tile_num in range(len(shapes)):
#        print(shapes[tile_num])
#        image = scale(shapes[tile_num], scale_factor)
#        print(image)
#        #write_png(
#            f"images/tile_{tile_num:02X}.png",
#            16 * scale_factor,
#            16 * scale_factor,
#            image,
#        )
#
