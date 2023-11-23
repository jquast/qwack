import contextlib
import subprocess
import functools
import struct
import array
import zlib
import os
import io
import timeit

# 3rd party
import PIL.Image

# import colorsys

MAX_DARKNESS_LEVEL = 4
MIN_TILE_SIZE = 2
MAX_TILE_SIZE = 16

CHAFA_BIN = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "chafa", "tools", "chafa", "chafa"
)
CHAFA_TRIM_START = len("\x1b[?25l\x1b[0m")
# very strange, higher optimization levels (-O 3 or greater) add aliasing, which causes blurring
# which ends up with images 50% darker, so -O 1 is used !
CHAFA_EXTRA_ARGS = ["-w", "9", "-O", "1"]


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


# @functools.lru_cache(maxsize=256 * 16)
def get_ansi_txt_tile(
    shape_data,
    tile_id,
    tile_width,
    tile_height,
    # effects,
    darkness=0,
#    x_offset=0,
#    y_offset=0,
#    inverse=False,
#    bg_tile_id=None,
#    max_darkness_level=1,  # todo repeated in main.py
):
    if tile_id == -1 or darkness >= MAX_DARKNESS_LEVEL:
        # special tile_id -1 is "Void" and cannot display, or so dark
        # that it cannot be displayed, generate a fast black tile in any case,
        # this could be a higher level but maybe we could do something
        # interesting for fully black tile, especially with 'effects', like
        # inverse.
        return ["\x1b[0m" + (" " * tile_width)] * tile_height
    return shape_data[tile_id][darkness]["value"]
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


def load_shapes_ega(filename):
    # from https://github.com/jtauber/ultima4 a bit outdated project
    shapes = []
    shape_bytes = open(
        os.path.join(os.path.dirname(__file__), "dat", filename), "rb"
    ).read()
    ega2rgb = [
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

    for i in range(256):
        shape = []
        for j in range(16):
            for k in range(8):
                d = shape_bytes[k + 8 * j + 128 * i]
                a, b = divmod(d, 16)
                shape.append(ega2rgb[a])
                shape.append(ega2rgb[b])
        shapes.append(shape)
    return shapes


def load_shapes_vga(filename):
    # loads the VGA set, from http://www.moongates.com/u4/upgrade/files/u4upgrad.zip
    # or, from https://github.com/jahshuwaa/u4graphics
    shapes = []
    shape_bytes = open(
        os.path.join(os.path.dirname(__file__), "dat", filename), "rb"
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
    height = width = (
        16 if len(pixels) == (16 * 16) else 32 if len(pixels) == (32 * 32) else -1
    )
    assert -1 not in (
        height,
        width,
    ), f"Invalid pixel count, cannot determine HxW: {len(pixels)}"
    pbdata = make_png_bytes(width, height, pixels)
    return PIL.Image.open(io.BytesIO(pbdata))



# --- offline functions, only, run once to create qwack/dat/sf*.zip files ---

@contextlib.contextmanager
def elapsed_timer():
    """Timer pattern, from https://stackoverflow.com/a/30024601."""
    start = timeit.default_timer()

    def elapser():
        return timeit.default_timer() - start

    # pylint: disable=unnecessary-lambda
    yield lambda: elapser()


# Define the function to be executed in parallel
def process_shape_file(sf):
    load_fn = load_shapes_ega if sf["mode"].upper() == "EGA" else load_shapes_vga
    shape_data = load_fn(sf["filename"])
    # each zip file for each shapefile character set contains
    # yaml files as {tile_size}.yaml, containing a dictionary
    # keyed by tile_id (0-255), with a value of a list of strings,
    # the result of make_ansi_txt_tile in all permutations of darkness.
    zip_filepath = os.path.join(os.path.dirname(__file__), 'dat', f'{sf["name"]}.zip')
    import zipfile
    import yaml

    result = {}
    if os.path.exists(zip_filepath):
        os.remove(zip_filepath)
    print(f'BEGIN {zip_filepath}')
    zip_file = zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED, compresslevel=1)

    min_tile_size = 2
    max_tile_size = 16
    with elapsed_timer() as elapsed_sf:
        for tile_size in range(min_tile_size, max_tile_size):
            with elapsed_timer() as elapsed_tile:
                for tile_id in range(len(shape_data)):
                    # do not generate "darkest" color, 
                    darkness_levels = range(0, MAX_DARKNESS_LEVEL)
                    result[tile_id] = [
                        {
                            "darkness": darkness,
                            "value": make_ansi_txt_tile(
                                shape_data, tile_id, tile_size, tile_size, darkness=darkness, max_darkness_level=MAX_DARKNESS_LEVEL + 1)
                        }
                        for darkness in darkness_levels
                    ]
                zip_file.writestr(f"{tile_size}.yaml", yaml.dump(result))
            print(f"Completed {sf["mode"]} mod from {sf["filename"]}, libchafa tile_size {tile_size} in {elapsed_tile():.2f} seconds.")
        zip_file.close()
    print(f'Completed {sf["mode"]} ShapeFile {"filename"} in {elapsed_sf():.2f} seconds.')
    print(f'END {zip_filepath}')


def make_all_tiles():
    import multiprocessing, yaml
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    FPATH_WORLD_YAML = os.path.join(os.path.dirname(__file__), "dat", "world.yaml")
    world_data = yaml.load(open(FPATH_WORLD_YAML, "r"), Loader=yaml.SafeLoader)

    # Apply the function to each shape file using multiple processes
    pool.map(process_shape_file, world_data["ShapeFiles"])

    # Close the pool of processes
    pool.close()
    pool.join()

if __name__ == "__main__":
    make_all_tiles()
    #
    # TODO: will re-add y_offset animation, and fg_image & bg_image
    # combinations, later!  we just need to define which tiles are known to be
    # "floor" tiles, and just permutate all creatures above them all, this
    # greatly reduces the possible 255 * 255 space to something like estimated
    # 32 * 32 (64x), this still allows us to have "darkness" *and* fg_image on
    # bg_image, *and* y_offset animation for those few water tiles that support
    # it!
    # TODO: inverse!