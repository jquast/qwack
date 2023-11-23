import struct
import collections
import os

FPATH_WORLD_MAP = os.path.join(os.path.dirname(__file__), "dat", "WORLD.MAP")
FPATH_SHAPES_VGA = os.path.join(os.path.dirname(__file__), "dat", "SHAPES.VGA")
FPATH_U4VGA_PAL = os.path.join(os.path.dirname(__file__), "dat", "U4VGA.pal")


def load_shapes_vga():
    shapes = []
    shape_bytes = open(FPATH_SHAPES_VGA, "rb").read()
    shape_pal = open(FPATH_U4VGA_PAL, "rb").read()

    chunk_dim = 16
    chunk_len = chunk_dim * chunk_dim
    for tile_idx in range(0, len(shape_bytes), chunk_len):
        shape = []
        for pixel_idx in range(chunk_len):
            idx = shape_bytes[tile_idx + pixel_idx]
            r = shape_pal[idx * 3] * 4
            g = shape_pal[(idx * 3) + 1] * 4
            b = shape_pal[(idx * 3) + 2] * 4
            shape.append((r, g, b))
        shapes.append(shape)
    return shapes


def read_u4_ult_map(fpath) -> dict[tuple[int, int], list[int]]:
    # Parse a U4 .ULT file, for only the 32x32 map at the starting 1024.
    # returns 'world_chunks' dictionary keyed by (y, x).
    #
    world_chunks = collections.defaultdict(list)
    town_map_bytes = open(fpath, "rb").read(1024)
    for y in range(32):
        for x in range(32):
            world_chunks[y, x] = [town_map_bytes[(32 * y) + x]]
    return world_chunks


def load_npcs_from_u4_ult_map(map_id: int, world_data: dict) -> list:
    """Returns NPCs as list of dictionaries compatible with Item."""
    fpath = os.path.join(
        os.path.dirname(__file__), "dat", ULT_FILENAME_MAPPING[map_id] + ".ULT"
    )

    town_bytes = open(fpath, "rb").read()
    npcs = []
    for idx in range(32):
        tile1, x_pos1, y_pos1, tile2, x_pos2, y_pos2, move, char_id = [
            town_bytes[1024 + (32 * i) + idx] for i in range(8)
        ]
        if tile1 > 0:
            npcs.append(
                {
                    "tile_id": tile1,
                    "pos": (y_pos1, x_pos1),
                    "name": f"char_id-{char_id}",
                    "material": "flesh",
                    "where": "floor",
                    "darkness": 1,
                    "land_passable": False,
                    "speed": 0,
                }
            )
    return npcs


def read_u4_world_chunks() -> dict[tuple[int, int], list[int]]:
    # read raw WORLD.DAT data as a dictionary keyed by (y, x) of 8x8 chunks
    # each value is a list of 32x32 tile bytes, keyed by their tileset id
    chunk_dim = 32
    chunk_len = chunk_dim * chunk_dim
    world_chunks = collections.defaultdict(list)
    with open(FPATH_WORLD_MAP, "rb") as fp:
        buf = bytearray(chunk_len)
        # map is sub-divded into 8x8 sectors
        for y in range(8):
            for x in range(8):
                # read all next 32x32 tiles of data into 'buf'
                n = fp.readinto(buf)
                assert n == chunk_len
                # for-each tile row,
                for j in range(chunk_dim):
                    chunk_line = []
                    # for-each tile column
                    for i in range(chunk_dim // 4):
                        o = j * chunk_dim + i * 4
                        # these 4 bytes make up the tiles, (tile_id, tile_id, tile_id, tile_id)
                        chunk_line.extend([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
                    world_chunks[y, x].extend(chunk_line)
    return world_chunks


ULT_FILENAME_MAPPING = {
    0x01: "LCB_1",
    0x02: "LYCAEUM",
    0x03: "EMPATH",
    0x04: "SERPENT",
    0x05: "MOONGLOW",
    0x06: "BRITAIN",
    0x07: "JHELOM",
    0x08: "YEW",
    0x09: "MINOC",
    0x0A: "TRINSIC",
    0x0B: "SKARA",
    0x0C: "MAGINCIA",
    0x0D: "PAWS",
    0x0E: "DEN",
    0x0F: "VESPER",
    0x10: "COVE",
}


def read_map(map_id):
    # we treat the world map and town maps as the same API interface,
    # we aren't with the memory limitations of the original,
    # similar optimizations are made with viewport.small_world.
    if map_id == 0:
        return read_u4_world_chunks()
    return read_u4_ult_map(
        os.path.join(
            os.path.dirname(__file__), "dat", f"{ULT_FILENAME_MAPPING[map_id]}.ULT"
        )
    )
