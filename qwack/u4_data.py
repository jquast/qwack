import struct
import collections
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "dat")
FPATH_WORLD_MAP = os.path.join(DATA_DIR, "WORLD.MAP")
FPATH_SHAPES_VGA = os.path.join(DATA_DIR, "SHAPES.VGA")
FPATH_U4VGA_PAL = os.path.join(DATA_DIR, "U4VGA.pal")
NPC_MOVE_BEHAVIOR_MAPPING = {
    0x00: "fixed",
    0x01: "wander",
    0x80: "follow",
    0xFF: "attack",
}
from qwack import models


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
            # this reads as an array, probably shouldn't ..
            world_chunks[y, x] = [town_map_bytes[(32 * y) + x]]
    return world_chunks


def load_npcs_from_u4_ult_map(map_id: int, world_data: dict) -> list:
    """Returns NPCs as list of dictionaries compatible with Item."""
    if map_id not in ULT_FILENAME_MAPPING:
        return []
    fpath = os.path.join(DATA_DIR, ULT_FILENAME_MAPPING[map_id] + ".ULT")

    town_bytes = open(fpath, "rb").read()
    npcs = []
    for idx in range(32):
        tile1, x_pos1, y_pos1, _tile2, _x_pos2, _y_pos2, move, npc_id = [
            town_bytes[1024 + (32 * i) + idx] for i in range(8)
        ]
        if tile1 > 0:
            npcs.append(
                {
                    "tile_id": tile1,
                    "pos": (y_pos1, x_pos1),
                    "material": "flesh",
                    "npc_id": npc_id,
                    "move_behavior": NPC_MOVE_BEHAVIOR_MAPPING[move],
                    "land_passable": world_data['Shapes'][tile1].get("land_passable", False),
                    "sort_value": world_data['Shapes'][tile1].get("sort_value", 1),
                    "speed": world_data['Shapes'][tile1].get("speed", 1),
                    "composite": world_data['Shapes'][tile1].get("composite", True),
                }
            )
    return npcs


def read_u4_tlk(fpath) -> dict:
    talk_data = open(fpath, "rb").read()
    npcs = []
    for idx in range(16):
        char_data = talk_data[0x120 * idx : 0x120 * (idx + 1)]

        slot = [None] * 12
        start = 0x3
        for i in range(12):
            length = char_data[start:].find(b"\x00")
            assert length >= 0, char_data[start:]
            slot[i] = char_data[start : start + length].replace(b"\n", b" ").decode()
            start += length + 1
        name = slot[0]
        pronoun = slot[1]
        prompts = {
            "LOOK": slot[2],
            "JOB": slot[3],
            "HEAL": slot[4],
            slot[10].rstrip(): slot[5],
            slot[11].rstrip(): slot[6],
        }
        question = [slot[7], slot[8], slot[9]]
        flag_question = char_data[0]
        flag_humility = char_data[1]
        flag_turn_away = char_data[2]
        npcs.append(
            dict(
                name=name,
                pronoun=pronoun,
                prompts=prompts,
                question=question,
                flag_question={
                    0x03: "JOB",
                    0x04: "HEALTH",
                    0x05: slot[10].rstrip(),
                    0x06: slot[11].rstrip(),
                }.get(flag_question),
                flag_humility=flag_humility,
                flag_turn_away=flag_turn_away,
            )
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
    0x38: "LCB_2",
}
TLK_FILENAME_MAPPING = {
    0x01: "LCB",
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
    0x38: "LCB",
}


def read_map(map_id):
    # we treat the world map and town maps as the same API interface,
    # we aren't with the memory limitations of the original,
    # similar optimizations are made with viewport.small_world.
    if map_id == 0:
        return read_u4_world_chunks()
    fpath = os.path.join(DATA_DIR, f"{ULT_FILENAME_MAPPING[map_id]}.ULT")
    assert os.path.exists(fpath), fpath
    return read_u4_ult_map(fpath)


def read_npcs(map_id):
    # there are no NPC's on the world map (but there could be!)
    if map_id == 0:
        return []
    if map_id not in TLK_FILENAME_MAPPING:
        return []
    fpath = os.path.join(DATA_DIR, f"{TLK_FILENAME_MAPPING[map_id]}.TLK")
    assert os.path.exists(fpath), fpath
    return read_u4_tlk(fpath)


def load_items_from_disk(map_id: int, world_data):
    # map_id of '0' means world map, which has a chunk_size of 32x32,
    # we don't have concern for apple ][ memory restrictions and load
    # the entire world, anyway.
    result = {}
    chunk_size = 32 if map_id == 0 else 1
    if map_id and "map_data" in world_data["Maps"][map_id]:
        # read tileset from datafile
        map_chunks = {}
        for y, row in enumerate(world_data["Maps"][map_id]["map_data"]):
            for x, tile_id in enumerate(row):
                map_chunks[y, x] = [tile_id]
    else:
        # read tileset from u4 .ULT file
        map_chunks = read_map(map_id)
    for (chunk_y, chunk_x), chunk_data in map_chunks.items():
        for idx, raw_val in enumerate(chunk_data):
            div_y, div_x = divmod(idx, chunk_size)
            pos = models.Position(
                y=(chunk_y * chunk_size) + div_y,
                x=(chunk_x * chunk_size) + div_x,
            )
            tile_definition = world_data["Shapes"][raw_val]
            item = models.Item(
                tile_id=raw_val,
                pos=pos,
                sort_value=tile_definition.get("sort_value", 0),
                name=tile_definition.get("name", None),
                material=tile_definition.get("material", "construction"),
                darkness=tile_definition.get("darkness", 0),
                land_passable=tile_definition.get("land_passable", True),
                speed=tile_definition.get("speed", 0),
                brightness=tile_definition.get("brightness", 0),
                composite=tile_definition.get("composite", True),
            )
            # TODO: some items like chests, ladders, etc. could be layered
            result[(pos.y, pos.x)] = [item]
    return result


def create_npcs(map_id: int, world_data: dict) -> list[models.Item]:
    items = []
    # load NPCs from ULT and TLK files
    npc_talk_data = read_npcs(map_id)
    npc_definitions = load_npcs_from_u4_ult_map(map_id, world_data)
    for npc_definition in npc_definitions:
        npc_id = npc_definition.pop("npc_id")
        try:
            npc_definition["name"] = npc_talk_data[npc_id - 1]["name"]
            npc_definition["talk_data"] = npc_talk_data[npc_id - 1]
        except IndexError:
            npc_definition["name"] = "IDX.ERR#" + str(npc_id)
            npc_definition["talk_data"] = None
        items.append(models.Item(**npc_definition))
    # load NPCs from world_data
    for npc_definition in world_data["Maps"][map_id].get("npcs", []):
        items.append(models.Item(**npc_definition))
    return items
