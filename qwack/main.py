#!/usr/bin/env python
import collections
import contextlib
import textwrap
import functools
import enum
import timeit
import subprocess
import random
import math
import io
import os

# 3rd party
import blessed
import yaml
import PIL.Image

echo = functools.partial(print, end="")
Position = collections.namedtuple("Position", ("z", "y", "x"))

FPATH_WORLD_MAP = os.path.join(os.path.dirname(__file__), "WORLD.MAP")
FPATH_WORLD_YAML = os.path.join(os.path.dirname(__file__), "world.yaml")
CHAFA_BIN = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "chafa", "tools", "chafa", "chafa"
)
CHAFA_TRIM_START = len("\x1b[?25l\x1b[0m")
CHAFA_EXTRA_ARGS = ["-w", "1", "-O", "1"]
# This was "font ratio", 3/2, but with tiles that have already been converted
# to their correct aspect ratio by CHAFA, '1' provides the best "circle" effect
DEFAULT_RADIUS = 5
VIS_RATIO = 1
MAX_DARKNESS_LEVEL = 4
TIME_TICK = 0.50
MIN_TILE_SIZE = 7
MAX_TILE_SIZE = 29
TEXT_HISTORY_LENGTH = 1000


# probably better in the YAML, but gosh, lots of junk in "World" ?
SHIP_TILE_DIRECTIONS = {16: "West", 17: "North", 18: "East", 19: "South"}
DIRECTION_SHIP_TILES = {v: k for k, v in SHIP_TILE_DIRECTIONS.items()}

@contextlib.contextmanager
def elapsed_timer():
    """Timer pattern, from https://stackoverflow.com/a/30024601."""
    start = timeit.default_timer()

    def elapser():
        return timeit.default_timer() - start

    # pylint: disable=unnecessary-lambda
    yield lambda: elapser()

def flatten(layers):
    return [item for row in layers for item in row]


class Item(object):
    DEFAULT_PLAYER_TILE_ID = 31

    def __init__(self, tile_id, pos, name, material='construction',
                 where='floor', darkness=0, land_passable=True,
                 speed=0):
        self.tile_id = tile_id
        self.name = name
        self.material = material
        self.where = where
        self._pos = pos
        self.darkness = darkness
        self.land_passable = land_passable
        self.speed = speed

    @classmethod
    def create_player(cls, pos):
        return cls(tile_id=cls.DEFAULT_PLAYER_TILE_ID,
                   pos=pos, name='player',
                   material='flesh', where='unattached')
    
    @classmethod
    def create_boat(cls, pos, tile_id):
        return cls(tile_id=16, pos=pos, name='boat')

    @property
    def pos(self):
        return self._pos

    def distance(self, other_item):
        if other_item:
            return (
                abs(self.z - other_item.z)
                + abs(self.y - other_item.y)
                + abs(self.x - other_item.x)
            )
        return 0

    @property
    def is_boat(self):
        return self.tile_id in (16, 17, 18, 19)

    @property
    def is_horse(self):
        return self.tile_id in (20, 21)

    @property
    def is_flying(self):
        return self.tile_id == 24

    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def z(self):
        return self._pos.z

    @property
    def y(self):
        return self._pos.y

    @property
    def x(self):
        return self._pos.x

    def __repr__(self):
        return (
            f"{self.name}<{self.where} {self.material},id={self.tile_id} at "
            f"y={self.y},x={self.x}>"
        )

    def __str__(self):
        return f"{self.name}, {self.where} {self.material}"

    @classmethod
    def create_void(cls, pos):
        "Create an item that represents void, black space."
        return cls(
            tile_id=-1,
            pos=pos,
            name="void",
            material="liquid",
            where="buried",
            darkness=0,
            land_passable=False,
            speed=8,
        )


class World(object):
    time = 0
    TICK = 1
    clipping = True

    def __init__(self, Materials, Where, Portals, items):
        self.Where = Where
        self.Materials = Materials
        self.Portals = Portals
        self.items = items

        # bounding dimensions
        self._height = max(item.y for item in items) if items else 0
        self._width = max(item.x for item in items) if items else 0
        self._depth = max(item.z for item in items) if items else 0

        # cache lookup
        self._player = None


    def debug_details(self, pos, prefix=''):
        local_items = self.find_iter(z=pos.z, y=pos.y, x=pos.x) if prefix else []
        portal = self.find_portal(pos) if not prefix else None
        return {
            **({f"time": self.time} if not prefix else {}),
            **({f"{prefix}no-Materials": len(self.Materials)} if self.Materials else {}),
            **({f"{prefix}no-Where": len(self.Where)} if self.Where else {}),
            **({f"{prefix}no-Portals": len(self.Portals)} if self.Portals else {}),
            f"{prefix}no-items": len(self.items),
            **{
                f"{prefix}itm-{num}": repr(item)
                for num, item in enumerate(local_items)
            },
            **({
                f"{prefix}portal": repr(portal)
            } if portal else {}),
        }

    @property
    def height(self):
        # how many y-rows?
        return self._height

    @property
    def width(self):
        # how many x-columns?
        return self._width

    @property
    def depth(self):
        # how many z-layers?
        return self._depth

    def __repr__(self):
        return repr(self.items)

    def find_iter(self, **kwargs):
        return (
            item
            for item in self.items
            if all(getattr(item, key) == value for key, value in kwargs.items())
        )

    def find_one(self, **kwargs):
        try:
            return next(self.find_iter(**kwargs))
        except StopIteration:
            return None

    @property
    def player(self):
        if self._player is None:
            self._player = self.find_one(name="player")
        return self._player

    @classmethod
    def load(cls, map_id=0):
        # create from u4 map
        world_data = yaml.load(open(FPATH_WORLD_YAML, "r"), Loader=yaml.SafeLoader)
        items = []
        for (chunk_y, chunk_x), chunk_data in cls.read_u4_world_chunks().items():
            for idx, raw_val in enumerate(chunk_data):
                div_y, div_x = divmod(idx, 32)
                pos = Position(z=0, y=(chunk_y * 32) + div_y, x=(chunk_x * 32) + div_x)
                for where_type, definition in world_data["WorldMap"][raw_val].items():
                    item = Item(
                        tile_id=raw_val,
                        pos=pos,
                        where=where_type,
                        name=definition.get("name", None),
                        material=definition.get("material", None),
                        darkness=definition.get("darkness", 0),
                        land_passable=definition.get("land_passable", True),
                        speed=definition.get("speed", 0),
                    )
                    items.append(item)

        # Create the player, start @ Lord British's castle!
        player_pos = Position(z=0, y=107, x=86)
        items.append(Item(pos=player_pos, **world_data["Player"]))
        
        # Add test boat!
        boat_pos = Position(z=0, y=110, x=86)
        items.append(Item(pos=boat_pos, **world_data["Boat"]))

        return cls(
            Materials=enum.Enum("Material", world_data["Materials"]),
            Where=enum.Enum("Where", world_data["Where"]),
            Portals=world_data["Portals"],
            items=items,
        )

    @staticmethod
    def read_u4_world_chunks() -> dict[tuple[int, int], list[int]]:
        # read raw WORLD.DAT data as a dictionary keyed by (y, x) of 8x8 chunks
        # each value is a list of 32x32 tiles, keyed by their tileset id
        chunk_dim = 32
        chunk_len = chunk_dim * chunk_dim
        world_chunks = collections.defaultdict(list)
        with open(FPATH_WORLD_MAP, "rb") as fp:
            buf = bytearray(chunk_len)
            for y in range(8):
                for x in range(8):
                    n = fp.readinto(buf)
                    assert (
                        n == chunk_len
                    ), f"Read error: expected {chunk_len} bytes, got {n}"

                    for j in range(chunk_dim):
                        chunk_line = []
                        for i in range(chunk_dim // 4):
                            offset = j * chunk_dim + i * 4
                            chunk_line.extend(
                                [
                                    buf[offset],
                                    buf[offset + 1],
                                    buf[offset + 2],
                                    buf[offset + 3],
                                ]
                            )
                        world_chunks[y, x].extend(chunk_line)
        return world_chunks

    def do_move_player(self, viewport, y=0, x=0, z=0):
        previous_pos = self.player.pos
        pos = Position(z=self.player.z + z, y=self.player.y + y, x=self.player.x + x)
        can_move = False
        if not self.clipping:
            can_move = True
        elif not self.player.is_boat and viewport.small_world.land_passable(pos):
            can_move = True
        elif viewport.small_world.water_passable(pos):
            if self.player.is_boat:
                can_move = True
            else:
                boat = self.find_one(name="boat", pos=pos)
                if not boat:
                    viewport.add_text("BLOCKED!")
                else:
                    can_move = True
        else:
            viewport.add_text("BLOCKED ELSE!")
        if can_move and self.player.is_boat:
            can_move = self.check_boat_direction(y, x)
        if can_move:
            if not self.check_tile_movement(pos):
                viewport.add_text("SLOW PROGRESS!")
                can_move = False
        if can_move:
            if y:
                viewport.add_text(f">{'North' if y < 0 else 'South'}")
            if x:
                viewport.add_text(f">{'West' if x < 0 else 'East'}")
            self.player.pos = pos
        self.time += self.TICK
        return pos != previous_pos
    
    def board(self):
        boat = self.find_one(name="boat", pos=self.player.pos)
        if not boat:
            return False
        self.player.tile_id = boat.tile_id
        self.items.remove(boat)
        return True

    def exit_ship_or_unmount_horse(self):
        if not self.player.is_boat:
            return False
        boat = Item.create_boat(self.player.pos, self.player.tile_id)
        self.items.append(boat)
        self.player.tile_id = Item.DEFAULT_PLAYER_TILE_ID
        return True

    def check_boat_direction(self, y, x):
        boat_direction = SHIP_TILE_DIRECTIONS.get(self.player.tile_id)
        can_move = (
            boat_direction in ("North", "West") if (y < 0 and x < 0) else
            boat_direction in ("South", "East") if (y > 0 and x > 0) else
            boat_direction in ("North", "East") if (y < 0 and x > 0) else
            boat_direction in ("South", "West") if (y > 0 and x < 0) else
            boat_direction == "North" if y < 0 else
            boat_direction == "South" if y > 0 else
            boat_direction == "West" if x < 0 else
            boat_direction == "East" if x > 0 else
            False)
        next_direction = "West" if x < 0 else "East" if x > 0 else "North" if y < 0 else "South"
        # You can't move that way, but, turn towards that direction, though.
        self.player.tile_id = DIRECTION_SHIP_TILES.get(next_direction, self.player.tile_id)
        return can_move


    def find_portal(self, pos):
        """
        Check for and return any matching portal definition found at pos
        """
        for portal in self.Portals:
            if portal['y'] == pos.y and portal['x'] == pos.x:
                return {'dest_id': portal['dest_id'], 'action': portal['action']}

    def check_tile_movement(self, pos):
        # if any tile at given location has a "speed" variable, then,
        # use as random "SLOW PROGRESS!" deterrent for difficult terrain
        for item in self.find_iter(z=pos.z, y=pos.y, x=pos.x, where='buried'):
            if item.speed:
                return random.randrange(item.speed) != 0
        return True

    def land_passable(self, pos):
        for item in self.find_iter(z=pos.z, y=pos.y, x=pos.x):
            if not item.land_passable:
                return False
            elif item.material == "liquid":
                return False
        return True

    def water_passable(self, pos):
        for item in self.find_iter(z=pos.z, y=pos.y, x=pos.x):
            if item.tile_id in (0, 1):
                return True
        return False

    def light_blocked(self, pos):
        # whether player movement, or casting of "light" is blocked
        is_void = True
        for item in self.find_iter(z=pos.z, y=pos.y, x=pos.x):
            is_void = False
            if item.darkness > 0:
                return True
        return is_void

    def darkness(self, item):
        distance = item.distance(self.player)
        return min(max(0, distance - 2), MAX_DARKNESS_LEVEL)


class UInterface(object):
    movement_map = {
        # given input key, move given x/y coords
        "h": {"x": -1},
        "j": {"y": 1},
        "k": {"y": -1},
        "l": {"x": 1},
        "y": {"y": -1, "x": -1},
        "u": {"y": -1, "x": 1},
        "b": {"y": 1, "x": -1},
        "n": {"y": 1, "x": 1},
    }

    def __init__(self, tile_size):
        self.term = blessed.Terminal()
        self.dirty = True
        self.radius = DEFAULT_RADIUS
        self.darkness = True
        self.tile_size = tile_size

    @property
    def tile_size(self):
        return self._tile_size

    @tile_size.setter
    def tile_size(self, value):
        self._tile_size = value
        self.tile_width = value
        # determine final tile height by encoding any tile and measuring result
        self.tile_height = len(
            UInterface.get_tile(id=0, width=value, height=value, darkness=0)
        )

    @property
    def window_size(self):
        return (self.term.height, self.term.width)

    def reader(self, timeout):
        return self.term.inkey(timeout=timeout)

    def reactor(self, inp, world, viewport):
        self.dirty = True
        if inp in self.movement_map:
            self.dirty = world.do_move_player(viewport, **self.movement_map[inp])
        elif inp == 'E' and world.find_portal(world.player.pos):
            # 'E'nter 'P'ortal !
        elif inp == 'B':
            self.dirty = world.board()
        elif inp == 'X':
            self.dirty = world.exit_ship_or_unmount_horse()
        # keys for wizards !
        elif inp == "C":
            world.clipping = not world.clipping
        elif inp == "A":
            self.auto_resize(viewport)
        elif inp == "R":
            self.radius = DEFAULT_RADIUS if not self.radius else None
        elif inp == "D":
            self.darkness = not self.darkness
        elif inp == ")" and self.radius is not None and self.radius <= 10:
            self.radius += 1
        elif inp == "(" and self.radius is not None and self.radius >= 1:
            self.radius -= 1
        elif inp == "[" and self.tile_size > MIN_TILE_SIZE:
            self.tile_size -= 2
        elif inp == "]" and self.tile_size < MAX_TILE_SIZE:
            self.tile_size += 2
        else:
            self.dirty = False

    def auto_resize(self, viewport):
        if self.radius:
            while self.tile_size < MAX_TILE_SIZE and (
                ((self.radius) * 2) + 2 > (viewport.width / self.tile_size) - 1
            ):
                self.dirty = True
                self.tile_size -= 2
                viewport.add_text(f"resize tile -1, ={self.tile_size}, "
                                  f"viewport_width={viewport.width}, "
                                  f"tile_width={self.tile_width}, "
                                  f"tile_height={self.tile_height}, "
                                  f"radius * 2={self.radius * 2}, ")
            while self.tile_size > MIN_TILE_SIZE and (
                (self.radius * 2) + 2 < (viewport.width / self.tile_size)
            ):
                self.dirty = True
                self.tile_size += 2
                viewport.add_text(f"resize tile +1, ={self.tile_size}, "
                                  f"viewport_width={viewport.width}, "
                                  f"tile_width={self.tile_width}, "
                                  f"tile_height={self.tile_height}, "
                                  f"radius * 2={self.radius * 2}, ")

    @contextlib.contextmanager
    def activate(self):
        with self.term.fullscreen(), self.term.keypad(), self.term.cbreak(), self.term.hidden_cursor():
            echo(self.term.clear)
            yield self

    def render_text(self, viewport, debug_details):
        ypos = viewport.yoffset - 1
        left = viewport.width + (viewport.xoffset * 2)
        width = max(0, self.term.width - left - (viewport.xoffset))
        if width == 0:
            return
        for debug_item in debug_details.items():
            debug_text_lines = textwrap.wrap(
                f'{debug_item[0]}: {debug_item[1]}',
                width=width, subsequent_indent=' ')
            for text_line in debug_text_lines:
                ypos += 1
                echo(self.term.move_yx(ypos, left))
                echo(self.term.ljust(text_line, width))
        ypos += 1
        echo(self.term.move_yx(ypos, left))
        echo(' ' * width)
        ypos += 1
        echo(self.term.move_yx(ypos, left))
        echo('=' * width)
        ypos += 1
        echo(self.term.move_yx(ypos, left))
        echo(' ' * width)
        all_text = ['x']
        remaining_y = viewport.height - ypos
        for text_message in list(viewport.text)[-remaining_y:]:
            all_text.extend(textwrap.wrap(
                text_message, width=width, subsequent_indent='  '))
        for text_line in all_text[-remaining_y:]:
            ypos += 1
            echo(self.term.move_yx(ypos, left))
            echo(self.term.ljust(text_line, width))
        while viewport.height - ypos > 0:
            ypos += 1
            echo(self.term.move_yx(ypos, left))
            echo(' ' * width)

    def maybe_draw_viewport(self, viewport, force=False):
        # todo: make exactly like IV, with moon phases, etc!
        if viewport.dirty or force:
            border_color = self.term.yellow_reverse
            echo(self.term.home)
            echo(border_color(" " * self.term.width) * viewport.yoffset)
            for ypos in range(viewport.height):
                echo(self.term.move(viewport.yoffset + ypos, 0))
                echo(border_color(" " * viewport.xoffset))
                echo(
                    self.term.move(
                        viewport.yoffset + ypos, viewport.xoffset + viewport.width
                    )
                )
                echo(border_color(" " * viewport.xoffset))
                echo(
                    self.term.move(
                        viewport.yoffset + ypos, self.term.width - viewport.xoffset
                    )
                )
                echo(border_color(" " * viewport.xoffset))
            echo(border_color(" " * self.term.width) * viewport.yoffset, flush=True)
        viewport.dirty = False

    def render(self, world, viewport):
        viewport.re_adjust(world, ui=self)
        if viewport.dirty:
            self.auto_resize(viewport)
        self.maybe_draw_viewport(viewport)
        if self.dirty or viewport.dirty:
            items_by_row = viewport.items_in_view_by_row(world, ui=self)
            for cell_row, cell_items in enumerate(items_by_row):
                ypos = cell_row * self.tile_height
                for cell_number, items in enumerate(cell_items):
                    xpos = cell_number * self.tile_width
                    actual_xpos = xpos + viewport.xoffset
                    if items:
                        tile_darkness = viewport.small_world.darkness(items[-1]) if self.darkness else 0
                        width = self.tile_width
                        tile_ans = UInterface.get_tile(
                            items[0].tile_id,
                            width=width,
                            height=self.tile_height,
                            darkness=tile_darkness)
                        for ans_y, ans_txt in enumerate(tile_ans):
                            actual_ypos = ypos + ans_y + viewport.yoffset
                            if actual_ypos <= viewport.height:
                                echo(self.term.move_yx(actual_ypos, actual_xpos))
                                echo(ans_txt)
        echo('', flush=True)
        self.dirty = False

    @functools.lru_cache(maxsize=256)
    @staticmethod
    def get_tile(
        id,
        width,
        height,
        # effects,
        darkness,
        x_offset=0,
        y_offset=0,
    ):
        if id == -1:
            return [" " * width] * height

        fpath_png = os.path.join(
            os.path.dirname(__file__), "tiles", f"tile_{id:02X}_{darkness}.png"
        )

        bdata = open(fpath_png, "rb").read()
        if y_offset or x_offset:
            # load PNG data to apply effects with PIL,
            image = PIL.Image.open(io.BytesIO(bdata))
            if y_offset or x_offset:
                image = image.offset(x_offset, y_offset)
            ## export image as PNG bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            bdata = img_byte_arr.getvalue()
        chafa_cmd_args = [
            CHAFA_BIN,
            *CHAFA_EXTRA_ARGS,
            "--size",
            f"{width}x{width}",
            "-",
        ]
        ans = subprocess.check_output(chafa_cmd_args, input=bdata).decode()
        lines = ans.splitlines()
        # remove any hide/show cursor attributes
        return [lines[0][CHAFA_TRIM_START:]] + lines[1:-1]


class Viewport:
    """
    A "Viewport" represents the game world window of tiles, where it is
    located on the screen (height, width, yoffset, xoffset), and what
    game world z/y/x is positioned at the top-left.
    """

    MULT = collections.namedtuple("fastmath_table", ["xx", "xy", "yx", "yy"])(
        xx=[1, 0, 0, -1, -1, 0, 0, 1],
        xy=[0, 1, -1, 0, 0, -1, 1, 0],
        yx=[0, 1, 1, 0, 0, -1, -1, 0],
        yy=[1, 0, 0, 1, -1, 0, 0, -1],
    )

    def __init__(self, z, y, x, height, width, yoffset, xoffset):
        (self.z, self.y, self.x) = (z, y, x)
        self.height, self.width = height, width
        self.yoffset, self.xoffset = yoffset, xoffset
        self.dirty = True
        self.text = collections.deque(maxlen=TEXT_HISTORY_LENGTH)
        self.small_world = World(
            Materials=None,
            Where=None,
            Portals=None,
            items=list())


    def __repr__(self):
        return f"{self.z}, {self.y}, {self.x}"

    def add_text(self, text):
        self.text.append(text)

    @classmethod
    def create(cls, world, ui, yoffset=1, xoffset=2):
        "Create viewport instance centered one z-level above player."
        vp = cls(0, 0, 0, 1, 1, yoffset, xoffset)
        vp.re_adjust(world, ui)
        vp.dirty = True
        return vp

    def re_adjust(self, world, ui):
        "re-center viewport on player and set 'dirty' flag on terminal resize"
        height = ui.term.height - (self.yoffset * 2)
        width = min(ui.term.width - 20, int(ui.term.width * 0.8))
        self.dirty = (height, width) != (self.height, self.width)
        self.height, self.width = height, width

        tiled_height = (height // ui.tile_height) + 1
        tiled_width = width // ui.tile_width

        self.z = world.player.z - 1
        self.y = world.player.y - int(math.floor(tiled_height / 2)) + 1
        self.x = world.player.x - int(math.floor(tiled_width / 2))

    def items_in_view_by_row(self, world, ui):
        # find the ideal tile_size,
        tiled_height = math.ceil(self.height / ui.tile_height)
        tiled_width = math.floor(self.width / ui.tile_width)
        y_min, y_max = self.y, self.y + tiled_height
        x_min, x_max = self.x, self.x + tiled_width

        def calc_visible(item):
            # rename to culling
            # may be seen from above, "eagle's eye", and is within
            # radius squares bounding box.
            return (
                self.z < item.z and x_min <= item.x < x_max and y_min <= item.y < y_max
            )

        # sort by top-most visible item
        def sort_func(item):
            return item.z, world.Where[item.where].value

        occlusions = collections.defaultdict(list)
        for item in sorted(
            filter(calc_visible, world.items), key=sort_func, reverse=False
        ):
            occlusions[(item.y, item.x)].append(item)

        self.small_world = World(
            Materials=None,
            Where=None,
            Portals=None,
            items=flatten(occlusions.values()),
        )
        visible = set()
        if ui.radius:
            for oct in range(8):
                visible.update(
                    self.cast_light(
                        z=world.player.z,
                        cx=world.player.x,
                        cy=world.player.y,
                        row=1,
                        start=1.0,
                        end=0.0,
                        radius=ui.radius,
                        xx=self.MULT.xx[oct],
                        xy=self.MULT.xy[oct],
                        yx=self.MULT.yx[oct],
                        yy=self.MULT.yy[oct],
                        depth=0,
                    )
                )

        for y in range(self.y, self.y + tiled_height):
            yield [
                occlusions.get((y, x)) or [Item.create_void(pos=Position(self.z, y, x))]
                for x in range(self.x, self.x + tiled_width)
            ]

            #yield [
            #    item
            #    if (
            #        ui.radius is None
            #        or (item.y, item.x) in visible
            #        or item == world.player
            #    )
            #    else Item.create_void(pos=Position(self.z, item.y, item.x))
            #    for item in candidate_row_items
            #]

    def cast_light(self, z, cx, cy, row, start, end, radius, xx, xy, yx, yy, depth):
        "Recursive lightcasting function"
        visible = set()
        if start < end:
            return visible
        radius_squared = radius * radius
        for j in range(row, radius + 1):
            dx, dy = -j - 1, -j
            blocked = False
            while dx <= 0:
                dx += 1
                # Translate the dx, dy coordinates into map coordinates:
                X = cx + dx * xx + dy * xy
                Y = cy + dx * yx + dy * yy
                # l_slope and r_slope store the slopes of the left and right
                # extremities of the square we're considering:
                l_slope, r_slope = (dx - 0.5) / (dy + 0.5), (dx + 0.5) / (dy - 0.5)
                if start < r_slope:
                    continue
                elif end > l_slope:
                    break
                # Our light beam is touching this square; light it,
                if (dx * dx + dy * dy) < radius_squared and abs(
                    dx * yx + dy * yy
                ) < radius * VIS_RATIO:
                    visible.add((Y, X))
                if blocked:
                    # we're scanning a row of blocked squares:
                    if self.small_world.light_blocked(Position(z, Y, X)):
                        new_start = r_slope
                    else:
                        blocked = False
                        start = new_start
                    continue
                if self.small_world.light_blocked(Position(z, Y, X)) and j < radius:
                    # This is a blocking square, start a child scan:
                    blocked = True
                    visible.update(
                        self.cast_light(
                            z,
                            cx,
                            cy,
                            j + 1,
                            start,
                            l_slope,
                            radius,
                            xx,
                            xy,
                            yx,
                            yy,
                            depth + 1,
                        )
                    )
                    new_start = r_slope
            # Row is scanned; do next row unless last square was blocked:
            if blocked:
                break
        return visible

    def do_fov(self, x, y, radius):
        "Calculate lit squares from the given location and radius"
        self.flag += 1
        for oct in range(8):
            self.cast_light(
                x,
                y,
                1,
                1.0,
                0.0,
                radius,
                self.mult[0][oct],
                self.mult[1][oct],
                self.mult[2][oct],
                self.mult[3][oct],
                0,
            )


def _loop(ui, world):
    inp = None
    (
        time_render,
        time_action,
        time_input,
        time_stats,
    ) = (
        0,
        0,
        0,
        0,
    )

    viewport = Viewport.create(world, ui)
    ui.auto_resize(viewport)
    ui.maybe_draw_viewport(viewport, force=True)

    time_render = time_action = time_input = time_stats = lambda: 0

    time_render = 0
    while True:
        with elapsed_timer() as time_stats:
            ui.render_text(
                viewport,
                debug_details={
                    "ms-render-world": int(time_render * 1000),
                    "ms-action": int(time_action() * 1000),
                    "ms-input": int(time_input() * 1000),
                    "ms-stats": int(time_stats() * 1000),
                    "tile-cache": str(ui.get_tile.cache_info()),
                    # of "whole world"
                    **world.debug_details(pos=world.player.pos),
                    # details of "small world"
                    **viewport.small_world.debug_details(pos=world.player.pos, prefix='sm-'),
                    "tile-width": ui.tile_width,
                    "tile-height": ui.tile_height,
                    "radius": ui.radius,
                    "darkness": ui.darkness,
                    "clipping": world.clipping,
                    "term-height": ui.term.height,
                    "term-width": ui.term.width,
                },
            )

        with elapsed_timer() as time_render:
            ui.render(world, viewport)
        time_render = time_render()

        with elapsed_timer() as time_input:
            inp = ui.reader(timeout=max(0, TIME_TICK))
            # throw away remaining input, small hack for
            # games where folks bang on the keys to run
            # (or boat!) as fast as they can, take "out"
            # all keys, then push back in the last-most
            # key.
            if inp:
                save_key = None
                while inp2 := ui.term.inkey(timeout=0):
                    save_key = inp2
                if save_key:
                    ui.term.ungetch(save_key)


        with elapsed_timer() as time_action:
            world = ui.reactor(inp, world, viewport)


def main():
    ui = UInterface(tile_size=16)
    with ui.activate():
        world = World.load()
        _loop(ui, world)


if __name__ == "__main__":
    exit(main())


# graphics improvements todo,
# - forests should cast shadows, but be movable
# - scale text character sets, so 16x+ get matching "large text"
# - trim tiles to fill screen edges
# - then better center the avatar, divide and center screen remainder
# - smooth scrolling
# - animated tiles,
#   - dynamically shift X+Y every frame?
#   - wave flags of castles
# TODO: brightness depending on moon cycles !!
#
# movement improvements
# - '_' go to function, pulls up map ..
# - use the "braille" for world map, maybe show & highlight relative position
# - when on balloon, and z+1, radius should also increase
#
# There should be a client/server interface ??
# - all player-to-player interaction is done through SQL
#   where possible (location, tile_id)
# - a client only renders what's going on, text + tile_id's,
#   that are interactions
# - a server generates the AI, responds to NPC chat, etc.
# There should be two blended worlds,
# - one static world, retrieved from SQL, never refreshed,
#   always in memory, chunked, and fast to query, for
#   drawing landscape, entire worldmap, all of buildings
#   everything inanimate. Could be a read-only SQL table!
# - a dynamic world of living items in an SQL table, just
#   living "items" managed by (something)
# todo: use 'small_world' !
# todo: implement new moon cycles effect darkness
# we can always see well
#

# ideas,
# - world map
# - fog of war for unexplored areas in map only
# - running ship into short or shallow water should beach it
#    you crash! ship -20, you are expelled to nearest land,
#    or, if no land, your party drowns! be more careful!
# - horse bucks you if mount in mud, you must unmount and
#   walk a horse through mud