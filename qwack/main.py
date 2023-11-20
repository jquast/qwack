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
import time
import io
import os

# 3rd party
import blessed
import yaml
import PIL.Image

# local
import u4_data

echo = functools.partial(print, end="")
Position = collections.namedtuple("Position", ("y", "x"))

CHAFA_BIN = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "chafa", "tools", "chafa", "chafa"
)
CHAFA_TRIM_START = len("\x1b[?25l\x1b[0m")
CHAFA_EXTRA_ARGS = ["-w", "1", "-O", "1"]
# This was "font ratio", 3/2, but with tiles that have already been converted
# to their correct aspect ratio by CHAFA, '1' provides the best "circle" effect
DEFAULT_RADIUS = 6
VIS_RATIO = 1
MAX_DARKNESS_LEVEL = 4
TIME_ANIMATION_TICK = 0.30
TIME_PLAYER_PASS = 23
MIN_TILE_SIZE = 7
MAX_TILE_SIZE = 29
TEXT_HISTORY_LENGTH = 1000
DEFAULT_TILE_SIZE = 16
MAX_RADIUS = 9
LORD_BRITISH_CASTLE_ID = 14


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

    def __init__(
        self,
        tile_id,
        pos,
        name,
        material="construction",
        where="floor",
        darkness=0,
        land_passable=True,
        speed=0,
    ):
        self.tile_id = tile_id
        self.name = name
        self.material = material
        self.where = where
        self._pos = pos
        self.darkness = darkness
        self.land_passable = land_passable
        self.speed = speed
        self.last_action_tick = 0

    @classmethod
    def create_player(cls, pos):
        return cls(
            tile_id=cls.DEFAULT_PLAYER_TILE_ID,
            pos=pos,
            name="player",
            material="flesh",
            where="unattached",
        )

    @classmethod
    def create_boat(cls, pos, tile_id=None):
        return cls(tile_id=16 if tile_id is None else tile_id, pos=pos, name="boat")

    @classmethod
    def create_horse(cls, pos_tile_id=None):
        return cls(tile_id=20 if pos_tile_id is None else pos_tile_id, name="horse")

    @property
    def pos(self):
        return self._pos

    def is_adjacent(self, other_item):
        # Check if the target coordinates are adjacent to the given coordinates
        return (abs(self.x - other_item.x) == 1 and self.y == other_item.y or
                abs(self.y - other_item.y) == 1 and self.x == other_item.x)

    def distance(self, other_item):
        return math.sqrt((self.x - other_item.x)**2 + (self.y - other_item.y)**2)

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
    def y(self):
        return self._pos[0]

    @property
    def x(self):
        return self._pos[1]

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
    wizard_mode = False

    def __init__(self, items=None, Materials=None, Where=None, Portals=None, world_data=None):
        self.Where = Where
        self.Materials = Materials
        self.Portals = Portals
        self.items = items or []
        self.world_data = world_data

        # bounding dimensions
        self._height = max(item.y for item in items) if items else 0
        self._width = max(item.x for item in items) if items else 0

        # cache lookup
        self._player = None
        self.time_monotonic_last_action = time.monotonic()

    def debug_details(self, pos, small_world=False):
        local_items = self.find_iter(y=pos.y, x=pos.x) if small_world else []
        portal = self.find_portal(pos) if not small_world else None
        prefix = "sm-" if small_world else ""
        return {
            **({f"time": self.time} if not small_world else {}),
            **({"clipping": self.clipping} if not small_world else {}),
            **(
                {f"{prefix}no-Materials": len(self.Materials)} if self.Materials else {}
            ),
            **({f"{prefix}no-Where": len(self.Where)} if self.Where else {}),
            **({f"{prefix}no-Portals": len(self.Portals)} if self.Portals else {}),
            f"{prefix}no-items": len(self.items),
            **{
                f"{prefix}itm-{num}": repr(item) for num, item in enumerate(local_items)
            },
            **({f"{prefix}portal": repr(portal)} if portal else {}),
        }

    @property
    def height(self):
        # how many y-rows?
        return self._height

    @property
    def width(self):
        # how many x-columns?
        return self._width

    def __repr__(self):
        return repr(self.items)

    def find_iter(self, **kwargs):
        return (
            item
            for item in self.items
            if all(getattr(item, key) == value for key, value in kwargs.items())
        )

    def find_iter_not_player(self, **kwargs):
        return (
            item
            for item in self.find_iter(**kwargs)
            if item.name != 'player'
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
    def load(cls, map_id: int, world_data: dict):
        # create from u4 map, join with "World Data" definitions from u4-world.yaml
        items = cls.make_tile_items(map_id, world_data)
        if map_id != -1:
            for npc_definition in u4_data.load_npcs_from_u4_ult_map(map_id, world_data):
                items.append(Item(**npc_definition))

        if map_id == -1:
            # Create the player, start @ Lord British's castle!
            items.append(Item.create_player(Position(y=107, x=86)))
        else:
            items.append(Item.create_player(Position(y=0, x=0)))

        return cls(
            Materials=enum.Enum("Material", world_data["Materials"]),
            Where=enum.Enum("Where", world_data["Where"]),
            Portals=world_data["Portals"],
            items=items,
            world_data=world_data,
        )

    @classmethod
    def make_tile_items(cls, map_id, world_data):
        items = []
        chunk_size = 32 if map_id == -1 else 1
        map_chunks = u4_data.read_map(map_id)
        for (chunk_y, chunk_x), chunk_data in map_chunks.items():
            for idx, raw_val in enumerate(chunk_data):
                div_y, div_x = divmod(idx, chunk_size)
                pos = Position(
                    y=(chunk_y * chunk_size) + div_y,
                    x=(chunk_x * chunk_size) + div_x,
                )
                tile_definition = world_data["WorldMap"][raw_val]
                item = Item(
                    tile_id=raw_val,
                    pos=pos,
                    where=tile_definition.get("where", "buried"),
                    name=tile_definition.get("name", None),
                    material=tile_definition.get("material", "construction"),
                    darkness=tile_definition.get("darkness", 0),
                    land_passable=tile_definition.get("land_passable", True),
                    speed=tile_definition.get("speed", 0),
                )
                items.append(item)
        return items

    def do_move_player(self, viewport, y=0, x=0):
        previous_pos = self.player.pos
        pos = Position(y=self.player.y + y, x=self.player.x + x)
        can_move = False
        if not self.clipping:
            can_move = True
        # we are not a boat, and it is land,
        elif not self.player.is_boat and viewport.small_world.land_passable(pos):
            can_move = True
        # the target is water,
        elif viewport.small_world.water_passable(pos):
            # we are a boat,
            if self.player.is_boat:
                can_move = True
            else:
                # we are not a boat, but we can board one
                boat = self.find_one(name="boat", pos=pos)
                if not boat:
                    viewport.add_text("BLOCKED!")
                else:
                    can_move = True
        else:
            viewport.add_text("BLOCKED!")
        if can_move and self.player.is_boat:
            can_move = self.check_boat_direction(y, x)
        if can_move:
            move_result = viewport.small_world.check_tile_movement(pos)
            if move_result == 0:
                viewport.add_text("SLOW PROGRESS!")
                can_move = False
            elif move_result == -1:
                viewport.add_text("BLOCKED!")
                can_move = False
        if can_move:
            if y:
                viewport.add_text(f">{'North' if y < 0 else 'South'}")
            if x:
                viewport.add_text(f">{'West' if x < 0 else 'East'}")
            self.player.pos = pos
        return pos != previous_pos

    def board_ship_or_mount_horse(self):
        boat = self.find_one(name="boat", pos=self.player.pos)
        if not boat and self.wizard_mode:
            # wizards can "Board" any tile, LoL!
            boat = next(self.find_iter_not_player(pos=self.player.pos))
        elif not boat:
            return False
        self.player.tile_id = boat.tile_id
        self.items.remove(boat)
        return True

    def exit_ship_or_unmount_horse(self):
        if not self.player.is_boat and not self.wizard_mode:
            return False
        boat = Item.create_boat(self.player.pos, self.player.tile_id)
        self.items.append(boat)
        self.player.tile_id = Item.DEFAULT_PLAYER_TILE_ID
        return True

    def check_boat_direction(self, y, x):
        boat_direction = SHIP_TILE_DIRECTIONS.get(self.player.tile_id)
        can_move = (
            boat_direction in ("North", "West")
            if (y < 0 and x < 0)
            else boat_direction in ("South", "East")
            if (y > 0 and x > 0)
            else boat_direction in ("North", "East")
            if (y < 0 and x > 0)
            else boat_direction in ("South", "West")
            if (y > 0 and x < 0)
            else boat_direction == "North"
            if y < 0
            else boat_direction == "South"
            if y > 0
            else boat_direction == "West"
            if x < 0
            else boat_direction == "East"
            if x > 0
            else False
        )
        next_direction = (
            "West" if x < 0 else "East" if x > 0 else "North" if y < 0 else "South"
        )
        # You can't move that way, but, turn towards that direction, though.
        self.player.tile_id = DIRECTION_SHIP_TILES.get(
            next_direction, self.player.tile_id
        )
        return can_move

    def find_portal(self, pos):
        """
        Check for and return any matching portal definition found at pos
        """
        if self.Portals:
            for portal in self.Portals:
                if portal["y"] == pos.y and portal["x"] == pos.x:
                    return {"dest_id": portal["dest_id"], "action": portal["action"]}

    def check_tile_movement(self, pos) -> int:
        # if any tile at given location has a "speed" variable, then,
        # use as random "SLOW PROGRESS!" deterrent for difficult terrain

        # When travelling north, check if player is on Lord British's Castle
        # and Deny movement on any match
        if pos.y < self.player.y:
            for item in self.find_iter(y=self.player.y, x=self.player.x, tile_id=LORD_BRITISH_CASTLE_ID):
                return -1
        for item in self.find_iter(y=pos.y, x=pos.x, where="buried"):
            if item.speed:
                # returns 0 when progress is impeded
                return int(random.randrange(item.speed) != 0)
            if item.tile_id == LORD_BRITISH_CASTLE_ID:
                # Lord British's Castle cannot be entered from the North
                if pos.y > self.player.y:
                    return -1
        return True

    def land_passable(self, pos):
        for item in self.find_iter(y=pos.y, x=pos.x):
            if not item.land_passable:
                return False
            elif item.material == "liquid":
                return False
        return True

    def water_passable(self, pos):
        for item in self.find_iter(y=pos.y, x=pos.x):
            if item.tile_id in (0, 1):
                return True
        return False

    def light_blocked(self, pos):
        # whether player movement, or casting of "light" is blocked
        is_void = True
        for item in self.find_iter(y=pos.y, x=pos.x):
            is_void = False
            if item.darkness > 0:
                return True
        return is_void

    def darkness(self, item):
        distance = item.distance(self.player)
        fn_trim = math.ceil if random.randrange(2) else math.floor
        return fn_trim(min(max(0, distance - 2), MAX_DARKNESS_LEVEL))

    def tick(self, small_world):
        # "tick" the engine forward and perform "AI",
        # using items in "small_world" as an optimization
        # to mutate the items in self
        #
        # Ultima IV was cruel, it always advanced the time, even without input
        # or making an invalid action, etc
        self.time += self.TICK
        self.check_close_opened_doors(small_world)
    
    def check_close_opened_doors(self, small_world):
        # close door after 4 game ticks
        for door in small_world.find_iter(tile_id=60):
            if self.time > door.last_action_tick + 4:
                door.tile_id = 59
        

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

    def __init__(self):
        self.term = blessed.Terminal()
        self.dirty = True
        self.radius = DEFAULT_RADIUS
        tile_size = DEFAULT_TILE_SIZE
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
        assert self.tile_height != self.tile_width # XXX CHECK VALV
        assert self.tile_height != 16, self.tile_height

    @property
    def window_size(self):
        return (self.term.height, self.term.width)

    def reader(self, timeout):
        return self.term.inkey(timeout=timeout)

    def reactor(self, inp, world, viewport):
        self.dirty = True
        if inp in self.movement_map:
            self.dirty = world.do_move_player(viewport, **self.movement_map[inp])
            # TODO: could return a new world, when exiting the current one
            # world = world.maybe_exit(world)
        elif inp == "O":
            # 'O'pen door
            door = world.find_one(y=world.player.y, x=world.player.x, tile_id=59)
            if door:
                door.tile_id = 60
                door.last_action_tick = world.time
                self.dirty = True
        elif inp == "E":
            # 'E'nter Portal
            portal = world.find_portal(world.player.pos)
            if portal:
                viewport.dirty = True
                world = World.load(map_id=portal["dest_id"], world_data=world.world_data)
        elif inp == "B":
            # 'B'oard ship or mount horse
            self.dirty = world.board_ship_or_mount_horse()
        elif inp == "X":
            # e 'X'it ship or unmount horse
            self.dirty = world.exit_ship_or_unmount_horse()
        elif inp == "{" and self.tile_size > MIN_TILE_SIZE:
            self.tile_size -= 2
        elif inp == "}" and self.tile_size < MAX_TILE_SIZE:
            self.tile_size += 2
        elif inp == '\x17':   # Control-W
            world.wizard_mode = not world.wizard_mode
        elif world.wizard_mode:
            # keys for wizards !
            if inp == "C":
                world.clipping = not world.clipping
            elif inp == "A":
                self.auto_resize(viewport)
            elif inp == "R":
                self.radius = DEFAULT_RADIUS if not self.radius else None
            elif inp == "\x04":  # ^D
                self.darkness = not self.darkness
            elif inp == ")" and self.radius is not None and self.radius < MAX_RADIUS:
                self.radius += 1
            elif inp == "(" and self.radius is not None and self.radius >= 2:
                self.radius -= 1
        # even when we don't move, the world may forcefully tick!
        else:
            if time.monotonic() > world.time_monotonic_last_action + TIME_PLAYER_PASS:
                world.player.last_action_tick = world.time
                world.tick(viewport.small_world)
            else:
                self.dirty = False
        return world

    def auto_resize(self, viewport):
        if self.radius:
            while self.tile_size < MAX_TILE_SIZE and (
                ((self.radius) * 2) + 2 > (viewport.width / self.tile_size) - 1
            ):
                self.dirty = True
                self.tile_size -= 2
                viewport.add_text(
                    f"resize tile -1, ={self.tile_size}, "
                    f"viewport_width={viewport.width}, "
                    f"tile_width={self.tile_width}, "
                    f"tile_height={self.tile_height}, "
                    f"radius * 2={self.radius * 2}, "
                )
            while self.tile_size > MIN_TILE_SIZE and (
                (self.radius * 2) + 2 < (viewport.width / self.tile_size)
            ):
                self.dirty = True
                self.tile_size += 2
                viewport.add_text(
                    f"resize tile +1, ={self.tile_size}, "
                    f"viewport_width={viewport.width}, "
                    f"tile_width={self.tile_width}, "
                    f"tile_height={self.tile_height}, "
                    f"radius * 2={self.radius * 2}, "
                )

    @contextlib.contextmanager
    def activate(self):
        with self.term.fullscreen(), self.term.keypad(), self.term.cbreak(), self.term.hidden_cursor():
            echo(self.term.clear)
            yield self

    def debug_details(self):
        tile_cache_efficiency = 1 - (self.get_tile.cache_info().misses / (self.get_tile.cache_info().hits or 1))
        return {
            "tile-cache": f'{tile_cache_efficiency*100:2.2f}%',
            "tile-width": self.tile_width,
            "tile-height": self.tile_height,
            "radius": self.radius,
            "darkness": self.darkness,
            "term-height": self.term.height,
            "term-width": self.term.width,
        }

    def render_text(self, viewport, debug_details):
        ypos = viewport.yoffset - 1
        left = viewport.width + (viewport.xoffset * 2)
        width = max(0, self.term.width - left - (viewport.xoffset))
        if width == 0:
            return
        for debug_item in debug_details.items():
            debug_text_lines = textwrap.wrap(
                f"{debug_item[0]}: {debug_item[1]}", width=width, subsequent_indent=" "
            )
            for text_line in debug_text_lines:
                ypos += 1
                echo(self.term.move_yx(ypos, left))
                echo(self.term.ljust(text_line, width))
        ypos += 1
        echo(self.term.move_yx(ypos, left))
        echo(" " * width)
        ypos += 1
        echo(self.term.move_yx(ypos, left))
        echo("=" * width)
        ypos += 1
        echo(self.term.move_yx(ypos, left))
        echo(" " * width)
        all_text = ["x"]
        remaining_y = viewport.height - ypos
        for text_message in list(viewport.text)[-remaining_y:]:
            all_text.extend(
                textwrap.wrap(text_message, width=width, subsequent_indent="  ")
            )
        for text_line in all_text[-remaining_y:]:
            ypos += 1
            echo(self.term.move_yx(ypos, left))
            echo(self.term.ljust(text_line, width))
        while viewport.height - ypos > 0:
            ypos += 1
            echo(self.term.move_yx(ypos, left))
            echo(" " * width)

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
            # self.auto_resize(viewport)
            self.dirty = True
        self.maybe_draw_viewport(viewport)
        if self.dirty:
            items_by_row = viewport.items_in_view_by_row(world, ui=self)
            for cell_row, cell_items in enumerate(items_by_row):
                ypos = cell_row * self.tile_height
                for cell_number, items in enumerate(cell_items):
                    xpos = cell_number * self.tile_width
                    actual_xpos = xpos + viewport.xoffset
                    if items:
                        tile_darkness = (
                            viewport.small_world.darkness(items[0])
                            if self.darkness
                            else 0
                        )
                        tile_ans = UInterface.get_tile(
                            items[0].tile_id,
                            width=self.tile_width,
                            height=self.tile_height,
                            darkness=tile_darkness,
                        )
                        for ans_y, ans_txt in enumerate(tile_ans):
                            actual_ypos = ypos + ans_y + viewport.yoffset
                            if actual_ypos <= viewport.height:
                                echo(self.term.move_yx(actual_ypos, actual_xpos))
                                echo(ans_txt)
            echo("", flush=True)
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
        if id == -1 or darkness == MAX_DARKNESS_LEVEL:
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
        return_list = [lines[0][CHAFA_TRIM_START:]] + lines[1:-1]
        return return_list


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
        (self.y, self.x) = (y, x)
        self.height, self.width = height, width
        self.yoffset, self.xoffset = yoffset, xoffset
        self.dirty = True
        self.text = collections.deque(maxlen=TEXT_HISTORY_LENGTH)
        self.small_world = World()

    def __repr__(self):
        return f"{self.y}, {self.x}"

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

        self.y = world.player.y - int(math.ceil(self.get_tiled_height(ui) / 2)) + 1
        self.x = world.player.x - int(math.floor(self.get_tiled_width(ui) / 2))

        # extend text area by redefining viewport width
        self.width = int(math.floor(ui.tile_width * self.get_tiled_width(ui)))

    def get_tiled_height(self, ui):
        return int(math.ceil(self.height / ui.tile_height))

    def get_tiled_width(self, ui):
        return math.floor(self.width / ui.tile_width)

    def items_in_view_by_row(self, world, ui):
        # create smaller world within bounding box of our viewport
        items_by_yx = self.reinit_small_world(world, ui)

        # cast 'field of view' from small_world
        if ui.radius:
            visible = self.do_fov(player=world.player, ui=ui)

        def make_void(y, x):
            return Item.create_void(pos=Position(y, x))

        for y in range(self.y, self.y + self.get_tiled_height(ui)):
            yield [
                (items_by_yx.get((y, x)) if not ui.radius or (y, x) in visible
                 else [make_void(y, x)]) or [make_void(y, x)]
                for x in range(self.x, self.x + self.get_tiled_width(ui))
            ]

    def reinit_small_world(self, world, ui):
        # find the ideal tile_size,
        y_min, y_max = self.y, self.y + self.get_tiled_height(ui)
        x_min, x_max = self.x, self.x + self.get_tiled_width(ui)
        def fn_culling(item):
            return (x_min <= item.x < x_max and y_min <= item.y < y_max)

        # sort by top-most visible item
        def sort_func(item):
            return world.Where[item.where].value

        occlusions = collections.defaultdict(list)
        # XXX this is probably the most expensive lookup, of ~65,500 items
        # how much faster would it be if we used SQLITE, which had optimized
        # indicies for (X, Y) lookups, or is there a python equivalent we
        # aren't using? and, we should drop 'z', and index everything by
        # (Y, X) naturally with values of items.
        for item in sorted(
                filter(fn_culling, world.items),
                key=sort_func, reverse=False):
            occlusions[(item.y, item.x)].append(item)

        # this creates a small world as a side-effect, but, it is useful
        # for many operations, like opening a door or something, to see
        # in a smaller list of items whether a door is at that position.
        self.small_world = World(items=flatten(occlusions.values()))

        # optimized lookup indexed by (y, x)
        return occlusions

    def do_fov(self, player, ui):
        # start with the 8 octants, and cast light in each direction,
        # recursively sub-dividing remaining quadrants, cancelling
        # quadrants behind shadows, and marking 'visible'
        visible = {(player.y, player.x)}
        for oct in range(8):
            visible.update(
                self.cast_light(
                    cx=player.x,
                    cy=player.y,
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
        return visible

    def cast_light(self, cx, cy, row, start, end, radius, xx, xy, yx, yy, depth):
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
                    if self.small_world.light_blocked(Position(Y, X)):
                        new_start = r_slope
                    else:
                        blocked = False
                        start = new_start
                    continue
                if self.small_world.light_blocked(Position(Y, X)) and j < radius:
                    # This is a blocking square, start a child scan:
                    blocked = True
                    visible.update(
                        self.cast_light(
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


def _loop(ui, world, viewport):
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

    time_render = time_action = time_input = time_stats = lambda: 0

    time_render = 0
    # cause very first key input to have a timeout of nearly 0
    first_tick = 0.0001
    while True:
        with elapsed_timer() as time_stats:
            ui.render_text(
                viewport,
                debug_details={
                    "ms-render-world": int(time_render * 1000),
                    "ms-action": int(time_action() * 1000),
                    "ms-input": int(time_input() * 1000),
                    "ms-stats": int(time_stats() * 1000),
                    # of "whole world"
                    **world.debug_details(pos=world.player.pos),
                    # details of "small world"
                    **viewport.small_world.debug_details(
                        pos=world.player.pos, small_world=True
                    ),
                    **ui.debug_details(),
                },
            )

        with elapsed_timer() as time_render:
            ui.render(world, viewport)
        time_render = time_render()

        if first_tick:
            # skip waiting for input after first tick
            first_tick = 0
        else:
            with elapsed_timer() as time_input:
                inp = ui.reader(timeout=max(0, TIME_ANIMATION_TICK))
                first_tick = 0
                # throw away remaining input, small hack for
                # games where folks bang on the keys to run
                # (or boat!) as fast as they can, take "out"
                # all keys, then push back in the last-most
                # key.
                if inp:
                    save_key = None
                    while inp2 := ui.term.inkey(timeout=0):
                        save_key = inp2
                    if save_key and save_key != inp:
                        ui.term.ungetch(save_key)

        with elapsed_timer() as time_action:
            world = ui.reactor(inp, world, viewport)

def init_begin_world():
    # a small optimization, global world data is carried
    # over for each exit/entry into other worlds on subsequent load(),
    FPATH_WORLD_YAML = os.path.join(os.path.dirname(__file__), "dat", "world.yaml")
    world_data = yaml.load(open(FPATH_WORLD_YAML, "r"), Loader=yaml.SafeLoader)
    world = World.load(map_id=-1, world_data=world_data)

    # Add test boat!
    world.items.append(Item.create_boat(Position(y=110, x=86)))
    return world

    # and a horse, balloon, whirlpool,


def main():
    # a ui provides i/o, keyboard input and screen output
    ui = UInterface()

    world = init_begin_world()
    viewport = Viewport.create(world, ui)
    ui.auto_resize(viewport)
    ui.maybe_draw_viewport(viewport)
    with ui.activate():
        _loop(ui, world, viewport)


if __name__ == "__main__":
    exit(main())


# graphics improvements todo,
# - scale text character sets, so 16x+ get matching "large text"
#   u4 original had aprox. 15 characters wide
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
# - with large font letters, spell "FOOD" but
#   use top-only cells for health pct. bar cells
#   below it, need only <, '=' of green, '=' of
#   or missing '>', and the ability to change the
#   color of the text would be nice ..