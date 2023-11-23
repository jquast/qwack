#!/usr/bin/env python
import collections
import contextlib
import textwrap
import functools
import enum
import timeit
import random
import math
import time
import os
import zipfile

# 3rd party
import blessed
import yaml

# local
from . import u4_data
from . import u4_tiler

echo = functools.partial(print, end="")
Position = collections.namedtuple("Position", ("y", "x"))


# todo: Vector? (direction)
def make_direction(y=0, x=0):
    text = []
    if y:
        text.append("North" if y < 0 else "South")
    if x:
        text.append("West" if x < 0 else "East")
    return " ".join(text)


# This was "font ratio", 3/2, but with tiles that have already been converted
# to their correct aspect ratio by CHAFA, '1' provides the best "circle" effect
DEFAULT_RADIUS = 6
VIS_RATIO = 1
TIME_ANIMATION_TICK = 0.20
TIME_PLAYER_PASS = 23
TEXT_HISTORY_LENGTH = 1000
DEFAULT_TILE_SIZE = 16
MAX_RADIUS = 9
LORD_BRITISH_CASTLE_ID = 14
# the tile size of our stored tiles/*.png files, they are 16x16
# natively but their resolution was doubled, it appears to give
# libchafa better hints about "shape" of these "blocky" tiles
SHAPE_TILE_SIZE = 32


# probably better in the YAML, but gosh, lots of junk in "World" ?
SHIP_TILE_DIRECTIONS = {16: "West", 17: "North", 18: "East", 19: "South"}
DIRECTION_SHIP_TILES = {v: k for k, v in SHIP_TILE_DIRECTIONS.items()}
DEFAULT_SHAPE_FILENAME = "jsteele-shapes.ega"

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

    def get_animation_y_offset(self, world):
        if self.tile_id in (0, 1, 2, 3, 68, 69, 70, 71):
            # water and fields are animated by vertical offset, this should
            # give 8 positions evenly spread across 32 pixels
            return 32 - (int(time.monotonic() * 100) % 8) * 4
        return 0

    def is_adjacent(self, other_item):
        # Check if the target coordinates are adjacent to the given coordinates
        return (
            abs(self.x - other_item.x) == 1
            and self.y == other_item.y
            or abs(self.y - other_item.y) == 1
            and self.x == other_item.x
        )

    def distance(self, other_item):
        return math.sqrt((self.x - other_item.x) ** 2 + (self.y - other_item.y) ** 2)

    @property
    def is_boat(self):
        return self.tile_id in (16, 17, 18, 19)

    @property
    def is_horse(self):
        return self.tile_id in (20, 21)

    @property
    def is_flying(self):
        return self.tile_id == 24

    @property
    def is_ladder_up(self):
        return self.tile_id == 27

    @property
    def is_ladder_down(self):
        return self.tile_id == 28


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
    wizard_mode = True
    # state of location in world, for entering/exiting towns
    world_y, world_x = 0, 0

    def __init__(
        self, world_0_items=None, items=None, Materials=None, Where=None, Portals=None, world_data=None,
        world_y=None, world_x=None,
    ):
        # this is just a basic copy of the mutated items in world_0 until we can find a save & restore
        # system for mutated world items, as maps will also need this for "S"earching and finding items
        # (by removing them from the map)
        self.world_0_items = world_0_items
        self.Where = Where
        self.Materials = Materials
        self.Portals = Portals
        self.items = items or []
        self.world_data = world_data
        self.world_y = world_y if world_y is not None else 0
        self.world_x = world_x if world_x is not None else 0

        # bounding dimensions
        self._height = max(item.y for item in self.items) if self.items else 0
        self._width = max(item.x for item in self.items) if self.items else 0

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
        return (item for item in self.find_iter(**kwargs) if item.name != "player")

    def find_one(self, **kwargs):
        try:
            return next(self.find_iter(**kwargs))
        except StopIteration:
            return None
    
    def find_one_not_player(self, **kwargs):
        try:
            return next(self.find_iter_not_player(**kwargs))
        except StopIteration:
            return None

    @property
    def player(self):
        if self._player is None:
            self._player = self.find_one(name="player")
        return self._player

    @classmethod
    def load(cls, world_0_items, map_id: int, world_data: dict, start_y=None, start_x=None, world_y=None, world_x=None):
        # create from u4 map, or persist between world exits, etc.
        if world_0_items is None:
            world_0_items = cls.make_tile_items(0, world_data)
            assert map_id == 0
            items = world_0_items
            start_y, start_x = world_y, world_x
            # start player @ world british's castle
            items.append(Item.create_player(Position(y=107, x=86)))
        elif map_id == 0:
            assert world_0_items
            items = world_0_items
        if map_id != 0:
            assert None not in (start_y, start_x)
            # load cities etc. always from file
            items = cls.make_tile_items(map_id, world_data)
            items.append(Item.create_player(Position(y=start_y, x=start_x)))
            # add npc's
            for npc_definition in u4_data.load_npcs_from_u4_ult_map(map_id, world_data):
                items.append(Item(**npc_definition))
        assert items

        return cls(
            Materials=enum.Enum("Material", world_data["Materials"]),
            Where=enum.Enum("Where", world_data["Where"]),
            Portals=world_data["World"]["Portals"],
            items=items,
            world_data=world_data,
            world_y=world_y,
            world_x=world_x,
            world_0_items=world_0_items,
        )

    @classmethod
    def make_tile_items(cls, map_id: int, world_data):
        # map_id of '0' means world map, which has a chunk_size of 32x32,
        # we don't have concern for apple ][ memory restrictions and load
        # the entire world, anyway.
        items = []
        chunk_size = 32 if map_id == 0 else 1
        map_chunks = u4_data.read_map(map_id)
        for (chunk_y, chunk_x), chunk_data in map_chunks.items():
            for idx, raw_val in enumerate(chunk_data):
                div_y, div_x = divmod(idx, chunk_size)
                pos = Position(
                    y=(chunk_y * chunk_size) + div_y,
                    x=(chunk_x * chunk_size) + div_x,
                )
                tile_definition = world_data["Shapes"][raw_val]
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

    def do_move_player(self, viewport, y=0, x=0) -> tuple[bool, bool]:
        # Returns whether the player moved, and screen should be refreshed,
        # and, whether the player exited the map (dirty, do_exit)
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
            viewport.add_text(make_direction(y=y, x=x))
            self.player.pos = pos
        do_exit = not viewport.small_world.find_one_not_player(pos=pos)
        player_moved = pos != previous_pos
        return player_moved, do_exit

    def do_open_door(self, viewport, y=0, x=0):
        # is there an (unlocked) door ?
        door = self.find_one(y=self.player.y + y, x=self.player.x + x, tile_id=59)
        if door:
            # then set it open!
            door.tile_id = 62
            door.last_action_tick = self.time
            door.land_passable = True
            door.darkness = 0
            return True
        else:
            viewport.add_text_append(f"{y or ''}{x or ''}")
            viewport.add_text("NOT HERE!")
        return False

    def board_ship_or_mount_horse(self, viewport):
        boat = self.find_one(name="boat", pos=self.player.pos)
        if not boat and self.wizard_mode:
            # wizards can "Board" any tile, LoL!
            items = list(self.find_iter_not_player(pos=self.player.pos))
            boat = items[-1]
        elif not boat:
            viewport.add_text("Board WHAT?")
            return False
        self.player.tile_id = boat.tile_id
        self.items.remove(boat)
        return True

    def exit_ship_or_unmount_horse(self, viewport):
        if not self.player.is_boat and not self.wizard_mode:
            viewport.add_text("Not HERE!")  # XXX Check
            return False
        boat = Item.create_boat(self.player.pos, self.player.tile_id)
        self.items.append(boat)
        self.player.tile_id = Item.DEFAULT_PLAYER_TILE_ID
        return True

    def check_boat_direction(self, y, x):
        boat_direction = SHIP_TILE_DIRECTIONS.get(self.player.tile_id)
        # is the boat facing the direction we want to move?
        can_move = boat_direction in make_direction(y=y, x=x).split()
        # turn towards the direction whether we can_move or not
        next_direction = make_direction(y, x).split(" ", 1)[0]
        # conditionally set boat direction
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
                    return {
                        "dest_id": portal["dest_id"],
                        "start_x": portal["start_x"],
                        "start_y": portal["start_y"],
                    }

    def check_tile_movement(self, pos) -> int:
        # if any tile at given location has a "speed" variable, then,
        # use as random "SLOW PROGRESS!" deterrent for difficult terrain

        # When travelling north, check if player is on Lord British's Castle
        # and Deny movement on any match
        if pos.y < self.player.y:
            for item in self.find_iter(
                y=self.player.y, x=self.player.x, tile_id=LORD_BRITISH_CASTLE_ID
            ):
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
        fn_trim = math.ceil if not random.randrange(12) else math.floor
        return fn_trim(min(max(0, distance - 2), u4_tiler.MAX_DARKNESS_LEVEL))

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
        # close door after 4 game ticks, a door is always named "Unlocked Door"
        # if it was "O"pened, but is temporarily with a different tile_id
        for door in small_world.find_iter(name="Unlocked Door"):
            if self.time > door.last_action_tick + 4:
                door.tile_id = 59
                door.land_passable = False
                door.darkness = 1
                return True
        return False


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

    # when defined, the monotonic time a user pressed "O"pen
    # and that we are awaiting a direction key (NSEW)
    waiting_open_direction = 0

    def __init__(self, ShapeFiles, shape_filename=DEFAULT_SHAPE_FILENAME, tile_size=DEFAULT_TILE_SIZE, darkness=True, radius=DEFAULT_RADIUS):
        self.term = blessed.Terminal()
        self.dirty = True
        self.radius = radius
        self.darkness = darkness
        # whether we are waiting for a direction key
        # after using "O"pen, there will probably be
        # a lot more of these, maybe a "store next action"
        # is needed -- this *seems* it should exist in the UI,
        # but maybe there needs to be another abstraction,
        # we will see..
        self.ShapeFiles = ShapeFiles
        # because this uses "magic" in setter, set only after shape_data
        # todo: move to u4_tiler.py as Shapes.py
        self.init_shape_data(shape_filename, tile_size)

    def init_shape_data(self, shape_filename, tile_size):
        self.shape_filename = shape_filename
        # update self.tile_size,
        self.tile_size = min(max(tile_size, u4_tiler.MIN_TILE_SIZE), u4_tiler.MAX_TILE_SIZE)
        for sf in self.ShapeFiles:
            if sf["filename"] == self.shape_filename:
                # then set shape_data,
                self.shape_data = self.load_txt_shape_data(sf)
                # and update tile_height from yaml data
                self.tile_width = tile_size
                self.tile_height = len(self.shape_data[0xFF][0]["value"])
                return
        raise ValueError(f"No matching records by filename={self.shape_filename!r}, ShapeFiles={self.ShapeFiles}")

    def load_txt_shape_data(self, sf):
        zipfile_name = f'{sf["name"]}.zip'
        zipfile_path = os.path.join(os.path.dirname(__file__), "dat", zipfile_name)
        yamlfile_name = f'{self.tile_size}.yaml'
                # extract and load yaml data from yamlfile_name inside zipfile_name in-memory,
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            return yaml.load(zip_ref.read(yamlfile_name), Loader=yaml.SafeLoader)
    
    def cycle_shapes(self):
        # given the current self.shape_filename, cycle to the next
        for idx in range(len(self.ShapeFiles)):
            if self.ShapeFiles[idx]["filename"] == self.shape_filename:
                shape_filename = self.ShapeFiles[(idx + 1) % len(self.ShapeFiles)]["filename"]
                self.init_shape_data(shape_filename, self.tile_size)
                return

    @property
    def window_size(self):
        return (self.term.height, self.term.width)

    def reader(self, timeout):
        return self.term.inkey(timeout=timeout)

    def reactor(self, inp, world, viewport):
        self.dirty = True
        if inp in self.movement_map:
            if self.waiting_open_direction:
                viewport.add_text_append(make_direction(**self.movement_map[inp]))
                self.dirty = world.do_open_door(viewport, **self.movement_map[inp])
                self.waiting_open_direction = 0
            else:
                self.dirty, do_exit = world.do_move_player(viewport, **self.movement_map[inp])
                if do_exit:
                    # TODO: track where we are! we can't exit the world map, either!
                    viewport.add_text("Leaving XXX!")
                    # all maps so far "exit" to the world map
                    world = World.load(
                        world_0_items=world.world_0_items,
                        map_id=0,
                        world_data=world.world_data,
                        start_y=world.world_y,
                        start_x=world.world_x,
                        world_y=world.world_y,
                        world_x=world.world_x,
                    )
        elif self.waiting_open_direction:
            # invalid direction after "O"pen
            if inp:
                viewport.add_text_append(f"{inp or ''}")
                viewport.add_text("NOT HERE!")
            self.waiting_open_direction = 0
        elif inp == "o":
            viewport.add_text("Open-")
            self.waiting_open_direction = time.monotonic()
        elif inp == "E" or inp == "K" or inp == "D":
            # 'E'nter Portal or "K"limb ladder
            portal = world.find_portal(world.player.pos)
            item = world.find_one_not_player(pos=world.player.pos)
            can_enter = (portal and
                         item.is_ladder_up and inp == "K" or
                         item.is_ladder_down and inp == "D" or
                         portal and inp == "E")
            if can_enter:
                map_name = world.world_data["Maps"][portal["dest_id"]]["name"]
                viewport.add_text(map_name.upper().center(15))
                viewport.dirty = True
                world = World.load(
                    world_0_items=world.items,
                    map_id=portal["dest_id"],
                    world_data=world.world_data,
                    start_x=portal["start_x"],
                    start_y=portal["start_y"],
                    world_y=world.world_y,
                    world_x=world.world_x,
                )
        elif inp == "B":
            # 'B'oard ship or mount horse
            self.dirty = world.board_ship_or_mount_horse(viewport)
        elif inp == "X":
            # e 'X'it ship or unmount horse
            self.dirty = world.exit_ship_or_unmount_horse(viewport)
        elif inp == "{" and self.tile_size > u4_tiler.MIN_TILE_SIZE:
            self.init_shape_data(self.shape_filename, self.tile_size - 1)
        elif inp == "}" and self.tile_size < u4_tiler.MAX_TILE_SIZE:
            self.init_shape_data(self.shape_filename, self.tile_size + 1)
        elif inp == "\x17":  # Control-W
            world.wizard_mode = not world.wizard_mode
            _enabled = {"enabled" if world.wizard_mode else "disabled"}
            viewport.add_text(f"Wizard mode {_enabled}")
        elif world.wizard_mode:
            # keys for wizards !
            if inp == "C":
                world.clipping = not world.clipping
                _enabled = {"enabled" if world.clipping else "disabled"}
                viewport.add_text(f"Clipping mode {_enabled}")
            elif inp == "A":
                self.auto_resize(viewport)
            elif inp == "R":
                self.radius = DEFAULT_RADIUS if not self.radius else None
            elif inp == "\x04":  # ^D
                self.darkness = not self.darkness
                _enabled = {"enabled" if world.darkness else "disabled"}
                viewport.add_text(f"Darkness {_enabled}")
            elif inp == "\x12":  # ^R
                self.cycle_shapes()
            elif inp == ")" and self.radius is not None and self.radius < MAX_RADIUS:
                self.radius += 1
            elif inp == "(" and self.radius is not None and self.radius >= 2:
                self.radius -= 1
        # even when we don't move, the world may forcefully tick!
        else:
            if time.monotonic() > world.time_monotonic_last_action + TIME_PLAYER_PASS:
                world.player.last_action_tick = world.time
            else:
                # return early
                self.dirty = False
                return world
        if self.dirty:
            # Ultima IV is cruel, if *anything* happens it drives
            # the game forward if it can!
            world.tick(viewport.small_world)
        return world

    def auto_resize(self, viewport):
        if self.radius:
            while self.tile_size > u4_tiler.MIN_TILE_SIZE and (
                ((self.radius) * 2) + 2 > (viewport.width / self.tile_size) - 1
            ):
                self.dirty = True
                self.tile_size -= 1
                viewport.add_text(
                    f"resize tile -1, ={self.tile_size}, "
                    f"viewport_width={viewport.width}, "
                    f"tile_width={self.tile_width}, "
                    f"tile_height={self.tile_height}, "
                    f"radius * 2={self.radius * 2}, "
                )
            while self.tile_size < u4_tiler.MAX_TILE_SIZE and (
                (self.radius * 2) + 2 < (viewport.width / self.tile_size)
            ):
                self.dirty = True
                self.tile_size += 1
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
        return {
            "tile-width": self.tile_width,
            "tile-height": self.tile_height,
            "radius": self.radius,
            "darkness": self.darkness,
            "term-height": self.term.height,
            "term-width": self.term.width,
            f"shape": self.shape_filename,
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
                        # todo: an ItemsCollection should have 'foreground' and 'background'
#                        bg_tile_id = None
#                        if len(items) > 1:
#                            bg_tile_id = items[-1].tile_id
                        tile_darkness = (
                            viewport.small_world.darkness(items[0])
                            if self.darkness
                            else 0
                        )
                        tile_ans = u4_tiler.get_ansi_txt_tile(
                            self.shape_data,
                            items[0].tile_id,
                            tile_width=self.tile_width,
                            tile_height=self.tile_height,
#                            # y_offset=items[-1].get_animation_y_offset(world),
                            darkness=tile_darkness,
#                            # bg_tile_id=bg_tile_id,
                        )
                        for ans_y, ans_txt in enumerate(tile_ans):
                            actual_ypos = ypos + ans_y + viewport.yoffset
                            if actual_ypos <= viewport.height:
                                echo(self.term.move_yx(actual_ypos, actual_xpos))
                                echo(ans_txt)
            echo("", flush=True)
            self.dirty = False


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
        self.dirty = True

    def add_text_append(self, text):
        # add text to the last line for continuation of paragraph,
        # actions to result, etc.
        self.text.append(self.text.pop() + text)
        self.dirty = True

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
                (
                    items_by_yx.get((y, x))
                    if not ui.radius or (y, x) in visible
                    else [make_void(y, x)]
                )
                or [make_void(y, x)]
                for x in range(self.x, self.x + self.get_tiled_width(ui))
            ]

    def reinit_small_world(self, world, ui):
        # find the ideal tile_size,
        y_min, y_max = self.y, self.y + self.get_tiled_height(ui)
        x_min, x_max = self.x, self.x + self.get_tiled_width(ui)

        def fn_culling(item):
            return x_min <= item.x < x_max and y_min <= item.y < y_max

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
            filter(fn_culling, world.items), key=sort_func, reverse=False
        ):
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


def init_begin_world(world_data):
    world = World.load(world_0_items=None, map_id=0, world_data=world_data)

    # Add test boat!
    world.items.append(Item.create_boat(Position(y=110, x=86)))
    return world

    # and a horse, balloon, whirlpool,


def main():
    # a small optimization, global world data is carried over for each
    # exit/entry into other worlds on subsequent load(), (..it could also
    # be refreshed automatically dynamically on each next world load)
    FPATH_WORLD_YAML = os.path.join(os.path.dirname(__file__), "dat", "world.yaml")
    world_data = yaml.load(open(FPATH_WORLD_YAML, "r"), Loader=yaml.SafeLoader)

    # a ui provides i/o, keyboard input and screen output, world.yaml now
    # stores non-world data, might as well be renamed into config_data.yaml
    # or such
    ui = UInterface(ShapeFiles=world_data["ShapeFiles"])
    world = init_begin_world(world_data)
    viewport = Viewport.create(world, ui)
    ui.auto_resize(viewport)
    ui.maybe_draw_viewport(viewport)
    with ui.activate():
        _loop(ui, world, viewport)


if __name__ == "__main__":
    exit(main())


# graphics improvements todo,
# - carve up & chafa to make glyph tiles for the screen borders
#   using josh steele's xu4 screen asset, this would allow us to
#   scale to many many resolutions to retain aspect ratio
# - intro screen & menu ?!
# - what about the "scene" that is played, too??
# - animated tiles,
#   - wave flags of castles somehow it is done..
# - water animations are actually really smooth on a2,
#   they move one pixel at a time! so maybe
#   y_offset = world.time % (tile_height // 2)
# - scale text character sets, so 16x+ get matching "large text"
#   u4 original had aprox. 15 characters wide "text area"
#   load_shapes_png
# - if re-implemented crop, smooth scrolling is possible
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
#
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
# - get_tile x_offset and y_offset can be used for much
#   more than seawaves, it can be used with random() jiter
#   for when under confusion, or maybe when hungry, nice!
# - implement nethack's hallucinations, with graphics effects
#   and crazy tile switching, do a bit better in letting the
#   hallucination persist with some random time, instead
#   of changing each frame.
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
#
#
# the way a tile can be labelled as a 'Boat' and flown as a wizard
# mechanic, this could be used for real in-game mechanic, of a
# "magic carpet" or similar (!)
#
#
# Today,
#
# - pre-render all ansi! persist to disk! removes chafa library need at runtime !!
#   - it would have to be all tiles
#     - at all resolutions
#     - at all darkness levels
#     - at all y-offset animations
#   - packed up into structures organized by
#     - size as filename
#     - contents as yaml dictionary
#     - keyed by tile_id
#     - results as array of ansi strings
#
#   (!)
#
# fix directionals, allow *both* ??
# - hjkl + yubn (already done)
# - wasd + qezc
# - numpad (by using application keys mode)
# with exception of common actions, all of them should be then upper case
# lowercase;
# - 'o'pen, 
# - 'b'oard,
# - 'x'it,
# - 'p'eer,
# uppercase to avoid conflict:
#
# 'E'nter, 'C'ast, 'K'limb, 'D'escend, 
