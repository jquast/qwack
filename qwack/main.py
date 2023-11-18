#!/usr/bin/env python
import collections
import contextlib
import functools
import enum
import time
import timeit
import subprocess
import os

# 3rd party
import blessed
import yaml

echo = functools.partial(print, end="")
Position = collections.namedtuple("Position", ("z", "y", "x"))

FPATH_WORLD_MAP = os.path.join(os.path.dirname(__file__), "WORLD.MAP")
FPATH_WORLD_YAML = os.path.join(os.path.dirname(__file__), "world.yaml")
CHAFA_BIN = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "chafa", "tools", "chafa", "chafa"
)
CHAFA_EXTRA_ARGS = ["-w", "1", "-O", "1"]


@contextlib.contextmanager
def elapsed_timer():
    """Timer pattern, from https://stackoverflow.com/a/30024601."""
    start = timeit.default_timer()

    def elapser():
        return timeit.default_timer() - start

    # pylint: disable=unnecessary-lambda
    yield lambda: elapser()


class Item(object):
    def __init__(self, tile_id, pos, name, material, where, visibility):
        self.tile_id = tile_id
        self.name = name
        self.material = material
        self.where = where
        self._pos = pos
        self.visibility = visibility

    @property
    def pos(self):
        return self._pos

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
            f"{self.name}<{self.where} {self.material} at "
            f"z={self.z},y={self.y},x={self.x}>"
        )

    def __str__(self):
        return f"{self.name}, {self.where} {self.material}"

    @classmethod
    def create(cls, tile_id, pos, name, material, where, visibility):
        return cls(tile_id, pos, name, material, where, visibility)

    @classmethod
    def create_void(cls, pos):
        "Create an item that represents void, black space."
        return cls.create(
            tile_id=-1,
            pos=pos,
            name="void",
            material="liquid",
            where="buried",
            visibility=0,
        )


class World(object):
    z_bedrock = 2

    time = 4.50
    TICK = 0.1
    clipping = False

    def __init__(self, Materials, Where, items):
        self.Where = Where
        self.Materials = Materials
        self.items = items

        # bounding dimensions
        self._height = max(item.y for item in items)
        self._width = max(item.x for item in items)
        self._depth = max(item.z for item in items)

        # cache lookup
        self._player = None

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
    def load(cls):
        # create from u4 map
        world_data = yaml.load(open(FPATH_WORLD_YAML, "r"), Loader=yaml.SafeLoader)
        world_map = world_data["WorldMap"]
        items = []
        for (chunk_y, chunk_x), chunk_data in cls.read_u4_world_chunks().items():
            for idx, raw_val in enumerate(chunk_data):
                div_y, div_x = divmod(idx, 32)
                pos = Position(z=0, y=(chunk_y * 32) + div_y, x=(chunk_x * 32) + div_x)
                for where_type, definition in world_map[raw_val].items():
                    item = Item(
                        tile_id=raw_val,
                        pos=pos,
                        name=definition.get("name", None),
                        material=definition.get("material", None),
                        visibility=definition.get("visibility", 2),
                        where=where_type,
                    )
                    items.append(item)

        # create the player
        player_pos = Position(z=0, y=107, x=86)
        items.append(Item.create(pos=player_pos, **world_data["Player"]))

        return cls(
            Materials=enum.Enum("Material", world_data["Materials"]),
            Where=enum.Enum("Where", world_data["Where"]),
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

    def do_move_player(self, y=0, x=0, z=0):
        pos = Position(z=self.player.z + z, y=self.player.y + y, x=self.player.x + x)

        moved = False
        if not self.clipping or not self.blocked(pos):
            # allow player to move to given position.
            self.player.pos = pos
            moved = True

        self.time += self.TICK
        return moved

    def blocked(self, pos):
        is_void = True
        for item in self.find_iter(z=pos.z, y=pos.y, x=pos.x):
            is_void = False
            if item.visibility < 2:
                return True
        return is_void

    def do(self, action, properties):
        if action == "do_move_player":
            self.dirty = self.do_move_player(**properties)
        elif action == "toggle_clipping":
            self.clipping = not self.clipping
        else:
            raise TypeError(f"do: {action}{properties}")


class UInterface:
    movement_map = {
        # given input key, move x/y/z coords given
        "h": {"x": -1},
        "H": {"x": -5},
        "j": {"y": 1},
        "J": {"y": 5},
        "k": {"y": -1},
        "K": {"y": -5},
        "l": {"x": 1},
        "L": {"x": 5},
        "y": {"y": -1, "x": -1},
        "u": {"y": -1, "x": 1},
        "b": {"y": 1, "x": -1},
        "n": {"y": 1, "x": 1},
        "<": {"z": -1},
        ">": {"z": 1},
    }

    def __init__(self, tile_size):
        self.term = blessed.Terminal()
        self.dirty = True
        self.radius = 10
        assert tile_size in (2, 4, 8, 16, 32)
        self.tile_size = self.tile_width = tile_size
        self.tile_height = len(UInterface.get_tile(tile_id=0, tile_size=tile_size))

    @property
    def window_size(self):
        return (self.term.height, self.term.width)

    def reader(self, timeout):
        return self.term.inkey(timeout=timeout)

    def reactor(self, world, inp):
        if not inp:
            yield ("do_move_player", {})
            return
        if inp in self.movement_map:
            self.dirty = True
            yield ("do_move_player", self.movement_map[inp])
        elif inp == "c":
            yield ("toggle_clipping", {})
        elif inp == "r":
            self.radius = 7 if not self.radius else None
        elif inp == "+" and self.radius is not None and self.radius <= 10:
            self.radius += 1
        elif inp == "-" and self.radius is not None and self.radius >= 1:
            self.radius -= 1

    @contextlib.contextmanager
    def activate(self):
        with self.term.fullscreen(), self.term.keypad(), self.term.cbreak():
            self.clear()
            yield self

    def clear(self):
        echo(self.term.clear)

    def render(self, world):
        with self.term.hidden_cursor(), self.term.location(
            0, 0
        ), elapsed_timer() as debug_elapsed:
            _yoff, _xoff = 1, 2
            self.dirty = False
            # todo: cache Viewport !
            viewport = Viewport.create(world, self, self.term, _yoff, _xoff)

            self.draw_decoration(viewport, world)

            txt_rows = viewport.calculate_view(world, self)
            for ypos, cell_items in enumerate(txt_rows):
                ypos *= self.tile_height
                for xpos, item in enumerate(cell_items):
                    xpos *= self.tile_width
                    actual_xpos = xpos + _xoff
                    tile_ans = UInterface.get_tile(item.tile_id, self.tile_size)
                    for ans_y, ans_txt in enumerate(tile_ans):
                        actual_ypos = ypos + ans_y + _yoff
                        if actual_ypos <= viewport.height:
                            echo(self.term.move_yx(actual_ypos, actual_xpos))
                            echo(ans_txt)

            for ypos, xpos, txt_status in self.generate_status(world, debug_elapsed):
                # display text left-of viewport
                xpos += _xoff + viewport.width + _xoff
                ljust_width = self.term.width - (viewport.width + 5 + _xoff)
                disp = txt_status.ljust(ljust_width, " ")
                echo(self.term.move(ypos + _yoff, xpos) + disp)

        echo("", flush=True)

    @functools.lru_cache(maxsize=256)
    @staticmethod
    def get_tile(tile_id, tile_size):
        if tile_id == -1:
            return [" " * tile_size] * tile_size
        # TODO: add horizontal and vertical trimmings, so that we can "fill" to the right,
        # but also, we can center, and do a "smooth scrolling" effect with those offset tiles,
        fpath_png = os.path.join(
            os.path.dirname(__file__), "tiles", f"tile_{tile_id:02X}.png"
        )
        ans = subprocess.check_output(
            [
                CHAFA_BIN,
                *CHAFA_EXTRA_ARGS,
                "--size",
                f"{tile_size}x{tile_size}",
                fpath_png,
            ]
        ).decode()
        lines = ans.splitlines()
        # remove any hide/show cursor attributes
        lines[0] = lines[0][len("\x1b[?25l\x1b[0m") :]
        del lines[-1]
        return lines

    def generate_status(self, world, debug_elapsed):
        # TODO: return fixed-size array ! let caller do layout!
        yield (1, 1, f"z={world.player.z} y={world.player.y} x={world.player.x}")
        yield (3, 1, f"time {world.time:2.2f}")
        # endpos = 0
        # items = list(world.find_iter(z=world.player.z, y=world.player.y, x=world.player.x))
        # if len(items) > 1:
        #    assert False, items
        endpos = 0
        for endpos, item in enumerate(
            world.find_iter(z=world.player.z, y=world.player.y, x=world.player.x)
        ):
            yield (5 + (endpos * 2), 1, "- " + item.__str__())

        endpos += 1
        yield (5 + (endpos * 2), 1, f"render {debug_elapsed():2.2f}")
        endpos += 1
        yield (5 + (endpos * 2), 1, f"--")
        endpos += 1
        yield (5 + (endpos * 2), 1, f"--")
        endpos += 1
        yield (5 + (endpos * 2), 1, f"--")

    def draw_decoration(self, viewport, world, _yoff=1, _xoff=2):
        # C64/spectrum-like border
        border_color = self.term.yellow_reverse
        echo(self.term.home)
        echo(border_color(" " * self.term.width) * _yoff)
        for ypos in range(viewport.height):
            echo(self.term.move(_yoff + ypos, 0))
            echo(border_color(" " * _xoff))
            echo(self.term.move(_yoff + ypos, _xoff + viewport.width))
            echo(border_color(" " * _xoff))
            echo(self.term.move(_yoff + ypos, self.term.width - _xoff))
            echo(border_color(" " * _xoff))
        echo(border_color(" " * self.term.width) * _yoff, flush=True)


class Viewport(object):
    MULT = collections.namedtuple("fastmath_table", ["xx", "xy", "yx", "yy"])(
        xx=[1, 0, 0, -1, -1, 0, 0, 1],
        xy=[0, 1, -1, 0, 0, -1, 1, 0],
        yx=[0, 1, 1, 0, 0, -1, -1, 0],
        yy=[1, 0, 0, 1, -1, 0, 0, -1],
    )

    def __init__(self, z, y, x, height, width):
        (self.z, self.y, self.x) = (z, y, x)
        self.height, self.width = height, width

    def __repr__(self):
        return f"{self.z}, {self.y}, {self.x}"

    @classmethod
    def create(cls, world, ui, term, width=40, height=20, _yoff=1, _xoff=2):
        "Create viewport instance centered one z-level above player."
        height = term.height - (_yoff * 2)
        width = term.width - 40
        tiled_height = (height // ui.tile_height) + 1
        tiled_width = width // ui.tile_width
        z = world.player.z - 1
        y = world.player.y - (tiled_height // 2)
        x = world.player.x - (tiled_width // 2)
        return cls(z, y, x, height, width)

    def calculate_view(self, world, ui):
        tiled_height = (self.height // ui.tile_height) + 1
        tiled_width = self.width // ui.tile_width
        y_min, y_max = self.y, self.y + tiled_height
        x_min, x_max = self.x, self.x + tiled_width

        def calc_visible(item):
            # rename to culling
            # may be seen from above, "eagle's eye", and is within
            # radius squares bounding box.
            return (
                self.z < item.z and x_min <= item.x < x_max and y_min <= item.y < y_max
            )

        # now, select the top-most visible item
        def sort_func(item):
            return item.z, world.Where[item.where].value

        occlusions = dict()
        for item in sorted(
            filter(calc_visible, world.items), key=sort_func, reverse=True
        ):
            occlusions[(item.y, item.x)] = item

        self._visible = set()
        if ui.radius:
            small_world = World(
                Materials=world.Materials,
                Where=world.Where,
                items=list(occlusions.values()),
            )
            for oct in range(8):
                self._cast_light(
                    world=small_world,
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

        for y in range(self.y, self.y + tiled_height):
            candidate_row_items = [
                occlusions.get((y, x), Item.create_void(pos=Position(self.z, y, x)))
                for x in range(self.x, self.x + tiled_width)
            ]

            yield [
                item
                if (
                    ui.radius is None
                    or (item.y, item.x) in self._visible
                    or item == world.player
                )
                else Item.create_void(pos=Position(self.z, item.y, item.x))
                for item in candidate_row_items
            ]

    # TODO: re-enable

    def _blocked(self, world, x, y, z):
        return world.blocked(Position(z, y, x))

    def _cast_light(
        self, world, z, cx, cy, row, start, end, radius, xx, xy, yx, yy, depth
    ):
        "Recursive lightcasting function"
        if start < end:
            return
        radius_squared = radius * radius
        for j in range(row, radius + 1):
            dx, dy = -j - 1, -j
            blocked = False
            while dx <= 0:
                dx += 1
                # Translate the dx, dy coordinates into map coordinates:
                X, Y = cx + dx * xx + dy * xy, cy + dx * yx + dy * yy
                # l_slope and r_slope store the slopes of the left and right
                # extremities of the square we're considering:
                l_slope, r_slope = (dx - 0.5) / (dy + 0.5), (dx + 0.5) / (dy - 0.5)
                if start < r_slope:
                    continue
                elif end > l_slope:
                    break
                else:
                    # Our light beam is touching this square; light it,
                    # enforcing a 2:3 aspect ratio
                    if (dx * dx + dy * dy) < radius_squared and abs(
                        dx * yx + dy * yy
                    ) < radius * (2 / 3):
                        self._visible.add((Y, X))
                    if blocked:
                        # we're scanning a row of blocked squares:
                        if self._blocked(world, X, Y, z):
                            new_start = r_slope
                            continue
                        else:
                            blocked = False
                            start = new_start
                    else:
                        if self._blocked(world, X, Y, z) and j < radius:
                            # This is a blocking square, start a child scan:
                            blocked = True
                            self._cast_light(
                                world,
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
                            new_start = r_slope
            # Row is scanned; do next row unless last square was blocked:
            if blocked:
                break

    def do_fov(self, x, y, radius):
        "Calculate lit squares from the given location and radius"
        self.flag += 1
        for oct in range(8):
            self._cast_light(
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


TIME_TICK = 0.50


def _loop(ui, world):
    while True:
        with elapsed_timer() as time_elapsed:
            ui.render(world)
        inp = ui.reader(timeout=max(0, TIME_TICK - time_elapsed()))

        for action, properties in ui.reactor(world, inp):
            world.do(action, properties)


def main():
    ui = UInterface(tile_size=16)
    with ui.activate():
        world = World.load()
        _loop(ui, world)


if __name__ == "__main__":
    exit(main())


# graphics improvements todo,
# - scale graphics, not just 16x but also 4x, 8x, even 32x
# - scale text character sets, so 16x+ get matching "large text"
# - trim tile edges to fill screen edge
# - shaded/dithered tiles for darkness like U6
# - smooth scrolling
# - animated tiles,
#   - dynamically shift X+Y every frame?
#   - wave flags of castles
#
# movement improvements
# - '_' go to function, pulls up map ..
# - use the "braille" for world map, maybe show & highlight relative position
