import collections
import contextlib
import functools
import random
import enum

import blessed
import yaml

echo = functools.partial(print, end='')
# goal:
#
# - trees/saplings become sticks
# - bundle sticks
# - start fire
#
# - fell tree,
# - cut logs,
# - split wood,
# - start furnace

Position = collections.namedtuple('Position', (
    'z', 'y', 'x'))


class Item(object):
    def __init__(self, pos, char, name, material, weight, where, color):
        self.pos = pos
        self.char = char
        self.name = name
        self.material = material
        self.weight = weight
        self.where = where
        self.color = color

    @property
    def z(self):
        return self.pos.z

    @property
    def y(self):
        return self.pos.y

    @property
    def x(self):
        return self.pos.x

    def __repr__(self):
        return ('{self.name}<{self.where} {self.material} at '
                '{self.pos.z},{self.pos.y},{self.pos.x}>'
                .format(self=self))

    @classmethod
    def create(cls, pos, char, name, material, weight_range, where, color):
        weight = random.uniform(*weight_range)
        return cls(pos, char, name, material, weight, where, color)

    @classmethod
    def create_void(cls, pos, char='#'):
        "Create an item that represents void, charted space."
        return cls.create(pos=pos, char=char,
                          name='void', material='liquid',
                          weight_range=(0, 0),
                          where='unattached',
                          color='bold_black')


class World(object):
    z_bedrock = 2  # will be ~200 for sequoia trees later ..

    time = 4.50 # begin at 4:30am
    TICK = 0.1

    def __init__(self, Materials, Where, items):
        self.Where = Where
        self.Materials = Materials
        self.items = items

        # bounding dimensions
        self._height = max(item.pos.y for item in items)
        self._width = max(item.pos.x for item in items)
        self._depth = max(item.pos.z for item in items)

        # cache lookup
        self._player = None

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    def __repr__(self):
        return repr(self.items)

    def find_iter(self, **kwargs):
        return (item for item in self.items
                if all(getattr(item, key) == value
                       for key, value in kwargs.items()))

    def find_one(self, **kwargs):
        try:
            return next(self.find_iter(**kwargs))
        except StopIteration:
            return None

    @property
    def player(self):
        if self._player is None:
            self._player = self.find_one(name='player')
        return self._player

    @classmethod
    def load(cls, worldfile, **kwargs):
        world_data = yaml.load(open(worldfile))
        return cls.create(
            materials=enum.Enum('Material', world_data['Materials']),
            where=enum.Enum('Where', world_data['Where']),
            worldgen=world_data['WorldGen'],
            **kwargs)

    @classmethod
    def create(cls, materials, where, worldgen, height, width):
        items = []

        coords = [(row, col) for row in range(height) for col in range(width)]

        # lay "bedrock", then soil, plant trees.
        z = cls.z_bedrock
        items.extend(cls._make_zlayer(z+2, coords, worldgen['stone']))
        items.extend(cls._make_zlayer(z+1, coords, worldgen['stone']))
        items.extend(cls._make_zlayer(z, coords, worldgen['soil']))
        items.extend(cls._make_zlayer(z, coords, worldgen['tree'], 14))
        items.append(Item.create(
            pos=Position(z, *random.choice(coords)),
            **worldgen['player']))

        return cls(materials, where, items)

    @staticmethod
    def _make_zlayer(z, coords, item_kwargs, rand_pct=None):
        if rand_pct is not None:
            coords = random.sample(coords, int(len(coords) * (rand_pct * .01)))

        result = []
        for (y, x) in coords:
            pos = Position(z=z, y=y, x=x)
            result.append(Item.create(pos, **item_kwargs))
        return result

    def do_move_player(self, y=0, x=0, z=0):
        pos = Position(z=self.player.pos.z + z,
                       y=self.player.pos.y + y,
                       x=self.player.pos.x + x)

        if not self.blocked(pos):
            # go ahead, tromp away.
            self.player.pos = pos
            self.time += self.TICK
            return True
        return False

    def blocked(self, pos):
        void = True
        for item in self.find_iter(z=pos.z, y=pos.y, x=pos.x):
            void = False
            if item.where in ('planted', 'buried'):
                return True
        if void:
            # nothing was found here
            return True
        return False

    def do(self, action, properties):
        if action == 'move_player':
            self.dirty = self.do_move_player(**properties)
        elif action == 'wait':
            assert self.do_move_player()
        else:
            raise TypeError('do: {0}{1}'.format(action, properties))


class Viewport(object):

    MULT = collections.namedtuple('fastmath_table', [
        'xx', 'xy', 'yx', 'yy'])(xx=[1,  0,  0, -1, -1,  0,  0,  1],
                                 xy=[0,  1, -1,  0,  0, -1,  1,  0],
                                 yx=[0,  1,  1,  0,  0, -1, -1,  0],
                                 yy=[1,  0,  0,  1, -1,  0,  0, -1])

    def __init__(self, z, y, x, height, width):
        self.z, self.y, self.x, = z, y, x
        self.height, self.width = height, width

    def __repr__(self):
        return '{self.z}, {self.y}, {self.x}'.format(self=self)

    @classmethod
    def create(cls, world, term, width=40, height=20, _yoff=1, _xoff=2):
        "Create viewport instance centered one z-level above player."
        height = 20 # min(height, max(1, (term.height - (_yoff * 2))))
        width = 40 #max(width, max(1, (term.width - (_xoff * 2))))
        z = world.player.pos.z - 1
        y = world.player.pos.y - (height // 2)
        x = world.player.pos.x - (width // 2)
        return cls(z, y, x, height, width)

    def calculate_view(self, world):
        radius = None
        y_min, y_max = self.y, self.y + self.height
        x_min, x_max = self.x, self.x + self.width
        radius = 7

        #if 4 < world.time <= 7:
        #    # 4am -> 7am: dawn
        #    radius = 3 + int((world.time - 4) * 1.3)
        #    #_timestr = 'dawn'
        #elif 17 < world.time <= 20:
        #    # 5pm -> 7pm: dusk
        #    radius = 3 + int((20 - world.time) * 1.3)
        if radius:
            y_min = world.player.y - radius
            y_max = world.player.y + radius
            x_min = world.player.x - radius
            x_max = world.player.x + radius

        def calc_visible(item):
            # may be seen from above, "eagle's eye", and is within
            # radius squares bounding box.
            return (self.z < item.pos.z and
                    x_min <= item.pos.x < x_max and
                    y_min <= item.pos.y < y_max)

        # now, select the top-most visible item
        def sort_func(item):
            return item.pos.z, world.Where[item.where].value
        occlusions = dict()
        for item in sorted(filter(calc_visible, world.items),
                           key=sort_func, reverse=True):
            occlusions[(item.pos.y, item.pos.x)] = item

        if radius:
            small_world = World(Materials=world.Materials,
                                Where=world.Where,
                                items=list(occlusions.values()))

            self._visible = set()
            for oct in range(8):
                self._cast_light(
                    world=small_world, z=world.player.z,
                    cx=world.player.x, cy=world.player.y,
                    row=1, start=1.0, end=0.0, radius=radius,
                    xx=self.MULT.xx[oct], xy=self.MULT.xy[oct],
                    yx=self.MULT.yx[oct], yy=self.MULT.yy[oct],
                    depth=0)

        for y in range(self.y, self.y + self.height):
            candidate_row_items = [
                occlusions.get((y, x), Item.create_void(
                    pos=Position(self.z, y, x), char='-'))
                for x in range(self.x, self.x + self.width)]

            yield [item if (radius is None or
                            (item.pos.y, item.pos.x) in self._visible
                            or item == world.player) else
                   Item.create_void(pos=Position(self.z, item.y, item.x),
                                    char=' ')
                   for item in candidate_row_items]

    def _blocked(self, world, x, y, z):
        return world.blocked(Position(z, y, x))
        return (x < 0 or y < 0
                or x >= self.width or y >= self.height
                or self.data[y][x] == "#")

    def _cast_light(self, world, z, cx, cy, row, start, end, radius,
                    xx, xy, yx, yy, depth):
        "Recursive lightcasting function"
        if start < end:
            return
        radius_squared = radius*radius
        for j in range(row, radius+1):
            dx, dy = -j-1, -j
            blocked = False
            while dx <= 0:
                dx += 1
                # Translate the dx, dy coordinates into map coordinates:
                X, Y = cx + dx * xx + dy * xy, cy + dx * yx + dy * yy
                # l_slope and r_slope store the slopes of the left and right
                # extremities of the square we're considering:
                l_slope, r_slope = (dx-0.5)/(dy+0.5), (dx+0.5)/(dy-0.5)
                if start < r_slope:
                    continue
                elif end > l_slope:
                    break
                else:
                    # Our light beam is touching this square; light it,
                    # enforcing a 2:3 aspect ratio
                    if ((dx*dx + dy*dy) < radius_squared and
                            abs(dx * yx + dy * yy) < radius * (2 / 3)):
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
                            self._cast_light(world, z, cx, cy, j+1,
                                             start, l_slope, radius,
                                             xx, xy, yx, yy, depth+1)
                            new_start = r_slope
            # Row is scanned; do next row unless last square was blocked:
            if blocked:
                break

    def do_fov(self, x, y, radius):
        "Calculate lit squares from the given location and radius"
        self.flag += 1
        for oct in range(8):
            self._cast_light(x, y, 1, 1.0, 0.0, radius,
                             self.mult[0][oct], self.mult[1][oct],
                             self.mult[2][oct], self.mult[3][oct], 0)


class UInterface(object):

    def __init__(self):
        self.term = blessed.Terminal()
        self.dirty = True
        self._dimensions = (self.term.height, self.term.width)

    @property
    def dimensions(self):
        return (self.term.height, self.term.width)

    def reader(self):
        return self.term.inkey(timeout=1)

    def reactor(self, world, inp):
        if not inp:
            return
        if inp in 'hjklyubn':
            self.dirty = True
            yield ('move_player', {
                'h': {'x': -1},
                'j': {'y': 1},
                'k': {'y': -1},
                'l': {'x': 1},
                'y': {'y': -1, 'x': -1},
                'u': {'y': -1, 'x': 1},
                'b': {'y': 1, 'x': -1},
                'n': {'y': 1, 'x': 1},
            }[inp])
        elif inp in '.':
            self.dirty = True
            yield ('wait', {})

    @contextlib.contextmanager
    def activate(self):
        with self.term.fullscreen(), self.term.keypad(), self.term.cbreak():
            load_txt = 'loading ...'
            ypos = self.term.height // 2
            xpos = self.term.width // 2 - len(load_txt)
            echo(self.term.move(ypos, xpos) + load_txt, flush=True)
            yield

    def render(self, world):
        self.dirty = False

        # C64/spectrum-like border
        _yoff, _xoff = 1, 2
        viewport = Viewport.create(world, self.term, _yoff, _xoff)
        rows = viewport.calculate_view(world)

        with self.term.hidden_cursor(), self.term.location(0, 0):
            self.draw_decoration(viewport, world)
        for ypos, cell_items in enumerate(rows):
            echo(self.term.move(ypos + _yoff, _xoff))
            echo(u''.join([getattr(self.term, item.color)(item.char)
                           for item in cell_items]))
        echo('', flush=True)

    def draw_decoration(self, viewport, world, _yoff=1, _xoff=2):
        #if 8 <= world.time <= 18:
        #    border_color = self.term.green_reverse
        #elif 5 <= world.time <= 20:
        border_color = self.term.yellow_reverse
        #else:
        #    border_color = self.term.red_reverse
        echo(self.term.home)
        echo(border_color(' ' * self.term.width) * _yoff)
        for ypos in range(viewport.height):
            echo(self.term.move(_yoff + ypos, 0))
            echo(border_color(' ' * _xoff))
            echo(self.term.move(_yoff + ypos, _xoff + viewport.width))
            echo(border_color(' ' * _xoff))
            echo(self.term.move(_yoff + ypos, self.term.width - _xoff))
            echo(border_color(' ' * _xoff))
        echo(border_color(' ' * self.term.width) * _yoff, flush=True)

def _loop(ui, world):

    _dimensions = ui.dimensions
    echo(ui.term.clear)

    while True:
        if ui.dirty:
            ui.render(world)

        inp = ui.reader()

        for action, properties in ui.reactor(world, inp):
            world.do(action, properties)

        ui.dirty = ui.dirty or _dimensions != UInterface.dimensions


def main(height=50, width=100, worldfile='dat/world.yaml'):
    ui = UInterface()
    with ui.activate():
        world = World.load(worldfile, height=height, width=width)
        _loop(ui, world)


if __name__ == '__main__':
    exit(main())