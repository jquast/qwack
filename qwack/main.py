#!/usr/bin/env python
import collections
import contextlib
import textwrap
import functools
import timeit
import random
import math
import threading
import time
import os

# 3rd party
import blessed
import yaml

# local
from qwack import u4_data
from qwack import u4_tiler
from qwack import models

echo = functools.partial(print, end="")

# todo: Vector? (direction)
def make_direction(y=0, x=0):
    text = []
    if y < 0:
        text.append("North")
    elif y > 0:
        text.append("South")
    if x < 0:
        text.append("West")
    elif x > 0:
        text.append("East")
    return " ".join(text)


# This was "font ratio", 3/2, but with tiles that have already been converted
# to their correct aspect ratio by CHAFA, '1' provides the best "circle" effect
VIS_RATIO = 1
TIME_ANIMATION_TICK = 0.20
TIME_PLAYER_PASS = 5
TEXT_HISTORY_LENGTH = 1000
MAX_RADIUS = 15
MIN_RADIUS = 2
LORD_BRITISH_CASTLE_ID = 14

# probably better in the YAML, but gosh, lots of junk in "World" ?
SHIP_TILE_DIRECTIONS = {16: "West", 17: "North", 18: "East", 19: "South"}
DIRECTION_SHIP_TILES = {v: k for k, v in SHIP_TILE_DIRECTIONS.items()}


LOG = []

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


class World(dict):
    """
    Represents the World Map
    """
    world_time = 0
    tick_amount = 1
    minutes_per_tick = 2  # from ultima V
    clipping = True
    wizard_mode = True
    # state of location in world, for entering/exiting towns
    world_y, world_x = 0, 0

    def __init__(
        self, map_id, player_pos, world_0=None, items=None, Portals=None,
        world_data=None, world_y=None, world_x=None,
    ):
        self.map_id = map_id
        self.player_pos = player_pos
        self.world_0 = world_0
        self.Portals = Portals
        self.world_data = world_data
        self.world_y = world_y if world_y is not None else 0
        self.world_x = world_x if world_x is not None else 0

        # bounding dimensions
        self._height = max(y for (y, _x) in items.keys()) if len(items) else 0
        self._width = max(x for (_y, x) in items.keys()) if len(items) else 0

        # cache lookup
        self._player = None
        super().__init__(items or {})

    def __str__(self):
        return f'World(id={self.map_id},pos={self.player_pos},height={self.height},width={self.width})'

    @classmethod
    def load(cls, world_data: dict, world_0=None, map_id: int=0, start_y=None, start_x=None, world_y=None, world_x=None):
        # TODO: split to load_portal() and load_world()
        # TODO: save()! persist between world exits, etc.
        Portals = world_data["World"]["Portals"]
        if world_0 is None:
            # load NEW world map from disk
            assert (None, None) == (start_y, start_x)
            assert map_id == 0, (map_id, start_y, start_x)
            world_0 = u4_data.load_items_from_disk(0, world_data)
            world_y = world_data['World']['start_y']
            world_x = world_data['World']['start_x']
            # add NEW player
            world_0[(world_y, world_x)].append(models.Item.create_player(
                models.Position(y=world_y, x=world_x)))
        if map_id == 0:
            player_pos = models.Position(y=world_y, x=world_x)
        if map_id != 0:
            # load cities etc. always from file
            items = u4_data.load_items_from_disk(map_id, world_data)
            # add NPC's
            for item in u4_data.create_npcs(map_id, world_data):
                items[(item.pos[0], item.pos[1])].append(item)
            # and portals
            Portals = world_data["Maps"][map_id].get("Portals", [])
            # and player
            items[(start_y, start_x)].append(models.Item.create_player(
                models.Position(y=start_y, x=start_x)))
            player_pos = models.Position(y=start_y, x=start_x)

        return cls(
            map_id=map_id,
            player_pos=player_pos,
            Portals=Portals,
            items=items if map_id != 0 else world_0,
            world_data=world_data,
            world_y=world_y,
            world_x=world_x,
            world_0=world_0,
        )

    def debug_details(self, pos):
        local_items = self.find_iter(pos=pos)
        portal = self.find_portal(pos=pos)
        return {
            **{f"map_id": repr(self.map_id)},
            **{f"worldtime": self.world_time},
            f"no-items": sum(len(items) for items in self.values()),
            **{f"itm-{num}": repr(item) for num, item in enumerate(local_items)},
            **({f"portal": repr(portal)} if portal else {}),
            **{f"player_pos": repr(self.player_pos)},
            **{f"world_y": repr(self.world_y)},
            **{f"world_x": repr(self.world_x)},
            **{f"world_height": repr(self.height)},
            **{f"world_width": repr(self.width)},
        }

    @property
    def height(self):
        # how many y-rows?
        return self._height

    @property
    def width(self):
        # how many x-columns?
        return self._width

    def find_iter(self, **kwargs):
        pos = kwargs.pop("pos", None)
        search_items = [self.get((pos.y, pos.x), [])] if pos else self.values()
        return (
            world_item
            for world_items in search_items
            for world_item in world_items
            if all(getattr(world_item, search_key) == search_value for search_key, search_value in kwargs.items())
        )

    def find_iter_not_player(self, pos):
        return (item for item in self.get((pos.y, pos.x), []) if item.name != "player")

    def find_one(self, **kwargs):
        try:
            return next(self.find_iter(**kwargs))
        except StopIteration:
            return None
    
    def find_one_not_player(self, pos):
        try:
            return next(self.find_iter_not_player(pos))
        except StopIteration:
            return None

    @property
    def player(self):
        player_obj = self.find_one(pos=self.player_pos, name="player")
        assert player_obj, (str(self), self.get((self.player_pos.y, self.player_pos.x)))
        return player_obj

    # ALL this stuff might better belong in "UI" class, or, at least,
    # some kind of independent WorldActionService ?
    def do_move_player(self, world, ui, y=0, x=0) -> tuple[bool, bool]:
        # Returns whether the player moved, and screen should be refreshed,
        # and, whether the player exited the map (dirty, do_exit)
        previous_pos = self.player_pos
        next_pos = models.Position(y=previous_pos.y + y, x=previous_pos.x + x)
        can_move = False
        if not self.clipping:
            can_move = True
        # we are not a boat, and it is land,
        elif not self.player.is_boat and world.land_passable(next_pos):
            can_move = True
        # the target is water,
        elif world.water_passable(next_pos):
            # we are a boat,
            if self.player.is_boat:
                can_move = True
            else:
                # we are not a boat, but we can board one
                boat = self.find_one(pos=next_pos, name="boat")
                if not boat:
                    ui.add_text("BLOCKED!")
                else:
                    can_move = True
        else:
            ui.add_text("BLOCKED!")
        if can_move and self.player.is_boat:
            can_move = self.check_boat_direction(y, x)
        if can_move and self.clipping:
            move_result = world.check_tile_movement(next_pos)
            if move_result == 0:
                ui.add_text("SLOW PROGRESS!")
                can_move = False
            elif move_result == -1:
                ui.add_text("BLOCKED!")
                can_move = False
        do_exit = False
        if can_move:
            ui.add_text(make_direction(y=y, x=x))
            # check whether this exits the world map, by stepping into void space
            do_exit = world.map_id != 0 and (
                (next_pos.y < 0 or next_pos.y > world.height) or
                (next_pos.x < 0 or next_pos.x > world.width))
            if not do_exit:
                self.move_item(self.player, next_pos)
                self.player_pos = next_pos
        player_moved = next_pos != previous_pos
        return player_moved, do_exit

    def move_item(self, item, pos):
        if (pos.y, pos.x) in self:
            if item in self[(item.pos.y, item.pos.x)]:
                self[(item.pos.y, item.pos.x)].remove(item)
                self[(pos.y, pos.x)].append(item)
                item.pos = pos

    def do_open_door(self, ui, y=0, x=0):
        # is there an (unlocked) door ?
        pos = models.Position(y=self.player.y + y, x=self.player.x + x)
        door = self.find_one(pos=pos, tile_id=59)
        if not door and self.wizard_mode:
            # wizards can "Open" any tile, LoL!
            items = list(self.find_iter_not_player(pos=pos))
            door = items[-1]
        if door:
            # then set it open!
            door.tile_id = 62
            door.last_action_tick = self.world_time
            door.land_passable = True
            door.darkness = 0
            return True
        ui.add_text("NOT HERE!")

    def do_begin_talk(self, ui, y=0, x=0):
        # is there an npc ?
        npc_pos = models.Position(y=self.player.y + y, x=self.player.x + x)
        ui.talk_npc = self.find_npc(pos=npc_pos)
        if not ui.talk_npc:
            ui.add_text("Funny, no response!")
            return
        ui.add_text("")
        ui.add_text("You meet " + ui.talk_npc.talk_data['prompts']['LOOK'])
        # 50% of the time they introduce themselves.
        if random.randrange(2):
            ui.add_text(f"{ui.talk_npc.talk_data['pronoun']} says: I am {ui.talk_npc.name}")
        ui.add_text(f"Your interest:")
        ui.add_text(f"?")
        return False

    def do_talk_npc(self, ui, viewport, inp):
        if inp.isalnum():
            ui.line_input += inp
            ui.add_text_append(f"{inp}")
            return
        if inp.name in ('KEY_BACKSPACE', 'KEY_DELETE', 'KEY_LEFT'):
            ui.backspace()
            return
        elif inp.name != 'KEY_ENTER':
            # invalid input, TODO: beep
            return

        ui.add_text("")
        line_inp = ui.line_input.strip().upper()[:4]
        ui.line_input = ''
        if ui.talk_npc_asked_question:
            if not (line_inp.startswith('Y') or line_inp.startswith('N')):
                ui.add_text(f"{ui.talk_npc.talk_data['pronoun']} says: yes or no:")
                ui.add_text(ui.talk_npc.talk_data['question'][0])
                ui.add_text("?")
                return
            if line_inp.startswith('Y'):
                ui.add_text(ui.talk_npc.talk_data['question'][1])
            else:
                ui.add_text(ui.talk_npc.talk_data['question'][2])
            ui.talk_npc_asked_question = False
        elif line_inp == 'NAME':
            ui.add_text(f"{ui.talk_npc.talk_data['pronoun']} says: I am {ui.talk_npc.name}")
        elif line_inp == 'PRON':
            # just for debugging
            ui.add_text(f"My preferred pronoun is {ui.talk_npc.talk_data['pronoun']}")
        elif line_inp == 'LOOK':
            ui.add_text("You see " + ui.talk_npc.talk_data['prompts']['LOOK'])
        elif line_inp in ui.talk_npc.talk_data['prompts']:
            ui.add_text(ui.talk_npc.talk_data['prompts'][line_inp])
        elif not line_inp or line_inp in ('BYE', 'THAN'):
            ui.add_text(f"{ui.talk_npc.talk_data["pronoun"]} says: Bye.")
            ui.talk_npc = None
            return
        elif line_inp == 'Z' and self.wizard_mode:
            # wizard mode 'Z' npc talk reveals full npc talk data structure
            ui.add_text(str(ui.talk_npc.talk_data))
        else:
            ui.add_text(f"{ui.talk_npc.talk_data['pronoun']} says: That, "
                              f"I cannot help thee with.")
        # keyword may trigger question
        if line_inp == ui.talk_npc.talk_data['flag_question']:
            ui.add_text("")
            ui.add_text(f"{ui.talk_npc.talk_data['pronoun']} asks:")
            ui.add_text(ui.talk_npc.talk_data['question'][0])
            ui.add_text("?")
            # begin yes/no question state
            ui.talk_npc_asked_question = True
            return
        ui.add_text("")
        ui.add_text("Your interest:")
        ui.add_text("?")


    def board_ship_or_mount_horse(self, ui):
        boat = self.find_one(name="boat", pos=self.player.pos)
        if not boat and self.wizard_mode:
            # wizards can "Board" any tile, LoL!
            items = list(self.find_iter_not_player(pos=self.player.pos))
            if not items:
                return
            boat = items[-1]
        elif not boat:
            ui.add_text("Board WHAT?")
            return
        # player "becomes a boat"
        self.player.tile_id = boat.tile_id
        # delete what was "boarded"
        self[(self.player.pos.y, self.player.pos.x)].remove(boat)
        if not self.get(self.player.pos.y, self.player.pos.x):
            rock = models.Item.create_horse(self.player.pos)
            self[(self.player.pos.y, self.player.pos.x)].append(rock)

    def exit_ship_or_unmount_horse(self, ui):
        if not self.player.is_boat and not self.wizard_mode:
            ui.add_text("Not HERE!")
            return
        boat = models.Item.create_boat(self.player.pos, self.player.tile_id)
        self[(self.player.pos.y, self.player.pos.x)].append(boat)
        self.player.tile_id = models.Item.DEFAULT_PLAYER_TILE_ID
        return

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
    
    def find_npc(self, pos):
        for item in self[(pos.y, pos.x)]:
            if item.talk_data:
                return item

    def check_tile_movement(self, pos) -> int:
        # if any tile at given location has a "speed" variable, then,
        # use as random "SLOW PROGRESS!" deterrent for difficult terrain
        #
        # And, when travelling north, check if player is on Lord British's
        # Castle and Deny movement on any match. Also deny travelling north
        # while on LBC tile.
        if pos.y < self.player.y and self.find_one(pos=self.player_pos, tile_id=LORD_BRITISH_CASTLE_ID):
            return -1
        for item in self.get((pos.y, pos.x), []):
            if item.speed:
                # returns 0 when progress is impeded
                return int(random.randrange(item.speed) != 0)
            if item.tile_id == LORD_BRITISH_CASTLE_ID:
                # Lord British's Castle cannot be entered from the North
                if pos.y > self.player.y:
                    return -1
        return True

    def land_passable(self, pos):
        for item in self.get((pos.y, pos.x), []):
            if not item.land_passable:
                return False
            elif item.material == "liquid":
                return False
        return True

    def water_passable(self, pos):
        for item in self.get((pos.y, pos.x), []):
            if item.tile_id in (0, 1):
                return True
        return False

    def light_blocked(self, pos):
        # whether player movement, or casting of "light" is blocked
        is_void = True
        items = self.get((pos.y, pos.x), [])
        for item in items:
            if item.darkness > 0:
                return True
            is_void = False
        return is_void

    def tick(self):
        # Ultima IV was cruel, it always advanced the time, even without input
        # or making an invalid action, etc
        self.world_time += self.tick_amount
        # XXX performance penalty, needs better tracking "self.Doors?"
        #self.check_close_opened_doors()

    # move to 'world' ?
    def check_close_opened_doors(self):
        # close door after 4 game ticks, a door is always named "Unlocked Door"
        # if it was "O"pened, but is temporarily with a different tile_id
        for door in self.find_iter(name="Unlocked Door"):
            if self.world_time > door.last_action_tick + 4:
                door.tile_id = 59
                door.land_passable = False
                door.darkness = 1
                return True
        return False


class UInterface(object):#
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
    waiting_talk_direction = 0
    show_debug = False

    def __init__(self, tile_svc, darkness=2, radius=9):
        self.term = blessed.Terminal()
        self.dirty_flags =  {
            'tiles': threading.Event(),
            'text': threading.Event(),
            'screen': threading.Event(),
        }
        self.radius = radius
        self.darkness = darkness
        self.tile_svc = tile_svc
        self.line_input = ''
        self.talk_npc = None
        self.talk_npc_asked_question = False
        # something like a double buffer, this tracks the 'tile ansi' result
        # at each (Y, X) location, if this tile has not changed, it is not
        # redrawn, which is less output I/O especially while idle, changing
        # time of render() from ~20ms to as little as 1ms.
        self.tile_output_buffer = {}
        self.text_output_buffer = {}
        self.time_monotonic_last_action = time.monotonic()
        self.text = collections.deque(maxlen=TEXT_HISTORY_LENGTH)

    @property
    def window_size(self):
        return (self.term.height, self.term.width)

    def reader(self, timeout):
        return self.term.inkey(timeout=timeout)

    def reactor(self, inp, ui, world, viewport):
        if ui.talk_npc:
            if inp:
                world.do_talk_npc(ui, viewport, inp)
        elif inp in self.movement_map:
            if self.waiting_open_direction:
                ui.add_text_append(make_direction(**self.movement_map[inp]))
                world.do_open_door(ui, **self.movement_map[inp])
                self.waiting_open_direction = 0
            elif self.waiting_talk_direction:
                ui.add_text_append(make_direction(**self.movement_map[inp]))
                world.do_begin_talk(ui, **self.movement_map[inp])
                self.waiting_talk_direction = 0
            else:
                player_moved, do_exit = world.do_move_player(world, ui, **self.movement_map[inp])
                if player_moved:
                    self.dirty_flags['tiles'].set()
                if do_exit:
                    self.dirty_flags['screen'].set()
                if do_exit:
                    # TODO: track where we are! we can't exit the world map, either!
                    ui.add_text(f"Leaving map_id={world.map_id}")
                    # all maps "exit" to the world map
                    world = World.load(
                        map_id=0,
                        world_0=world.world_0,
                        world_data=world.world_data,
                        # restore player at world position
                        start_y=world.world_y,
                        start_x=world.world_x,
                        world_y=world.world_y,
                        world_x=world.world_x,
                    )
        elif inp and (self.waiting_open_direction or self.waiting_talk_direction):
            # invalid direction after "O"pen or "t"alk
            ui.add_text_append(f"{inp or ''}")
            ui.add_text("NOT HERE!")
            self.waiting_open_direction = 0
            self.waiting_talk_direction = 0
        elif inp == "o":
            ui.add_text("Open-")
            self.waiting_open_direction = time.monotonic()
        elif inp == "t":
            # 'T'alk
            ui.add_text("Talk-")
            self.waiting_talk_direction = time.monotonic()
        elif inp == "E" or inp == "K" or inp == "D":
            # 'E'nter Portal, 'K'limb, and 'D'escend ladder,
            # TODO: also can be used for Air Balloon !
            portal = world.find_portal(world.player.pos)
            item = world.find_one_not_player(pos=world.player.pos)
            can_enter = portal and (inp == "K" and item.is_ladder_up or
                                    inp == "D" and item.is_ladder_down or
                                    # anything else (castles, cities, can be entered)
                                    inp == "E")
            if can_enter:
                map_name = world.world_data["Maps"][portal["dest_id"]]["name"]
                ui.add_text(map_name.upper().center(15))
                world = World.load(
                    world_data=world.world_data,
                    world_0=world.world_0,
                    map_id=portal["dest_id"],
                    # set player position in new map
                    start_x=portal["start_x"],
                    start_y=portal["start_y"],
                    # save or re-use player position in world
                    world_y=world.player_pos.y if world.map_id == 0 else world.world_y,
                    world_x=world.player_pos.x if world.map_id == 0 else world.world_x,
                )
            else:
                action = {"D": "Descend",
                          "K": "Klimb",
                          "E": "Enter"}[inp]
                ui.add_text(f"{action} WHAT?")
        elif inp == "B":
            # 'B'oard ship or mount horse
            world.board_ship_or_mount_horse(ui)
        elif inp == "X":
            # e'X'it ship or unmount horse
            world.exit_ship_or_unmount_horse(ui)
        # todo move into tile_svc
        elif inp == "\x17":  # Control-W
            world.wizard_mode = not world.wizard_mode
            _enabled = {"enabled" if world.wizard_mode else "disabled"}
            ui.add_text(f"Wizard mode {_enabled}")
        elif inp and world.wizard_mode:
            # keys for wizards !
            if inp == "C":
                world.clipping = not world.clipping
                _enabled = {"enabled" if world.clipping else "disabled"}
                ui.add_text(f"Clipping mode {_enabled}")
            elif inp == "A":
                self.auto_resize(ui, viewport)
            elif inp == "R":
                self.radius = 9 if not self.radius else 0
            elif inp in ("[", "]"):
                modifier = 1 if inp == "]" else -1
                self.darkness = max(min(self.darkness + modifier, u4_tiler.MAX_DARKNESS), -60)
                ui.add_text(f"* set darkness level {self.darkness}")
            elif inp == "\x12":  # ^R
                self.tile_svc.cycle_tileset()
                self.dirty_flags['screen'].set()
            elif inp == "\x14":  # ^T
                self.tile_svc.cycle_charset()
                self.dirty_flags['screen'].set()
            elif inp == "\x04":  # ^D
                self.show_debug = not self.show_debug
            elif inp in ("(", ")") and self.radius is not None:
                modifier = 1 if inp == ")" else -1
                self.radius = max(min(self.radius + modifier, MAX_RADIUS), MIN_RADIUS)
                ui.add_text(f"* set radius to {self.radius}")
                self.dirty_flags['tiles'].set()
            elif inp in ("{", "}"):
                modifier = 1 if inp == "}" else -1
                tile_idx = min(max(0, u4_tiler.TILE_SIZES.index(self.tile_svc.tile_size) + modifier), len(u4_tiler.TILE_SIZES) - 1)
                next_size = u4_tiler.TILE_SIZES[tile_idx]
                self.tile_svc.tile_data = self.tile_svc.init_tiles(self.tile_svc.tile_filename, next_size)
                ui.add_text(f"* set tile size to {self.tile_svc.tile_size}")
                self.dirty_flags['screen'].set()
            elif inp in ("<", ">"):
                modifier = 1 if inp == ">" else -1
                char_idx = min(max(0, u4_tiler.CHAR_SIZES.index(self.tile_svc.char_size) + modifier), len(u4_tiler.CHAR_SIZES) - 1)
                next_size = u4_tiler.CHAR_SIZES[char_idx]
                self.tile_svc.char_data = self.tile_svc.init_chars(self.tile_svc.char_filename, next_size)
                ui.add_text(f"* set char size to {self.tile_svc.char_size}")
                self.dirty_flags['screen'].set()
            elif inp == '\x0c':  # ^L
                # used for tracking difficult issues ..
                assert False, LOG
        else:
            # even when we don't move, the world may forcefully tick!
            if ui.time_monotonic_last_action + TIME_PLAYER_PASS > time.monotonic():
                dirty = 0
            else:
                ui.add_text('Pass')

        # Ultima IV is cruel, if *anything* happens it drives the game forward!!
        if any(flag.is_set() for flag in self.dirty_flags.values()):
            world.tick()
            self.time_monotonic_last_action = time.monotonic()

        # the cursor is always animated if the main game loop is running
        ui.animate_cursor()
        return world

    def auto_resize(self, ui, viewport):
        # TODO: belongs in TileService ??
        # self.tile_svc.tile_data = self.tile_svc.init_tiles(self.tile_svc.tile_filename, self.tile_svc.tile_size)
        if self.radius:
            while self.tile_svc.tile_size > u4_tiler.MIN_TILE_SIZE and (
                (self.radius * 2) + 1 > min(
                    math.ceil(viewport.width // self.tile_svc.tile_width),
                    math.ceil(viewport.height // self.tile_svc.tile_height))
            ):
                tile_idx = u4_tiler.TILE_SIZES.index(self.tile_svc.tile_size)
                next_tile_size = u4_tiler.TILE_SIZES[tile_idx - 1]
                self.tile_svc.tile_data = self.tile_svc.init_tiles(self.tile_svc.tile_filename, next_tile_size)
                self.dirty_flags['screen'].set()
            while self.tile_svc.tile_size < u4_tiler.MAX_TILE_SIZE and (
                (self.radius * 2) + 1 < min(
                    math.ceil(viewport.width // self.tile_svc.tile_width),
                    math.ceil(viewport.height // self.tile_svc.tile_height))
            ):
                tile_idx = u4_tiler.TILE_SIZES.index(self.tile_svc.tile_size)
                next_tile_size = u4_tiler.TILE_SIZES[tile_idx + 1]
                self.tile_svc.tile_data = self.tile_svc.init_tiles(self.tile_svc.tile_filename, next_tile_size)
                self.dirty_flags['screen'].set()
            if self.dirty_flags['screen'].is_set():
                ui.add_text(
                    f"autoresize tile={self.tile_svc.tile_size}, "
                    f"tiles_width={viewport.width // self.tile_svc.tile_width}"
                    f"tiles_height={viewport.height // self.tile_svc.tile_height}"
                    f"radius * 2={self.radius * 2}, "
                )

    @contextlib.contextmanager
    def activate(self):
        with self.term.fullscreen(), self.term.keypad(), self.term.cbreak(), self.term.hidden_cursor():
            echo(self.term.clear)
            yield self

    def debug_details(self):
        return {
            "tile-width": self.tile_svc.tile_width,
            "tile-height": self.tile_svc.tile_height,
            "tile-cache": self.tile_svc._make_ansi_tile.cache_info(),
            "tileset": self.tile_svc.tile_filename,
            "radius": self.radius,
            "darkness": self.darkness,
        }


    def _render_debug_details(self, viewport, debug_details):
        ypos, xpos = viewport.yoffset, (viewport.xoffset * 2) + viewport.width
        left = viewport.width + (viewport.xoffset * 2)
        width = max(0, self.term.width - left - (viewport.xoffset))
        output = ''
        for debug_key, debug_val  in debug_details.items():
            for line in textwrap.wrap(
                f"{debug_key}: {debug_val}",
                width=viewport.width, subsequent_indent=" ", drop_whitespace=False,
                replace_whitespace=False,
            ):
                output += self.term.move_yx(ypos, xpos) + line.ljust(width)
                ypos += 1
        echo(output)
        

    def render_text(self, viewport):
        if self.dirty_flags['text'].is_set():
            ypos = viewport.yoffset
            left = viewport.width + (viewport.xoffset * 2)
            text_width = max(0, self.term.width - left - (viewport.xoffset)) // self.tile_svc.char_width
            if not text_width:
                return
            text_height = max(0, viewport.height - ypos) // self.tile_svc.char_height
            all_text = []
            for text_message in list(self.text)[-text_height:]:
                all_text.extend(
                    textwrap.wrap(text_message, width=text_width, subsequent_indent='',
                                  drop_whitespace=False, replace_whitespace=False) or ['']
                )
            echo(self.make_displayed_text(text_lines=all_text[-text_height:], viewport=viewport,
                                          text_height=text_height, text_width=text_width))
            self.dirty_flags['text'].clear()

    def make_displayed_text(self, viewport, text_lines, text_height, text_width):
        # JOE's own terminal emulator
        ypos = viewport.yoffset
        left = viewport.width + (viewport.xoffset * 2)
        output = ''
        while len(text_lines) < text_height:
            text_lines.append(''.ljust(text_width))
        actual_y = 0
        for y, text_line in enumerate(text_lines):
            actual_y = ypos + (y * self.tile_svc.char_height)
            actual_x = 0
            for x, text_char in enumerate(text_line.ljust(text_width,' ')):
                actual_x = left + (x * self.tile_svc.char_width)
                char_tile = self.tile_svc.make_character_tile(character=text_char)
                if self.dirty_flags['screen'].is_set() or char_tile != self.text_output_buffer.get((y, x)):
                    self.text_output_buffer[(y, x)] = char_tile
                    for y_offset, char_tile_row in enumerate(char_tile):
                        output += self.term.move(actual_y + y_offset, actual_x) + char_tile_row
        #     if self.dirty_flags['screen'].is_set():
        #         # clear to right margin of viewport
        #         remaining_xloc = actual_x + self.tile_svc.char_width
        #         remaining_xspace = max(0, self.term.width - 2 - remaining_xloc)
        #         if remaining_xspace:
        #             for y_offset in range(self.tile_svc.char_height):
        #                 output += self.term.move(actual_y + y_offset, remaining_xloc) + ('*' * remaining_xspace)
        # if self.dirty_flags['screen'].is_set():
        #     actual_y += (self.tile_svc.char_height - 1)
        #     while actual_y < viewport.height:
        #         remaining_xspace = max(0, self.term.width - 2 - left)
        #         actual_y += 1
        #         # add blank links to clear to bottom of viewport
        #         output += self.term.move(actual_y, left) + ('*' * remaining_xspace)
        return output

    def draw_decoration(self, viewport):
        # todo: make exactly like IV, with moon phases, etc!
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
        echo(border_color(" " * self.term.width) * viewport.yoffset)

    def render(self, world, viewport):
        viewport.re_adjust(ui=self, player_pos=world.player_pos)
        if self.dirty_flags['screen'].is_set():
            # clear the screen, clear refresh buffers
            echo(self.term.clear)
            self.tile_output_buffer.clear()
            self.text_output_buffer.clear()
            self.draw_decoration(viewport)
            self.dirty_flags['tiles'].set()
            self.dirty_flags['text'].set()
        if self.dirty_flags['tiles'].is_set():
            # Does this belong in TileService ?
            output_text = ''
            items_by_row = viewport.items_in_view_by_row(world, ui=self)
            for cell_row, cell_items in enumerate(items_by_row):
                ypos = cell_row * self.tile_svc.tile_height
                for cell_number, items in enumerate(cell_items):
                    xpos = cell_number * self.tile_svc.tile_width
                    tile_ans = self.tile_svc.make_ansi_tile(items, world.player_pos, self.darkness)
                    if self.dirty_flags['screen'].is_set() or tile_ans != self.tile_output_buffer.get((ypos, xpos)):
                        self.tile_output_buffer[(ypos, xpos)] = tile_ans
                        actual_xpos = xpos + viewport.xoffset
                        for ans_y, ans_txt in enumerate(tile_ans):
                            actual_ypos = ypos + ans_y + viewport.yoffset
                            if actual_ypos <= viewport.height:
                                output_text += (self.term.move_yx(actual_ypos, actual_xpos) + ans_txt)
            echo(output_text, flush=True)
            self.dirty = 0
        self.dirty_flags['screen'].clear()

    def add_text(self, text):
        self.maybe_delete_cursor()
        self.text.append(text)
        self.dirty_flags['text'].set()

    def add_text_append(self, text):
        # add text to the last line for continuation of paragraph,
        # actions to result, etc.
        self.maybe_delete_cursor()
        try:
            val = self.text.pop()
        except IndexError:
            val = ''
        self.text.append(val + text)
        self.dirty_flags['text'].set()

    def animate_cursor(self):
        cursor_chr = chr((28, 29, 30, 31)[int(time.monotonic() * 8) % 4])
        self.add_text_append(f'{cursor_chr}')

    def maybe_delete_cursor(self):
        try:
            last_char = self.text[-1][-1]
        except IndexError:
            return
        # delete cursor from prev. line before appending next line
        if last_char in (chr(28), chr(29), chr(30), chr(31)):
            val = self.text.pop()
            val = val[:-1]
            if val:
                self.text.append(val)
    
    def backspace(self):
        self.maybe_delete_cursor()
        self.text.append(self.text.pop()[:-1])
 

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

    def __init__(self, y, x, height, width, yoffset, xoffset):
        (self.y, self.x) = (y, x)
        self.height, self.width = height, width
        self.yoffset, self.xoffset = yoffset, xoffset

    def __repr__(self):
        return f"{self.y}, {self.x}"

    @classmethod
    def create(cls, ui, player_pos, yoffset=1, xoffset=2):
        "Create viewport instance centered one z-level above player."
        vp = cls(0, 0, 1, 1, yoffset, xoffset)
        vp.re_adjust(ui=ui, player_pos=player_pos)
        return vp

    def re_adjust(self, ui, player_pos):
        "re-center viewport on player and set 'dirty' flag on terminal resize"
        viewport_char_width = 15
        viewport_pct_width = 0.80
        height = ui.term.height - (self.yoffset * 2)
        width = min(ui.term.width - (viewport_char_width * ui.tile_svc.char_width), int(ui.term.width * viewport_pct_width))
        # terminal resize detected
        if (height, width) != (self.height, self.width):
            ui.dirty_flags['screen'].set()
        self.height, self.width = height, width
        self.y, self.x = 0, 0

        if player_pos is not None:
            self.y = player_pos.y - int(math.ceil(self.get_tiled_height(ui.tile_svc) / 2)) + 1
            self.x = player_pos.x - int(math.floor(self.get_tiled_width(ui.tile_svc) / 2))


    def get_tiled_height(self, tile_svc):
        return int(math.ceil(self.height / tile_svc.tile_height))

    def get_tiled_width(self, tile_svc):
        return int(math.floor(self.width / tile_svc.tile_width))

    def items_in_view_by_row(self, world, ui):
        # cast 'field of view' from small_world
        if ui.radius:
            visible = self.do_fov(player_pos=world.player_pos, radius=ui.radius,
                                  fn_light_blocked=world.light_blocked)
        else:
            visible = set((Y, X) for Y in range(world.height) for X in range(world.width))

        def make_void(y, x):
            return models.Item.create_void(pos=models.Position(y, x))

        for y in range(self.y, self.y + self.get_tiled_height(ui.tile_svc)):
            yield [
                (
                    world.get((y, x)) if (ui.radius == 0 or (y, x) in visible)
                    else [make_void(y, x)]
                )
                or [make_void(y, x)]
                for x in range(self.x, self.x + self.get_tiled_width(ui.tile_svc))
            ]

    def do_fov(self, player_pos, radius, fn_light_blocked):
        # start with the 8 octants, and cast light in each direction,
        # recursively sub-dividing remaining quadrants, cancelling
        # quadrants behind shadows, and marking 'visible'
        visible = {player_pos}
        for oct in range(8):
            visible.update(
                self.cast_light(
                    cx=player_pos.x,
                    cy=player_pos.y,
                    row=1,
                    start=1.0,
                    end=0.0,
                    radius=radius,
                    xx=self.MULT.xx[oct],
                    xy=self.MULT.xy[oct],
                    yx=self.MULT.yx[oct],
                    yy=self.MULT.yy[oct],
                    depth=0,
                    fn_light_blocked=fn_light_blocked,
                )
            )
        return visible

    def cast_light(self, cx, cy, row, start, end, radius, xx, xy, yx, yy, depth, fn_light_blocked):
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
                    if fn_light_blocked(models.Position(Y, X)):
                        new_start = r_slope
                    else:
                        blocked = False
                        start = new_start
                    continue
                if fn_light_blocked(models.Position(Y, X)) and j < radius:
                    # This is a blocking square, start a child scan:
                    blocked = True
                    visible.update(
                        self.cast_light(
                            cx=cx,
                            cy=cy,
                            row=j + 1,
                            start=start,
                            end=l_slope,
                            radius=radius,
                            xx=xx,
                            xy=xy,
                            yx=yx,
                            yy=yy,
                            depth=depth + 1,
                            fn_light_blocked=fn_light_blocked
                        )
                    )
                    new_start = r_slope
            # Row is scanned; do next row unless last square was blocked:
            if blocked:
                break
        return visible


def _loop(ui, world, viewport):
    inp = None
    time_render = 0
    time_action = 0
    time_input = 0
    time_text = 0
    time_render = 0
    # cause very first key input to have a timeout of nearly 0
    while True:
        with elapsed_timer() as time_text:
            ui.render_text(viewport)
        time_text = time_text()

        with elapsed_timer() as time_render:
            ui.render(world, viewport)
            if ui.show_debug:
                ui._render_debug_details(viewport, debug_details={
                    "ms-world-render": int(time_render() * 1e3),
                    "ms-reactor": int(time_action * 1e3),
                    "ms-input": int(time_input * 1e3),
                    "ms-text": int(time_text * 1e3),
                    # of "whole world"
                    **world.debug_details(pos=world.player_pos),
                    # details of "small world"
                    **ui.debug_details(),
                })
            echo('', flush=True)

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
        time_input = time_input()

        with elapsed_timer() as time_action:
            world = ui.reactor(inp, ui, world, viewport)
        time_action = time_action()


def init_begin_world(world_data):
    world = World.load(world_data)

    # Add test boat!
    world[(110, 86)].append(models.Item.create_boat(models.Position(y=110, x=86)))
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
    tile_svc = u4_tiler.TileService(
        Tilesets=world_data["Tilesets"],
        tile_filename=world_data['DEFAULT_TILESET'],
        tile_size=world_data['DEFAULT_TILE_SIZE'],
        Charsets=world_data["Charsets"],
        char_filename=world_data['DEFAULT_CHARSET'],
        char_size=world_data['DEFAULT_CHAR_SIZE'],
        )
    ui = UInterface(tile_svc=tile_svc,
                    darkness=world_data.get('DEFAULT_DARKNESS', 1),
                    radius=world_data.get('DEFAULT_RADIUS', 9))
    world = init_begin_world(world_data)
    for _ in range(4):
        for n in range(128):
            ui.add_text_append(chr(n))
    viewport = Viewport.create(ui=ui, player_pos=world.player_pos)
    with ui.activate():
        _loop(ui, world, viewport)


if __name__ == "__main__":
    exit(main())


# todo today,
# - character set, blown up text
# - (C)ast (P)eer, we should be able to calculate optimal
#   tile size that can view the 32x32 map in viewport, and
#   scale for the given tile_size.
#    - it should be a second "viewport", temporarily rendered
#      in place of the original .. we also need to blink/inverse
#      the tile of the current location.
#    - and, we should compound all tiles together into one large PNG,
#      this should allow for much greater detail, we could render the
#      animations of water and fields, include NPCs, etc., !
# - more effects, like for hallucinations !
# - finish NPC talk, why are there so many "Landri" characters in LBC?
#   because, they are special characters .. ?
# - we can still crash when exitting the world map (but requires wizard mode ..?)
#
# Persistent worlds,
# - Wizard mode has (C)opy and some kind of tile editor
# - save worlds to disk
# - pre-create u4 world into yaml-only!
#
# Investigate,
# - import Ultima V map tiles and tileset!
# - why can't we load *all* tilesets, and select them by definition?
#
# later,
# - re-implement ultima V's lighthouse, what an effect !!
#   and firepit should also be like a candlelight ..
# - world map ! (P)eer at Gem!
#   copy is just board but leave the first item
# - tilesets!
# - bugfix darkness, seems levels 3/4 are reversed?
# - implement detailed viewport borders
# - implement save and restore, save as modifable yaml!
#   - this should allow us to delete *.ULT and WORLD.MAP
# - implement wind
# - implement moon cycles
# - is "shadow" casting possible?
# - make "test map" that can be entered
#
#
# kids want
# - isaac to be a pharoh
# - luke wants wolves to attack
# - luke wants to be able to farm
# - the ability to move or "repair" a computer
#
#
# npcs,
# - add custom 'goodbyes'
# - add more lines (chat-gpt?)
# graphics improvements,
# - WHITE STONE + WATER tile could animation just the water portion