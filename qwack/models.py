import collections
import functools
import random
import time
import math

Position = collections.namedtuple("Position", ("y", "x"))

from enum import Enum

SpellState = Enum('SpellState', [
    'NONE',
    'WAIT_FOR_SPELL',
    'WAIT_FOR_TARGET',
    'BEGIN_SPELL',
    'CAST_SPELL',
])

KarmaAction = Enum('KarmaAction', [
    'FOUND_ITEM',
    'STOLE_CHEST',
    'GAVE_TO_BEGGAR',
    'GAVE_ALL_TO_BEGGAR',
    'BRAGGED',
    'HUMBLE',
    'HAWKWIND',
    'MEDITATION',
    'BAD_MANTRA',
    'ATTACKED_GOOD',
    'FLED_EVIL',
    'FLED_GOOD',
    'HEALTHY_FLED_EVIL',
    'KILLED_EVIL',
    'SPARED_GOOD',
    'DONATED_BLOOD',
    'DIDNT_DONATE_BLOOD',
    'CHEAT_REAGENTS',
    'DIDNT_CHEAT_REAGENTS',
    'USED_SKULL',
    'DESTROYED_SKULL'])


@functools.lru_cache(maxsize=256 * 256)
def square_distance(x_delta, y_delta):
    return math.sqrt((x_delta) ** 2 + (y_delta) ** 2)


class PartyMember:
    def __init__(self, name, gender, player_class):
        self.name = name
        self.gender = gender
        self.player_class = player_class

        self.max_hp = 0
        self.max_magic_points = 0
        self.experience = 0
        self.strength = 0
        self.dexterity = 0
        self.intelligence = 0
        self.magic_points = 0
        self.weapon = None
        self.armor = None

class Party:
    def __init__(self, members: PartyMember | None = None):
        self.members = members or []
        self.food = 0
        self.gold = 0
        self.karma = 0
        self.reagents = {
            "sulfurous ash": 0,
            "ginseng": 0,
            "garlic": 0,
            "spider silk": 0,
            "blood moss": 0,
            "black pearl": 0,
            "nightshade": 0,
            "mandrake root": 0,
        }

class Item:
    DEFAULT_PLAYER_TILE_ID = 31

    def __init__(
        self,
        tile_id,
        pos,
        name,
        material="construction",
        sort_value=0,
        darkness=0,
        brightness=0,
        land_passable=True,
        speed=0,
        move_behavior=None,
        talk_data=None,
        composite=True,
    ):
        self.tile_id = tile_id
        self.name = name
        self.material = material
        self.sort_value = sort_value
        self._pos = pos
        self.darkness = darkness
        self.brightness = brightness
        self.land_passable = land_passable
        self.speed = speed
        self.move_behavior = move_behavior
        self.talk_data = talk_data
        self.composite = composite
        self.last_action_tick = 0

    def __repr__(self):
        return (f"{self.name}<id:{self.tile_id},"
                f"y={self.y},x={self.x},d={self.darkness}b={self.brightness},s={self.sort_value},c={int(self.composite)}>")

    def __str__(self):
        return self.name

    @classmethod
    def create_player(cls, pos):
        return cls(
            tile_id=cls.DEFAULT_PLAYER_TILE_ID,
            pos=pos,
            name="player",
            material="flesh",
            sort_value=2,
        )

    @classmethod
    def create_boat(cls, pos, tile_id=None):
        return cls(tile_id=16 if tile_id is None else tile_id, pos=pos, name="boat", sort_value=1)

    @classmethod
    def create_horse(cls, pos, tile_id=None):
        return cls(tile_id=20 if tile_id is None else tile_id, pos=pos, name="horse", sort_value=1)

    @classmethod
    def create_balloon(cls, pos, tile_id=None):
        return cls(tile_id=24 if tile_id is None else tile_id, pos=pos, name="balloon", sort_value=1)

    @classmethod
    def create_void(cls, pos):
        "Create an item that represents void, black space."
        return cls(
            tile_id=-1,
            pos=pos,
            name="void",
            material="liquid",
            darkness=0,
            land_passable=False,
            speed=8)

    @classmethod
    def create_rock(cls, pos):
        "Create a 'rock', used to fill empty space in wizard_mode.,"
        return cls(
            tile_id=55,
            pos=pos,
            name="Wizard's Rock",
            material="stone",
            land_passable=False)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def y(self):
        return self._pos[0]

    @property
    def x(self):
        return self._pos[1]

    def get_y_offset(self, tick):
        if self.tile_id in (0, 1, 2, 68, 69, 70, 71):
            return tick % 16
        return 0


    def animation_tile_id(self, tick):
        animatable_tile_ids_12 = (32, 34, 36, 38, 40, 42, 44, 46, 80, 82, 84, 86, 88, 90, 92, 94, 132, 134, 136, 138, 140, 142)
        animatable_tile_ids_1234 = (144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                                    220, 224, 228, 232, 236, 240, 244, 248, 252)
        frames = []
        for atid in animatable_tile_ids_12:
            if self.tile_id in (atid, atid + 1):
                frames = [atid, atid + 1]
                break
        if not frames:
            for atid in animatable_tile_ids_1234:
                if self.tile_id in (atid, atid+1, atid+2, atid+3):
                    frames = [atid, atid+1, atid+2, atid+3]
                    break
        if not frames:
            return self.tile_id
        random.seed(tick + self.tile_id)
        return random.choice(frames)

    def is_adjacent(self, other_item):
        # Check if the target coordinates are adjacent to the given coordinates
        return (
            abs(self.x - other_item.x) == 1
            and self.y == other_item.y
            or abs(self.y - other_item.y) == 1
            and self.x == other_item.x
        )

    def distance(self, other_pos):
        return square_distance(self.x - other_pos.x, self.y - other_pos.y)

    @property
    def is_field(self):
        return self.tile_id in (68, 69, 70, 71)

    @property
    def is_water(self):
        return self.tile_id in (0, 1, 2)

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
    def is_enterable(self):
        return self.tile_id in (10, 11, 12, 14)

    @property
    def is_ladder_up(self):
        return self.tile_id == 27

    @property
    def is_ladder_down(self):
        return self.tile_id == 28

    @property
    def is_void(self):
        return self.tile_id == -1
