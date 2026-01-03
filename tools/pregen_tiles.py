#!/usr/bin/env python3
"""
Pre-generate all tiles for the tile cache.

This script generates all possible tile combinations to avoid runtime chafa calls.
It handles:
- All 256 tile IDs
- All tile sizes (1, 2, 3, 6, 8, 12, 16)
- All darkness levels (0-5)
- All inverse states (True, False)
- Composite tiles (foreground over background)
- Water animation offsets (y_offset_bg 0-15)
- Field animation offsets (y_offset_fg 0-15)
- Confusion spell offsets (x/y from -4 to +4)
- All charsets and character tiles
"""

import argparse
import io
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator, Optional

import yaml

# Add parent directory to path to import qwack modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwack.u4_tiler import (
    CHAFA_BIN,
    CHAFA_EXTRA_ARGS,
    CHAFA_TRIM_START,
    TILE_SIZES as DEFAULT_TILE_SIZES,
    CHAR_SIZES as DEFAULT_CHAR_SIZES,
    MIN_DARKNESS,
    MAX_DARKNESS,
    load_tileset,
    load_charset,
    make_image_from_pixels,
    apply_darkness,
    apply_offsets,
    apply_composite,
    apply_inverse,
)

# Constants (can be overridden by --default-sizes)
TILE_SIZES = DEFAULT_TILE_SIZES
CHAR_SIZES = DEFAULT_CHAR_SIZES
DARKNESS_LEVELS = tuple(range(MIN_DARKNESS, MAX_DARKNESS + 1))  # 0-5
INVERSE_STATES = (False, True)
CONFUSION_OFFSETS = tuple(range(-4, 5))  # -4 to +4


class TileProperties:
    """Tile properties extracted from world.yaml Shapes section."""

    def __init__(self, world_data: dict):
        shapes = world_data.get("Shapes", {})
        creatures = world_data.get("Creatures", {})

        # Extract tile properties from Shapes
        self.water_tiles = set()        # material: liquid
        self.field_tiles = set()        # sort_value: 3 (fields render above terrain)
        self.non_composite_tiles = set()  # composite: no
        self.walkable_bg_tiles = set()  # land_passable: yes or default (terrain)
        self.bright_tiles = set()       # brightness > 0
        self.composite_fg_tiles = set()  # tiles that can walk on terrain

        for tile_id, props in shapes.items():
            if not isinstance(props, dict):
                continue

            # Water tiles (material: liquid)
            if props.get("material") == "liquid":
                self.water_tiles.add(tile_id)

            # Field tiles (sort_value: 3)
            if props.get("sort_value") == 3:
                self.field_tiles.add(tile_id)

            # Non-composite tiles (composite: no)
            if props.get("composite") == "no" or props.get("composite") is False:
                self.non_composite_tiles.add(tile_id)

            # Bright tiles (brightness > 0)
            brightness = props.get("brightness", 0)
            if brightness and brightness > 0:
                self.bright_tiles.add(tile_id)

            # Walkable background tiles - terrain and floors that things can walk on
            # Default is land_passable: yes unless explicitly set to no
            land_passable = props.get("land_passable", True)
            if land_passable is True or land_passable == "yes":
                # Also include water as walkable backgrounds for boats/sea creatures
                self.walkable_bg_tiles.add(tile_id)
            # Also add water tiles as walkable backgrounds
            if props.get("material") == "liquid":
                self.walkable_bg_tiles.add(tile_id)

        # Extract creature tiles from Creatures section
        creature_tiles = set()
        for creature_id, props in creatures.items():
            if not isinstance(props, dict):
                continue
            tile_id = props.get("tile")
            # Only include numeric tile IDs (skip string references like "magic_flash")
            if tile_id is not None and isinstance(tile_id, int):
                creature_tiles.add(tile_id)
                # Creatures often have 2-4 animation frames
                for offset in range(1, 4):
                    creature_tiles.add(tile_id + offset)

        # Composite foreground tiles: creatures, player, party members, fields
        # These are tiles that can appear over walkable terrain
        # Include: player (31), party members (32-47), creatures, fields
        self.composite_fg_tiles = set()

        # Vehicles (ships, horses, balloon) - these appear over terrain
        for tile_id in range(16, 25):  # 16-19 ships, 20-21 horses, 22 tile floor, 23 bridge, 24 balloon
            self.composite_fg_tiles.add(tile_id)

        # Player and party members (31-47)
        for tile_id in range(31, 48):
            if tile_id not in self.non_composite_tiles:
                self.composite_fg_tiles.add(tile_id)

        # Add all creature tiles
        self.composite_fg_tiles.update(creature_tiles)

        # Add field tiles (they composite over terrain)
        self.composite_fg_tiles.update(self.field_tiles)

        # Add NPC tiles (80-95) - guards, citizens, etc.
        for tile_id in range(80, 96):
            if tile_id not in self.non_composite_tiles:
                self.composite_fg_tiles.add(tile_id)

        # Remove any non-composite tiles from composite_fg
        self.composite_fg_tiles -= self.non_composite_tiles

        # Ensure walkable_bg_tiles has common terrain types even if not in yaml
        # These are known walkable tiles from the game
        known_walkable = {
            0, 1, 2,   # Water (for boats/sea creatures)
            3,         # Swamp
            4,         # Grassland
            5,         # Scrubland
            6,         # Forest
            7,         # Hills
            22,        # Tile floor
            23,        # Bridge
            25, 26,    # Bridge north/south
            62,        # Brick floor
            63,        # Wooden planks
            74,        # Altar
            76,        # Lava flow
        }
        self.walkable_bg_tiles.update(known_walkable)

        # For wizard mode (clipping disabled), player can walk on ANY tile
        # Add all tiles as potential backgrounds
        for tile_id in range(256):
            self.walkable_bg_tiles.add(tile_id)

    def __repr__(self):
        return (
            f"TileProperties(\n"
            f"  water_tiles={sorted(self.water_tiles)},\n"
            f"  field_tiles={sorted(self.field_tiles)},\n"
            f"  non_composite_tiles={sorted(self.non_composite_tiles)},\n"
            f"  walkable_bg_tiles={sorted(self.walkable_bg_tiles)},\n"
            f"  bright_tiles={sorted(self.bright_tiles)},\n"
            f"  composite_fg_tiles={sorted(self.composite_fg_tiles)}\n"
            f")"
        )


# Global tile properties (loaded from world.yaml)
TILE_PROPS: Optional[TileProperties] = None


def get_tile_props() -> TileProperties:
    """Get or load tile properties from world.yaml."""
    global TILE_PROPS
    if TILE_PROPS is None:
        world_data = load_world_data()
        TILE_PROPS = TileProperties(world_data)
    return TILE_PROPS


@dataclass
class TileSpec:
    """Specification for a tile to generate."""
    tileset_filename: str
    tile_id: int
    bg_tile_id: Optional[int]
    tile_width: int
    tile_height: int
    tile_darkness: int
    x_offset_bg: int
    y_offset_bg: int
    x_offset_fg: int
    y_offset_fg: int
    inverse: bool

    @property
    def cache_key(self) -> str:
        """Generate the cache key for this tile."""
        return os.path.join(
            self.tileset_filename,
            str(self.tile_id),
            f"{self.bg_tile_id}_{self.tile_width}_{self.tile_height}_{self.tile_darkness}_"
            f"{self.x_offset_bg}_{self.y_offset_bg}_{self.x_offset_fg}_{self.y_offset_fg}_{self.inverse}.txt"
        )


@dataclass
class CharSpec:
    """Specification for a character tile to generate."""
    charset_filename: str
    char_id: int
    char_width: int
    char_height: int

    @property
    def cache_key(self) -> str:
        """Generate the cache key for this character."""
        return os.path.join(
            self.charset_filename,
            str(self.char_id),
            f"None_{self.char_width}_{self.char_height}_0_0_0_0_0_False.txt"
        )


def get_tile_height_for_width(tile_width: int) -> int:
    """Calculate tile height based on width (from chafa aspect ratio)."""
    # Based on how TileService.init_tiles calculates tile_height
    # The height is determined by rendering a test tile and measuring lines
    # For --font-ratio=9/16, height is approximately width * 9/16 rounded
    # But looking at the actual cache, heights are: 1->1, 2->2, 3->2, 6->4, 8->5, 12->7, 16->9
    height_map = {1: 1, 2: 2, 3: 2, 6: 4, 8: 5, 12: 7, 16: 9}
    return height_map.get(tile_width, max(1, int(tile_width * 9 / 16)))


def get_char_height_for_width(char_width: int) -> int:
    """Calculate char height based on width."""
    # Character tiles use different dimensions
    # Based on actual cache: 6->4, 7->4, 8->5
    height_map = {6: 4, 7: 4, 8: 5}
    return height_map.get(char_width, max(1, int(char_width * 9 / 16)))


def enumerate_single_tiles(tileset_filename: str) -> Iterator[TileSpec]:
    """Enumerate all single tiles (no background) for a tileset."""
    props = get_tile_props()

    for tile_id in range(256):
        for tile_width in TILE_SIZES:
            tile_height = get_tile_height_for_width(tile_width)

            # For bright tiles, only generate darkness=0
            darkness_levels = (0,) if tile_id in props.bright_tiles else DARKNESS_LEVELS

            for darkness in darkness_levels:
                for inverse in INVERSE_STATES:
                    yield TileSpec(
                        tileset_filename=tileset_filename,
                        tile_id=tile_id,
                        bg_tile_id=None,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        tile_darkness=darkness,
                        x_offset_bg=0,
                        y_offset_bg=0,
                        x_offset_fg=0,
                        y_offset_fg=0,
                        inverse=inverse,
                    )


def enumerate_composite_tiles(tileset_filename: str) -> Iterator[TileSpec]:
    """Enumerate all composite tiles (foreground over background) for a tileset."""
    props = get_tile_props()

    for fg_tile in props.composite_fg_tiles:
        for bg_tile in props.walkable_bg_tiles:
            for tile_width in TILE_SIZES:
                tile_height = get_tile_height_for_width(tile_width)

                # For bright backgrounds, only darkness=0
                darkness_levels = (0,) if bg_tile in props.bright_tiles else DARKNESS_LEVELS

                for darkness in darkness_levels:
                    for inverse in INVERSE_STATES:
                        # Base case (no offsets)
                        yield TileSpec(
                            tileset_filename=tileset_filename,
                            tile_id=fg_tile,
                            bg_tile_id=bg_tile,
                            tile_width=tile_width,
                            tile_height=tile_height,
                            tile_darkness=darkness,
                            x_offset_bg=0,
                            y_offset_bg=0,
                            x_offset_fg=0,
                            y_offset_fg=0,
                            inverse=inverse,
                        )

                        # Water animation offsets (y_offset_bg for water backgrounds)
                        if bg_tile in props.water_tiles:
                            for y_off in range(1, 16):  # 1-15, 0 already done
                                yield TileSpec(
                                    tileset_filename=tileset_filename,
                                    tile_id=fg_tile,
                                    bg_tile_id=bg_tile,
                                    tile_width=tile_width,
                                    tile_height=tile_height,
                                    tile_darkness=darkness,
                                    x_offset_bg=0,
                                    y_offset_bg=y_off,
                                    x_offset_fg=0,
                                    y_offset_fg=0,
                                    inverse=inverse,
                                )

                        # Field animation offsets (y_offset_fg for field foregrounds)
                        if fg_tile in props.field_tiles:
                            for y_off in range(1, 16):  # 1-15, 0 already done
                                yield TileSpec(
                                    tileset_filename=tileset_filename,
                                    tile_id=fg_tile,
                                    bg_tile_id=bg_tile,
                                    tile_width=tile_width,
                                    tile_height=tile_height,
                                    tile_darkness=darkness,
                                    x_offset_bg=0,
                                    y_offset_bg=0,
                                    x_offset_fg=0,
                                    y_offset_fg=y_off,
                                    inverse=inverse,
                                )


def enumerate_confusion_tiles(tileset_filename: str) -> Iterator[TileSpec]:
    """Enumerate all confusion spell offset variants for a tileset."""
    props = get_tile_props()

    for tile_id in range(256):
        for tile_width in TILE_SIZES:
            tile_height = get_tile_height_for_width(tile_width)

            # For bright tiles, only generate darkness=0
            darkness_levels = (0,) if tile_id in props.bright_tiles else DARKNESS_LEVELS

            for darkness in darkness_levels:
                for x_off in CONFUSION_OFFSETS:
                    for y_off in CONFUSION_OFFSETS:
                        # Skip 0,0 (already generated in single tiles)
                        if x_off == 0 and y_off == 0:
                            continue

                        # Generate both inverse states for confusion
                        for inverse in INVERSE_STATES:
                            yield TileSpec(
                                tileset_filename=tileset_filename,
                                tile_id=tile_id,
                                bg_tile_id=None,
                                tile_width=tile_width,
                                tile_height=tile_height,
                                tile_darkness=darkness,
                                x_offset_bg=0,
                                y_offset_bg=0,
                                x_offset_fg=x_off,
                                y_offset_fg=y_off,
                                inverse=inverse,
                            )


def enumerate_water_animation_tiles(tileset_filename: str) -> Iterator[TileSpec]:
    """Enumerate water tiles with animation offsets (single tiles, no composite)."""
    props = get_tile_props()

    for tile_id in props.water_tiles:
        for tile_width in TILE_SIZES:
            tile_height = get_tile_height_for_width(tile_width)

            for inverse in INVERSE_STATES:
                # Water animation uses y_offset_fg when not composited
                for y_off in range(1, 16):  # 1-15, 0 already in single tiles
                    yield TileSpec(
                        tileset_filename=tileset_filename,
                        tile_id=tile_id,
                        bg_tile_id=None,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        tile_darkness=0,  # Water is bright
                        x_offset_bg=0,
                        y_offset_bg=0,
                        x_offset_fg=0,
                        y_offset_fg=y_off,
                        inverse=inverse,
                    )


def enumerate_field_animation_tiles(tileset_filename: str) -> Iterator[TileSpec]:
    """Enumerate field tiles with animation offsets (single tiles, no composite)."""
    props = get_tile_props()

    for tile_id in props.field_tiles:
        for tile_width in TILE_SIZES:
            tile_height = get_tile_height_for_width(tile_width)

            for inverse in INVERSE_STATES:
                # Field animation uses y_offset_fg
                for y_off in range(1, 16):  # 1-15, 0 already in single tiles
                    yield TileSpec(
                        tileset_filename=tileset_filename,
                        tile_id=tile_id,
                        bg_tile_id=None,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        tile_darkness=0,  # Fields are bright
                        x_offset_bg=0,
                        y_offset_bg=0,
                        x_offset_fg=0,
                        y_offset_fg=y_off,
                        inverse=inverse,
                    )


def enumerate_character_tiles(charset_filename: str, num_chars: int = 256) -> Iterator[CharSpec]:
    """Enumerate all character tiles for a charset."""
    for char_id in range(num_chars):
        for char_width in CHAR_SIZES:
            char_height = get_char_height_for_width(char_width)
            yield CharSpec(
                charset_filename=charset_filename,
                char_id=char_id,
                char_width=char_width,
                char_height=char_height,
            )


def make_ansi_text_from_image(ref_image, tile_width: int, tile_height: int) -> list[str]:
    """Convert a PIL image to ANSI text using chafa."""
    img_byte_arr = io.BytesIO()
    ref_image.save(img_byte_arr, format="PNG")

    chafa_cmd_args = [
        CHAFA_BIN,
        *CHAFA_EXTRA_ARGS,
        "--size",
        f"{tile_width}x{tile_height + 1}",
        "-",
    ]
    ans = subprocess.check_output(
        chafa_cmd_args, input=img_byte_arr.getvalue()
    ).decode()
    lines = ans.splitlines()

    # Remove preceding and trailing hide/show cursor attributes
    return [lines[0][CHAFA_TRIM_START:]] + lines[1:-1]


def generate_tile(spec: TileSpec, tile_data: list) -> tuple[str, str]:
    """Generate a single tile and return (cache_key, content)."""
    # Get foreground image
    fg_image = make_image_from_pixels(pixels=tile_data[spec.tile_id])

    # Get background image if compositing
    bg_image = None
    if spec.bg_tile_id is not None:
        bg_image = make_image_from_pixels(pixels=tile_data[spec.bg_tile_id])

    # Apply darkness to both layers
    bg_image = apply_darkness(bg_image, spec.tile_darkness)
    fg_image = apply_darkness(fg_image, spec.tile_darkness)

    # Apply offsets
    bg_image = apply_offsets(bg_image, spec.x_offset_bg, spec.y_offset_bg)
    fg_image = apply_offsets(fg_image, spec.x_offset_fg, spec.y_offset_fg)

    # Composite if we have a background
    ref_image = fg_image
    if bg_image and fg_image:
        ref_image = apply_composite(bg_image, fg_image)

    # Apply inverse for spell effects
    if spec.inverse:
        ref_image = apply_inverse(ref_image)

    # Convert to ANSI
    ansi_lines = make_ansi_text_from_image(ref_image, spec.tile_width, spec.tile_height)

    return spec.cache_key, '\n'.join(ansi_lines)


def generate_char(spec: CharSpec, char_data: list) -> tuple[str, str]:
    """Generate a single character tile and return (cache_key, content)."""
    # Get character image
    char_image = make_image_from_pixels(pixels=char_data[spec.char_id])

    # Convert to ANSI
    ansi_lines = make_ansi_text_from_image(char_image, spec.char_width, spec.char_height)

    return spec.cache_key, '\n'.join(ansi_lines)


def load_world_data():
    """Load world.yaml configuration."""
    world_yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "qwack", "dat", "world.yaml"
    )
    with open(world_yaml_path, "r") as f:
        return yaml.safe_load(f)


def process_work_item(item):
    """Worker function to generate a single tile/char (must be at module level for pickling)."""
    item_type, spec, data = item
    try:
        if item_type == 'tile':
            return generate_tile(spec, data)
        else:
            return generate_char(spec, data)
    except Exception as e:
        return None, str(e)


def get_existing_cache_keys(cache_zip_path: str) -> set:
    """Get all existing cache keys from the zip file."""
    if not os.path.exists(cache_zip_path):
        return set()

    try:
        with zipfile.ZipFile(cache_zip_path, 'r') as zf:
            return set(zf.namelist())
    except (zipfile.BadZipFile, IOError):
        return set()


def main():
    parser = argparse.ArgumentParser(description="Pre-generate tiles for the tile cache")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=1000, help="Tiles per batch save")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate all tiles even if already cached")
    parser.add_argument("--tileset", type=str, help="Generate only for specific tileset")
    parser.add_argument("--charset", type=str, help="Generate only for specific charset")
    parser.add_argument("--default-tileset", action="store_true",
                        help="Generate only for the default tileset (from world.yaml)")
    parser.add_argument("--default-charset", action="store_true",
                        help="Generate only for the default charset (from world.yaml)")
    parser.add_argument("--type", choices=["single", "composite", "confusion", "water", "field", "char", "all"],
                        default="all", help="Type of tiles to generate")
    parser.add_argument("--dry-run", action="store_true", help="Count tiles without generating")
    parser.add_argument("--show-props", action="store_true", help="Show extracted tile properties")
    parser.add_argument("--default-sizes", action="store_true",
                        help="Only generate default tile/char sizes (16 and 8)")
    parser.add_argument("--no-confusion", action="store_true",
                        help="Skip confusion spell offset tiles")
    args = parser.parse_args()

    # Show tile properties if requested
    if args.show_props:
        props = get_tile_props()
        print(props)
        return

    # Load configuration
    world_data = load_world_data()
    tilesets = world_data["Tilesets"]
    charsets = world_data["Charsets"]

    # Override sizes if --default-sizes is set
    global TILE_SIZES, CHAR_SIZES
    if args.default_sizes:
        default_tile_size = world_data.get("DEFAULT_TILE_SIZE", 16)
        default_char_size = world_data.get("DEFAULT_CHAR_SIZE", 8)
        TILE_SIZES = (default_tile_size,)
        CHAR_SIZES = (default_char_size,)
        print(f"Using default sizes only: tile={default_tile_size}, char={default_char_size}")

    # Filter tilesets/charsets if specified
    if args.default_tileset:
        default_tileset = world_data.get("DEFAULT_TILESET", "jsteele-shapes.ega")
        tilesets = [ts for ts in tilesets if ts["filename"] == default_tileset]
        print(f"Using default tileset: {default_tileset}")
    elif args.tileset:
        tilesets = [ts for ts in tilesets if ts["filename"] == args.tileset]

    if args.default_charset:
        default_charset = world_data.get("DEFAULT_CHARSET", "CHARSET.VGA")
        charsets = [cs for cs in charsets if cs["filename"] == default_charset]
        print(f"Using default charset: {default_charset}")
    elif args.charset:
        charsets = [cs for cs in charsets if cs["filename"] == args.charset]

    # Cache file path
    cache_zip_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "qwack", "tileset_cache.zip"
    )

    # Get existing cache keys
    existing_keys = set()
    if not args.regenerate:
        print("Loading existing cache keys...")
        existing_keys = get_existing_cache_keys(cache_zip_path)
        print(f"Found {len(existing_keys)} existing tiles in cache")
        if existing_keys and args.dry_run:
            # Show sample keys for debugging
            sample = list(existing_keys)[:3]
            print(f"  Sample existing keys: {sample}")

    # Collect all tiles to generate
    all_tile_specs = []
    all_char_specs = []

    print("Enumerating tiles to generate...")

    for tileset in tilesets:
        tileset_filename = tileset["filename"]
        print(f"  Tileset: {tileset_filename}")

        if args.type in ("single", "all"):
            for spec in enumerate_single_tiles(tileset_filename):
                if args.regenerate or spec.cache_key not in existing_keys:
                    all_tile_specs.append((spec, tileset))

        if args.type in ("composite", "all"):
            for spec in enumerate_composite_tiles(tileset_filename):
                if args.regenerate or spec.cache_key not in existing_keys:
                    all_tile_specs.append((spec, tileset))

        if args.type in ("confusion", "all") and not args.no_confusion:
            for spec in enumerate_confusion_tiles(tileset_filename):
                if args.regenerate or spec.cache_key not in existing_keys:
                    all_tile_specs.append((spec, tileset))

        if args.type in ("water", "all"):
            for spec in enumerate_water_animation_tiles(tileset_filename):
                if args.regenerate or spec.cache_key not in existing_keys:
                    all_tile_specs.append((spec, tileset))

        if args.type in ("field", "all"):
            for spec in enumerate_field_animation_tiles(tileset_filename):
                if args.regenerate or spec.cache_key not in existing_keys:
                    all_tile_specs.append((spec, tileset))

    if args.type in ("char", "all"):
        for charset in charsets:
            charset_filename = charset["filename"]
            # Load charset to determine number of characters
            charset_data = load_charset(charset)
            num_chars = len(charset_data)
            print(f"  Charset: {charset_filename} ({num_chars} chars)")

            for spec in enumerate_character_tiles(charset_filename, num_chars):
                if args.regenerate or spec.cache_key not in existing_keys:
                    all_char_specs.append((spec, charset))

    total_tiles = len(all_tile_specs)
    total_chars = len(all_char_specs)
    total = total_tiles + total_chars

    print(f"\nTiles to generate: {total_tiles}")
    print(f"Characters to generate: {total_chars}")
    print(f"Total: {total}")

    if args.dry_run:
        print("\nDry run - not generating tiles")
        return

    if total == 0:
        print("\nNothing to generate!")
        return

    # Load tile data for each tileset
    tileset_data = {}
    for tileset in tilesets:
        print(f"Loading tileset: {tileset['filename']}")
        tileset_data[tileset["filename"]] = load_tileset(tileset)

    # Load char data for each charset
    charset_data = {}
    for charset in charsets:
        print(f"Loading charset: {charset['filename']}")
        charset_data[charset["filename"]] = load_charset(charset)

    # Generate tiles using multiprocessing
    print(f"\nGenerating {total} tiles with {args.workers} workers...")

    generated = 0
    errors = 0

    # Prepare work items with tile data included
    def make_tile_work_items():
        for spec, tileset in all_tile_specs:
            yield ('tile', spec, tileset_data[tileset["filename"]])

    def make_char_work_items():
        for spec, charset in all_char_specs:
            yield ('char', spec, charset_data[charset["filename"]])

    # Combine all work items
    import itertools
    all_work = list(itertools.chain(make_tile_work_items(), make_char_work_items()))

    # Process in parallel and write to zip
    with zipfile.ZipFile(cache_zip_path, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit work in batches to avoid memory issues
            for cache_key, content in executor.map(process_work_item, all_work, chunksize=100):
                if cache_key is None:
                    errors += 1
                    if errors <= 10:
                        print(f"  Error: {content}")
                else:
                    zf.writestr(cache_key, content)
                    generated += 1

                    if generated % 1000 == 0:
                        print(f"  Generated {generated}/{total} tiles ({100*generated/total:.1f}%)")

    print(f"\nDone! Generated {generated} tiles.")


if __name__ == "__main__":
    main()
