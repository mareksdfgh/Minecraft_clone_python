

from __future__ import annotations

import concurrent.futures
import functools
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import pyglet
from pyglet import gl
from pyglet.graphics import Batch, Group
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse

CHUNK_SIZE = 16
WORLD_HEIGHT = 180
WORLD_SEED = 4177
SEA_LEVEL = 15
RIVER_BED_LEVEL = SEA_LEVEL - 3
WATER_SURFACE_LEVEL = SEA_LEVEL + 1
RIVER_SURFACE_LEVEL = SEA_LEVEL + 2
WATER_LEVEL_BOOST = 4
MIN_RIVER_DEPTH = 3
TREE_MARGIN = 4
RENDER_DISTANCE = 9
MIN_RENDER_DISTANCE = 4
MAX_RENDER_DISTANCE = 64
CHUNKS_PER_FRAME = 1
MESHES_PER_FRAME = 1
CHUNK_COMPLETIONS_PER_FRAME = 2
MESH_UPLOADS_PER_FRAME = 1
MESH_UPLOAD_INTERVAL = 2
STREAM_TIME_BUDGET = 0.0025
STREAM_WHILE_MOVING = True
USE_PROCESS_WORKERS = True
INITIAL_LOAD_RADIUS = 1
CPU_COUNT = os.cpu_count() or 4
WORKER_COUNT = max(2, min(2, max(1, CPU_COUNT - 1)))
CHUNK_WORKER_COUNT = 1
MESH_WORKER_COUNT = max(1, WORKER_COUNT - CHUNK_WORKER_COUNT)
MAX_PENDING_CHUNK_JOBS = max(CHUNKS_PER_FRAME * 4, CHUNK_WORKER_COUNT * 8)
MAX_PENDING_MESH_JOBS = max(MESHES_PER_FRAME * 4, MESH_WORKER_COUNT * 6)
TILE_SIZE = 16
ATLAS_COLUMNS = 4
RAY_STEP = 0.05
MAX_REACH = 6.0
EPSILON = 0.0001
MOVE_COLLISION_STEP = 0.12

AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
SAND = 4
LOG = 5
LEAVES = 6
BEDROCK = 7
WATER = 8
SNOW = 9
RED_SAND = 10
PLANKS = 11
DARK_PLANKS = 12
COBBLESTONE = 13
BRICKS = 14
GLASS = 15
CLAY = 16
THATCH = 17
CACTUS = 18
DRY_GRASS = 19
SHORT_GRASS = 20
TALL_GRASS = 21
PINE_LEAVES = 22
BIRCH_LOG = 23
BIRCH_LEAVES = 24
ACACIA_LOG = 25
ACACIA_LEAVES = 26
DEAD_BUSH = 27
FLOWERS = 28

INVENTORY_BLOCKS = [
    GRASS,
    DIRT,
    STONE,
    COBBLESTONE,
    BRICKS,
    SAND,
    RED_SAND,
    CLAY,
    LOG,
    PLANKS,
    DARK_PLANKS,
    THATCH,
    LEAVES,
    SNOW,
    GLASS,
    CACTUS,
    DRY_GRASS,
    SHORT_GRASS,
    TALL_GRASS,
    PINE_LEAVES,
    BIRCH_LOG,
    BIRCH_LEAVES,
    ACACIA_LOG,
    ACACIA_LEAVES,
    DEAD_BUSH,
    FLOWERS,
]
HOTBAR_SIZE = 9

BLOCK_NAMES = {
    GRASS: "Grass",
    DIRT: "Dirt",
    STONE: "Stone",
    SAND: "Sand",
    LOG: "Log",
    LEAVES: "Leaves",
    BEDROCK: "Bedrock",
    WATER: "Water",
    SNOW: "Snow",
    RED_SAND: "Red Sand",
    PLANKS: "Planks",
    DARK_PLANKS: "Dark Planks",
    COBBLESTONE: "Cobblestone",
    BRICKS: "Bricks",
    GLASS: "Glass",
    CLAY: "Clay",
    THATCH: "Thatch",
    CACTUS: "Cactus",
    DRY_GRASS: "Dry Grass",
    SHORT_GRASS: "Short Grass",
    TALL_GRASS: "Tall Grass",
    PINE_LEAVES: "Pine Leaves",
    BIRCH_LOG: "Birch Log",
    BIRCH_LEAVES: "Birch Leaves",
    ACACIA_LOG: "Acacia Log",
    ACACIA_LEAVES: "Acacia Leaves",
    DEAD_BUSH: "Dead Bush",
    FLOWERS: "Flowers",
}

TEX_GRASS_TOP = 0
TEX_GRASS_SIDE = 1
TEX_DIRT = 2
TEX_STONE = 3
TEX_SAND = 4
TEX_LOG_SIDE = 5
TEX_LOG_TOP = 6
TEX_LEAVES = 7
TEX_BEDROCK = 8
TEX_WATER = 9
TEX_SNOW = 10
TEX_RED_SAND = 11
TEX_PLANKS = 12
TEX_DARK_PLANKS = 13
TEX_COBBLESTONE = 14
TEX_BRICKS = 15
TEX_GLASS = 16
TEX_CLAY = 17
TEX_THATCH = 18
TEX_CACTUS = 19
TEX_DRY_GRASS_TOP = 20
TEX_DRY_GRASS_SIDE = 21
TEX_SHORT_GRASS = 22
TEX_TALL_GRASS = 23
TEX_PINE_LEAVES = 24
TEX_BIRCH_LOG_SIDE = 25
TEX_BIRCH_LOG_TOP = 26
TEX_BIRCH_LEAVES = 27
TEX_ACACIA_LOG_SIDE = 28
TEX_ACACIA_LOG_TOP = 29
TEX_ACACIA_LEAVES = 30
TEX_DEAD_BUSH = 31
TEX_FLOWERS = 32
ATLAS_TILE_COUNT = 33
ATLAS_ROWS = math.ceil(ATLAS_TILE_COUNT / ATLAS_COLUMNS)
ATLAS_WIDTH = ATLAS_COLUMNS * TILE_SIZE
ATLAS_HEIGHT = ATLAS_ROWS * TILE_SIZE

VERTEX_SHADER = """#version 330 core
    in vec3 POSITION;
    in vec3 NORMAL;
    in vec2 TEXCOORD_0;

    out vec2 texcoord_0;
    out float shade;
    out vec3 world_pos;
    out vec3 normal_world;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;
    uniform float time;

    void main()
    {
        vec3 pos = POSITION;
        float plant_top = fract(POSITION.y);
        if (plant_top > 0.01) {
            float wave = sin(POSITION.x * 0.35 + POSITION.z * 0.27 + time * 1.6) * 0.08
                       + sin(POSITION.z * 0.21 - time * 1.10) * 0.04;
            pos.x += wave * plant_top;
            pos.z += wave * 0.6 * plant_top;
        }
        vec4 world = model * vec4(pos, 1.0);
        vec3 normal = normalize(mat3(model) * NORMAL);
        vec3 sun = normalize(vec3(-0.36, 0.82, 0.44));
        float directional = max(dot(normal, sun), 0.0);
        float sky = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
        float horizon = 1.0 - abs(normal.y) * 0.28;
        float face_shadow = normal.y < -0.50 ? 0.55 : 1.0;
        shade = (0.20 + directional * 0.70 + sky * 0.13 + horizon * 0.04) * face_shadow;
        texcoord_0 = TEXCOORD_0;
        world_pos = world.xyz;
        normal_world = normal;
        gl_Position = window.projection * window.view * world;
    }
"""

FRAGMENT_SHADER = """#version 330 core
    in vec2 texcoord_0;
    in float shade;
    in vec3 world_pos;
    in vec3 normal_world;

    out vec4 final_colors;

    uniform sampler2D atlas;
    uniform float time;
    uniform vec3 camera_pos;

    float hash21(vec2 p)
    {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
    }

    float value_noise_2d(vec2 p)
    {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = hash21(i);
        float b = hash21(i + vec2(1.0, 0.0));
        float c = hash21(i + vec2(0.0, 1.0));
        float d = hash21(i + vec2(1.0, 1.0));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    void main()
    {
        vec4 texel = texture(atlas, texcoord_0);
        if (texel.a < 0.10) {
            discard;
        }
        vec3 warm_sun = vec3(1.12, 1.06, 0.96);
        vec3 cool_shadow = vec3(0.30, 0.36, 0.46);
        float cloud_shadow = value_noise_2d(world_pos.xz * 0.010 + vec2(time * 0.018, -time * 0.012));
        cloud_shadow = mix(0.82, 1.0, smoothstep(0.28, 0.72, cloud_shadow));
        vec3 lit = mix(cool_shadow, warm_sun, clamp(shade * cloud_shadow, 0.0, 1.0));
        vec3 color = texel.rgb * lit;
        float height_light = clamp((world_pos.y - 8.0) / 80.0, 0.0, 1.0) * 0.025;
        color += vec3(height_light);

        vec3 cell = fract(world_pos + vec3(0.002));
        float edge_x = min(cell.x, 1.0 - cell.x);
        float edge_y = min(cell.y, 1.0 - cell.y);
        float edge_z = min(cell.z, 1.0 - cell.z);
        float block_edge = 1.0 - smoothstep(0.020, 0.105, min(min(edge_x, edge_y), edge_z));
        float contact = (1.0 - max(normal_world.y, 0.0)) * block_edge * 0.12;
        float top_corner_ao = max(normal_world.y, 0.0) * (1.0 - smoothstep(0.018, 0.090, min(edge_x, edge_z))) * 0.10;
        color *= 1.0 - contact - top_corner_ao;

        float blue_dominance = texel.b - max(texel.r, texel.g) * 0.72;
        float water_mask = smoothstep(0.16, 0.34, blue_dominance) * (1.0 - smoothstep(0.40, 0.72, texel.r));
        float wave = sin(world_pos.x * 0.25 + time * 1.45) * 0.020;
        wave += sin(world_pos.z * 0.21 - time * 1.15) * 0.015;
        wave += sin((world_pos.x + world_pos.z) * 0.12 + time * 0.65) * 0.010;
        float facing_sky = max(normal_world.y, 0.0);
        vec3 water_deep = vec3(0.025, 0.120, 0.285);
        vec3 water_shallow = vec3(0.055, 0.255, 0.500);
        vec3 water_glint = vec3(0.22, 0.38, 0.52) * pow(clamp(facing_sky + wave + shade * 0.16, 0.0, 1.0), 6.0);
        vec3 water_color = mix(water_deep, water_shallow, clamp(shade * 0.75 + wave, 0.0, 1.0)) + water_glint;
        color = mix(color, water_color, water_mask * 0.68);

        float fog = smoothstep(700.0, 1180.0, length(world_pos.xz - camera_pos.xz));
        color = mix(color, vec3(0.56, 0.72, 0.88), fog * 0.38);
        color = pow(clamp(color, 0.0, 1.0), vec3(0.92));
        final_colors = vec4(color, texel.a);
    }
"""

@dataclass(frozen=True)
class FaceDef:
    offset: tuple[int, int, int]
    name: str
    normal: tuple[float, float, float]
    corners: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

FACES = (
    FaceDef(
        (0, 1, 0),
        "top",
        (0.0, 1.0, 0.0),
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
    ),
    FaceDef(
        (0, -1, 0),
        "bottom",
        (0.0, -1.0, 0.0),
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    ),
    FaceDef(
        (1, 0, 0),
        "east",
        (1.0, 0.0, 0.0),
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    ),
    FaceDef(
        (-1, 0, 0),
        "west",
        (-1.0, 0.0, 0.0),
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    ),
    FaceDef(
        (0, 0, 1),
        "south",
        (0.0, 0.0, 1.0),
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    ),
    FaceDef(
        (0, 0, -1),
        "north",
        (0.0, 0.0, -1.0),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    ),
)

@dataclass(frozen=True)
class Biome:
    name: str
    top_block: int
    tree_density: float
    base_height: float
    continent: float
    hills: float
    detail: float
    ridge: float
    dune: float = 0.0

PLAINS = Biome("Grassland", GRASS, 0.0030, SEA_LEVEL + 7, 5.0, 2.8, 1.0, 0.8)
FOREST = Biome("Temperate Forest", GRASS, 0.0550, SEA_LEVEL + 8, 6.0, 5.8, 1.8, 2.4)
DESERT = Biome("Desert", SAND, 0.0000, SEA_LEVEL + 4, 3.5, 1.0, 0.8, 0.5, dune=5.8)
OCEAN = Biome("Ocean", SAND, 0.0000, SEA_LEVEL - 7, 5.0, 1.5, 0.8, 0.2)
SAVANNA = Biome("Savanna", GRASS, 0.0140, SEA_LEVEL + 6, 4.8, 3.0, 1.2, 0.9)
HILLS = Biome("Epic Hills", GRASS, 0.0100, SEA_LEVEL + 24, 14.0, 23.0, 4.6, 9.0)
KYIV_LOWLAND = Biome("Kyiv Lowlands", GRASS, 0.0060, SEA_LEVEL + 3, 1.8, 0.8, 0.35, 0.2)
TAIGA = Biome("Snowy Taiga", SNOW, 0.0360, SEA_LEVEL + 10, 5.2, 5.4, 1.4, 2.2)
BADLANDS = Biome("Badlands", RED_SAND, 0.0000, SEA_LEVEL + 9, 4.4, 3.0, 1.0, 1.2, dune=3.0)
ALPINE = Biome("Alpine Peaks", SNOW, 0.0020, SEA_LEVEL + 46, 16.0, 34.0, 4.2, 15.0)
WETLAND = Biome("Wetland", GRASS, 0.0100, SEA_LEVEL + 1, 1.9, 0.9, 0.5, 0.1)
BIOMES = (PLAINS, FOREST, DESERT, OCEAN, SAVANNA, HILLS, KYIV_LOWLAND, TAIGA, BADLANDS, ALPINE, WETLAND)

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def hash_int(x: int, z: int, seed: int = WORLD_SEED) -> int:
    n = x * 374761393 + z * 668265263 + seed * 1442695041
    n = (n ^ (n >> 13)) * 1274126177
    return (n ^ (n >> 16)) & 0xFFFFFFFF

def hash_unit(x: int, z: int, seed: int = WORLD_SEED) -> float:
    return hash_int(x, z, seed) / 0xFFFFFFFF * 2.0 - 1.0

def value_noise(wx: float, wz: float, scale: float, seed: int) -> float:
    x = wx / scale
    z = wz / scale
    x0 = math.floor(x)
    z0 = math.floor(z)
    tx = smoothstep(x - x0)
    tz = smoothstep(z - z0)

    a = hash_unit(x0, z0, seed)
    b = hash_unit(x0 + 1, z0, seed)
    c = hash_unit(x0, z0 + 1, seed)
    d = hash_unit(x0 + 1, z0 + 1, seed)
    return lerp(lerp(a, b, tx), lerp(c, d, tx), tz)

@functools.lru_cache(maxsize=262_144)
def biome_weights(wx: int, wz: int) -> dict[Biome, float]:
    temperature = clamp((value_noise(wx, wz, 620.0, WORLD_SEED + 21) + 1.0) * 0.5, 0.0, 1.0)
    moisture = (value_noise(wx, wz, 560.0, WORLD_SEED + 22) + 1.0) * 0.5
    elevation_noise = (value_noise(wx, wz, 720.0, WORLD_SEED + 23) + 1.0) * 0.5
    continent_main = (value_noise(wx, wz, 3000.0, WORLD_SEED + 26) + 1.0) * 0.5
    continent_detail = (value_noise(wx, wz, 760.0, WORLD_SEED + 76) + 1.0) * 0.5
    continentalness = clamp(continent_main * 0.74 + continent_detail * 0.26, 0.0, 1.0)

    ocean = smoothstep(clamp((0.31 - continentalness) / 0.15, 0.0, 1.0)) * 3.80
    coast_fade = clamp(1.0 - ocean * 0.52, 0.0, 1.0)
    elevation = smoothstep(clamp((elevation_noise - 0.55) / 0.30, 0.0, 1.0))
    warm = smoothstep(clamp((temperature - 0.42) / 0.24, 0.0, 1.0))
    hot = smoothstep(clamp((temperature - 0.62) / 0.24, 0.0, 1.0))
    cold = smoothstep(clamp((0.38 - temperature) / 0.26, 0.0, 1.0))
    dry = smoothstep(clamp((0.52 - moisture) / 0.40, 0.0, 1.0))
    wet = smoothstep(clamp((moisture - 0.48) / 0.38, 0.0, 1.0))
    very_wet = smoothstep(clamp((moisture - 0.62) / 0.30, 0.0, 1.0))
    basin = smoothstep(clamp((0.54 - elevation_noise) / 0.34, 0.0, 1.0))
    temperate = 1.0 - abs(temperature - 0.50) * 1.9
    terrace_noise = (value_noise(wx, wz, 900.0, WORLD_SEED + 119) + 1.0) * 0.5
    broad_channel = kyiv_broad_river_strength(wx, wz)

    desert = hot * dry * coast_fade * 2.05 * (1.0 - elevation * 0.40)
    savanna = warm * (1.0 - wet * 0.62) * (1.0 - dry * 0.20) * coast_fade * 1.75 * (1.0 - elevation * 0.45)
    forest = wet * (1.0 - hot * dry * 0.75) * coast_fade * 1.18 * (1.0 - elevation * 0.30)
    plains = (0.25 + (1.0 - abs(moisture - 0.50) * 1.35)) * (1.0 - hot * 0.46) * coast_fade * 0.92 * (1.0 - elevation * 0.55)
    hills = elevation * coast_fade * 1.90 * (1.0 - cold * 0.25)
    kyiv_lowland = (basin * wet * 1.05 + broad_channel * 1.85) * max(temperate, 0.0) * coast_fade * (1.0 - dry * 0.85)
    taiga = cold * wet * coast_fade * 1.45 * (1.0 - elevation * 0.28)
    badlands = hot * dry * coast_fade * 1.75 * smoothstep(clamp((terrace_noise - 0.26) / 0.36, 0.0, 1.0))
    alpine = elevation * cold * coast_fade * 1.90
    wetland = very_wet * basin * coast_fade * 2.25 * (1.0 - hot * 0.35)

    scores = {
        PLAINS: max(plains, 0.05),
        FOREST: max(forest, 0.0),
        DESERT: max(desert, 0.0),
        OCEAN: max(ocean, 0.0),
        SAVANNA: max(savanna, 0.0),
        HILLS: max(hills, 0.0),
        KYIV_LOWLAND: max(kyiv_lowland, 0.0),
        TAIGA: max(taiga, 0.0),
        BADLANDS: max(badlands, 0.0),
        ALPINE: max(alpine, 0.0),
        WETLAND: max(wetland, 0.0),
    }
    total = sum(scores.values()) or 1.0
    return {biome: score / total for biome, score in scores.items()}

@functools.lru_cache(maxsize=262_144)
def dominant_biome(wx: int, wz: int) -> Biome:
    weights = biome_weights(wx, wz)
    return max(weights, key=weights.get)

def biome_name_at(wx: int, wz: int) -> str:
    return dominant_biome(wx, wz).name

def biome_minimap_color(biome: Biome) -> tuple[int, int, int]:
    if biome is OCEAN:
        return (42, 111, 180)
    if biome is DESERT:
        return (206, 185, 106)
    if biome is KYIV_LOWLAND:
        return (88, 154, 91)
    if biome is FOREST:
        return (38, 121, 61)
    if biome is SAVANNA:
        return (154, 161, 82)
    if biome is HILLS:
        return (108, 132, 92)
    if biome is TAIGA:
        return (124, 158, 142)
    if biome is BADLANDS:
        return (178, 94, 58)
    if biome is ALPINE:
        return (184, 194, 198)
    if biome is WETLAND:
        return (62, 132, 101)
    return (92, 171, 73)

@functools.lru_cache(maxsize=1_048_576)
def biome_height(wx: int, wz: int, biome: Biome) -> float:
    continent = value_noise(wx, wz, 430.0, WORLD_SEED + 1)
    hills = value_noise(wx, wz, 150.0, WORLD_SEED + 2)
    detail = value_noise(wx, wz, 38.0, WORLD_SEED + 3)
    micro = value_noise(wx, wz, 16.0, WORLD_SEED + 13)
    ridge_a = 1.0 - abs(value_noise(wx, wz, 230.0, WORLD_SEED + 4))
    ridge_b = 1.0 - abs(value_noise(wx, wz, 86.0, WORLD_SEED + 14))
    ridges = ridge_a * ridge_a * 0.65 + ridge_b * ridge_b * 0.35
    dune_warp = value_noise(wx, wz, 260.0, WORLD_SEED + 24) * 36.0
    dune_cross = value_noise(wx, wz, 140.0, WORLD_SEED + 25) * 15.0
    dunes = math.sin((wx + dune_warp) * 0.075 + math.sin(wz * 0.010) * 1.4)
    dunes += math.sin((wx * 0.55 + wz * 0.83 + dune_cross) * 0.046) * 0.75
    dunes += max(0.0, math.sin((wx - wz + dune_warp) * 0.024)) * 0.9

    if biome is OCEAN:
        shelf = value_noise(wx, wz, 260.0, WORLD_SEED + 27)
        trench = max(0.0, 1.0 - abs(value_noise(wx, wz, 180.0, WORLD_SEED + 28))) ** 2
        return biome.base_height + continent * 2.0 + shelf * 2.0 + detail * 0.35 - trench * 4.0

    if biome is KYIV_LOWLAND:
        flat_roll = value_noise(wx, wz, 520.0, WORLD_SEED + 121) * 1.1
        bank_rise = max(0.0, value_noise(wx, wz, 180.0, WORLD_SEED + 122)) * 1.6
        meadow = value_noise(wx, wz, 52.0, WORLD_SEED + 123) * 0.30
        return biome.base_height + continent * biome.continent + flat_roll + bank_rise + meadow

    if biome is WETLAND:
        hummocks = max(0.0, value_noise(wx, wz, 70.0, WORLD_SEED + 124)) * 1.2
        soggy_flats = value_noise(wx, wz, 240.0, WORLD_SEED + 125) * 0.8
        return biome.base_height + continent * biome.continent + hummocks + soggy_flats + detail * 0.25

    if biome is TAIGA:
        rolling = hills * biome.hills * 0.70 + ridges * biome.ridge * 0.70
        snowy_drifts = max(0.0, value_noise(wx, wz, 90.0, WORLD_SEED + 126)) * 1.5
        return biome.base_height + continent * biome.continent + rolling + snowy_drifts + detail * biome.detail

    if biome is BADLANDS:
        bands = math.floor((value_noise(wx, wz, 95.0, WORLD_SEED + 127) + 1.0) * 4.0) * 1.25
        mesas = max(0.0, value_noise(wx, wz, 210.0, WORLD_SEED + 128)) ** 2 * 8.0
        gullies = max(0.0, 1.0 - abs(value_noise(wx, wz, 48.0, WORLD_SEED + 129))) * -2.5
        return biome.base_height + continent * biome.continent + hills * 1.2 + bands + mesas + gullies + dunes * biome.dune

    if biome is ALPINE:
        massif = value_noise(wx, wz, 260.0, WORLD_SEED + 130) * 16.0
        sharp_ridge = (1.0 - abs(value_noise(wx, wz, 90.0, WORLD_SEED + 131))) ** 2
        tooth = (1.0 - abs(value_noise(wx, wz, 30.0, WORLD_SEED + 132))) ** 3
        snow_shelf = max(0.0, value_noise(wx, wz, 55.0, WORLD_SEED + 142)) * 7.0
        glacier_cut = max(0.0, -value_noise(wx, wz, 150.0, WORLD_SEED + 143)) * 7.0
        return biome.base_height + continent * biome.continent + hills * biome.hills * 0.95 + massif + sharp_ridge * 34.0 + tooth * 20.0 + snow_shelf - glacier_cut

    if biome is DESERT:
        dune_lines = dunes * biome.dune
        dry_plateaus = max(0.0, value_noise(wx, wz, 320.0, WORLD_SEED + 29)) ** 2 * 5.8
        return biome.base_height + continent * 2.2 + hills * 0.7 + detail * 0.45 + dune_lines + dry_plateaus

    if biome is FOREST:
        rolling = hills * biome.hills * 0.75 + ridges * biome.ridge * 0.55
        hollows = max(0.0, -value_noise(wx, wz, 220.0, WORLD_SEED + 37)) * 2.7
        return biome.base_height + continent * biome.continent + rolling + detail * biome.detail - hollows

    if biome is SAVANNA:
        broad_rises = max(0.0, value_noise(wx, wz, 260.0, WORLD_SEED + 49)) * 3.0
        dry_waves = math.sin((wx * 0.020) + value_noise(wx, wz, 180.0, WORLD_SEED + 50) * 1.5) * 1.4
        return biome.base_height + continent * biome.continent + hills * 1.1 + broad_rises + dry_waves + detail * 0.7

    if biome is HILLS:
        tight_hills = value_noise(wx, wz, 46.0, WORLD_SEED + 53)
        cliff_noise = value_noise(wx, wz, 28.0, WORLD_SEED + 54)
        epic_ridges = (1.0 - abs(value_noise(wx, wz, 82.0, WORLD_SEED + 55))) ** 2
        broad_mass = max(0.0, value_noise(wx, wz, 210.0, WORLD_SEED + 56)) * 16.0
        cliff_steps = max(0.0, cliff_noise - 0.08) * 18.0
        rolling = hills * biome.hills * 1.05 + tight_hills * 11.0 + ridges * biome.ridge * 1.45
        crests = max(0.0, value_noise(wx, wz, 115.0, WORLD_SEED + 57)) * 16.0
        return biome.base_height + continent * biome.continent + rolling + broad_mass + crests + cliff_steps + epic_ridges * 28.0 + detail * biome.detail + micro * 3.0

    meadow_swells = max(0.0, value_noise(wx, wz, 240.0, WORLD_SEED + 38)) * 1.6
    return biome.base_height + continent * biome.continent + hills * 1.1 + detail * 0.8 + ridges * 0.45 + meadow_swells

@functools.lru_cache(maxsize=262_144)
def river_strength(wx: int, wz: int) -> float:
    warp_x = value_noise(wx, wz, 380.0, WORLD_SEED + 36) * 130.0
    warp_z = value_noise(wx, wz, 380.0, WORLD_SEED + 37) * 130.0
    sx = wx + warp_x
    sz = wz + warp_z
    n_c = value_noise(sx, sz, 820.0, WORLD_SEED + 31)
    n_dx = value_noise(sx + 2.0, sz, 820.0, WORLD_SEED + 31) - n_c
    n_dz = value_noise(sx, sz + 2.0, 820.0, WORLD_SEED + 31) - n_c
    grad = math.sqrt(n_dx * n_dx + n_dz * n_dz) * 0.5 + 1e-6
    distance = abs(n_c) / grad
    river_core = 5.0
    bank_edge = 26.0
    if distance >= bank_edge:
        return 0.0
    if distance <= river_core:
        strength = 1.0
    else:
        t = (distance - river_core) / (bank_edge - river_core)
        strength = 1.0 - smoothstep(t)
    ocean_weight = biome_weights(wx, wz)[OCEAN]
    return clamp(strength * (1.0 - ocean_weight * 0.85), 0.0, 1.0)

@functools.lru_cache(maxsize=262_144)
def kyiv_broad_river_strength(wx: int, wz: int) -> float:
    warp = value_noise(wx, wz, 700.0, WORLD_SEED + 133) * 90.0
    channel_center = math.sin((wx + warp) * 0.0085) * 150.0
    channel_center += value_noise(wx, wz, 1450.0, WORLD_SEED + 134) * 90.0
    distance = abs(wz - channel_center)
    core = 18.0
    bank = 70.0
    if distance >= bank:
        return 0.0
    if distance <= core:
        return 1.0
    return 1.0 - smoothstep((distance - core) / (bank - core))

@functools.lru_cache(maxsize=262_144)
def lake_strength(wx: int, wz: int) -> float:
    basin = (value_noise(wx, wz, 1080.0, WORLD_SEED + 70) + 1.0) * 0.5
    pocket = (value_noise(wx, wz, 380.0, WORLD_SEED + 71) + 1.0) * 0.5
    shore_shape = 1.0 - abs(value_noise(wx, wz, 240.0, WORLD_SEED + 72))
    weights = biome_weights(wx, wz)
    wet_biomes = weights[FOREST] + weights[PLAINS] * 0.55 + weights[KYIV_LOWLAND] * 1.25 + weights[WETLAND] * 1.45
    dry_penalty = weights[DESERT] * 0.80 + weights[OCEAN]
    lake = smoothstep(clamp((basin - 0.48) / 0.34, 0.0, 1.0))
    lake *= smoothstep(clamp((pocket - 0.34) / 0.44, 0.0, 1.0))
    lake *= smoothstep(clamp((shore_shape - 0.12) / 0.62, 0.0, 1.0))
    lake *= clamp(wet_biomes + 0.30 - dry_penalty, 0.0, 1.0)
    small_lake_noise = (value_noise(wx, wz, 82.0, WORLD_SEED + 135) + 1.0) * 0.5
    small_lake_shape = 1.0 - abs(value_noise(wx, wz, 46.0, WORLD_SEED + 136))
    kyiv_lakes = smoothstep(clamp((small_lake_noise - 0.42) / 0.30, 0.0, 1.0))
    kyiv_lakes *= smoothstep(clamp((small_lake_shape - 0.04) / 0.40, 0.0, 1.0))
    kyiv_lakes *= weights[KYIV_LOWLAND] * 1.15
    wetland_pools = smoothstep(clamp((small_lake_noise - 0.42) / 0.36, 0.0, 1.0)) * weights[WETLAND] * 0.65
    lake = max(lake, kyiv_lakes, wetland_pools)
    return clamp(lake, 0.0, 1.0)

@functools.lru_cache(maxsize=262_144)
def effective_river_strength(wx: int, wz: int) -> float:
    weights = biome_weights(wx, wz)
    kyiv_river = kyiv_broad_river_strength(wx, wz) * clamp(weights[KYIV_LOWLAND] * 1.85, 0.0, 1.0)
    return max(river_strength(wx, wz), kyiv_river)

@functools.lru_cache(maxsize=262_144)
def canyon_factor(wx: int, wz: int) -> float:
    return 0.0

@functools.lru_cache(maxsize=262_144)
def sub_biome_relief(wx: int, wz: int) -> float:
    weights = biome_weights(wx, wz)
    bumps = value_noise(wx, wz, 30.0, WORLD_SEED + 110)
    fine = value_noise(wx, wz, 14.0, WORLD_SEED + 111)
    relief = bumps * 1.4 + fine * 0.7
    biome_factor = (
        weights[PLAINS] * 0.8
        + weights[FOREST] * 1.2
        + weights[SAVANNA] * 1.0
        + weights[HILLS] * 2.9
        + weights[DESERT] * 0.5
        + weights[KYIV_LOWLAND] * 0.18
        + weights[TAIGA] * 1.1
        + weights[BADLANDS] * 0.9
        + weights[ALPINE] * 2.4
        + weights[WETLAND] * 0.16
        + weights[OCEAN] * 0.0
    )
    return relief * biome_factor

@functools.lru_cache(maxsize=262_144)
def base_terrain_height(wx: int, wz: int) -> int:
    weights = biome_weights(wx, wz)
    height = sum(weight * biome_height(wx, wz, biome) for biome, weight in weights.items())
    height += sub_biome_relief(wx, wz)
    return int(clamp(round(height), 3, WORLD_HEIGHT - 5))

@functools.lru_cache(maxsize=262_144)
def eroded_base_height(wx: int, wz: int) -> float:
    center = base_terrain_height(wx, wz)
    neighbors = (
        base_terrain_height(wx + 4, wz)
        + base_terrain_height(wx - 4, wz)
        + base_terrain_height(wx, wz + 4)
        + base_terrain_height(wx, wz - 4)
    ) * 0.10
    diagonals = (
        base_terrain_height(wx + 8, wz + 8)
        + base_terrain_height(wx - 8, wz + 8)
        + base_terrain_height(wx + 8, wz - 8)
        + base_terrain_height(wx - 8, wz - 8)
    ) * 0.05
    return center * 0.40 + neighbors + diagonals

@functools.lru_cache(maxsize=262_144)
def terrain_height(wx: int, wz: int) -> int:
    raw_base = base_terrain_height(wx, wz)
    river = effective_river_strength(wx, wz)
    lake = lake_strength(wx, wz)
    weights = biome_weights(wx, wz)
    ocean_weight = weights[OCEAN]
    kyiv_weight = weights[KYIV_LOWLAND]
    lake_shore = smoothstep(clamp((lake - 0.22) / 0.42, 0.0, 1.0))
    coast_shelf = smoothstep(clamp((ocean_weight - 0.035) / 0.27, 0.0, 1.0))
    local_average = eroded_base_height(wx, wz)
    roughness = abs(raw_base - local_average)
    thermal_erosion = smoothstep(clamp((roughness - 2.0) / 14.0, 0.0, 1.0))
    river_valley = smoothstep(clamp((river - 0.05) / 0.40, 0.0, 1.0))
    water_erosion = max(river_valley, lake_shore, coast_shelf)
    erosion_mix = clamp(thermal_erosion * (0.28 + water_erosion * 0.50), 0.0, 0.78)
    raw = lerp(raw_base, local_average, erosion_mix)

    river_flat_zone = smoothstep(clamp((river - 0.005) / 0.45, 0.0, 1.0))
    river_deep_zone = smoothstep(clamp((river - 0.55) / 0.30, 0.0, 1.0))
    floodplain_target = SEA_LEVEL + 1.0 - kyiv_weight * 1.4 + value_noise(wx, wz, 240.0, WORLD_SEED + 78) * 0.6
    river_flatten = river_flat_zone * max(raw - floodplain_target, 0.0) * (0.96 + kyiv_weight * 0.18)
    river_dip = river_deep_zone * (4.8 + kyiv_weight * 3.4)

    lake_proximity = smoothstep(clamp((lake - 0.10) / 0.50, 0.0, 1.0))
    lake_target = SEA_LEVEL + 1.0 - kyiv_weight * 0.8
    lake_flatten = lake_proximity * max(raw - lake_target, 0.0) * (0.88 + kyiv_weight * 0.08)
    lake_basin = lake_shore * (4.5 + kyiv_weight * 1.7 + max(raw - SEA_LEVEL, 0) * 0.28)
    lake_floor = SEA_LEVEL - 5.5 + value_noise(wx, wz, 70.0, WORLD_SEED + 73) * 1.3
    lake_cut = smoothstep(clamp((lake - 0.46) / 0.34, 0.0, 1.0)) * max(raw - lake_floor, 0.0)

    coast_target = SEA_LEVEL + 4.0 + value_noise(wx, wz, 220.0, WORLD_SEED + 75) * 1.4
    coast_cut = coast_shelf * max(raw - coast_target, 0.0) * 0.90
    shore_smoothing = ocean_weight * max(raw - (SEA_LEVEL - 7), 0) * 0.34
    ocean_basin = smoothstep(clamp((ocean_weight - 0.30) / 0.45, 0.0, 1.0)) * 16.0
    coast_profile = SEA_LEVEL + 1.0 + (clamp((0.20 - ocean_weight) / 0.18, 0.0, 1.0) ** 1.35) * 4.5
    coast_terrace = smoothstep(clamp((ocean_weight - 0.055) / 0.15, 0.0, 1.0)) * max(raw - coast_profile, 0.0) * 0.88
    desert_blown_sand = weights[DESERT] * (1.0 - river_valley) * value_noise(wx, wz, 60.0, WORLD_SEED + 40) * 0.8

    away_from_water = 1.0 - max(river_flat_zone, lake_proximity, coast_shelf * 0.8)
    sub_var = 0.0
    if weights[HILLS] > 0.30:
        h_sub = value_noise(wx, wz, 16.0, WORLD_SEED + 98)
        if h_sub > 0.20:
            sub_var += weights[HILLS] * (h_sub - 0.20) * 20.0
        elif h_sub < -0.34:
            sub_var += weights[HILLS] * (h_sub + 0.34) * 8.0
    if weights[ALPINE] > 0.25:
        a_sub = 1.0 - abs(value_noise(wx, wz, 22.0, WORLD_SEED + 140))
        knife_edge = value_noise(wx, wz, 14.0, WORLD_SEED + 144)
        sub_var += weights[ALPINE] * max(0.0, a_sub - 0.34) * 25.0
        if knife_edge > 0.35:
            sub_var += weights[ALPINE] * (knife_edge - 0.35) * 12.0
    if weights[KYIV_LOWLAND] > 0.25:
        k_sub = value_noise(wx, wz, 60.0, WORLD_SEED + 141)
        sub_var += weights[KYIV_LOWLAND] * k_sub * 0.35
    if weights[DESERT] > 0.40:
        d_sub = value_noise(wx, wz, 32.0, WORLD_SEED + 96)
        if d_sub > 0.45:
            sub_var += weights[DESERT] * (d_sub - 0.45) * 4.0
    if weights[SAVANNA] > 0.30:
        s_sub = value_noise(wx, wz, 38.0, WORLD_SEED + 102)
        if s_sub > 0.45:
            sub_var += weights[SAVANNA] * (s_sub - 0.45) * 2.2
    if weights[PLAINS] > 0.30:
        p_sub = value_noise(wx, wz, 30.0, WORLD_SEED + 99)
        if p_sub > 0.55:
            sub_var += weights[PLAINS] * (p_sub - 0.55) * 1.8
        elif p_sub < -0.62:
            sub_var += weights[PLAINS] * (p_sub + 0.62) * 1.4
    if weights[FOREST] > 0.30:
        f_sub = value_noise(wx, wz, 26.0, WORLD_SEED + 100)
        if f_sub < -0.55:
            sub_var += weights[FOREST] * (f_sub + 0.55) * 1.6
    sub_var *= away_from_water

    height = raw - river_flatten - river_dip - lake_flatten - lake_basin - lake_cut - coast_terrace - coast_cut - shore_smoothing - ocean_basin + desert_blown_sand + sub_var
    final_height = int(clamp(round(height), 3, WORLD_HEIGHT - 5))
    if river > 0.18:
        river_surface = RIVER_SURFACE_LEVEL if river > 0.48 or kyiv_weight > 0.18 else WATER_SURFACE_LEVEL
        river_depth = MIN_RIVER_DEPTH + int(round(clamp((river - 0.18) / 0.70, 0.0, 1.0) * 2.0 + kyiv_weight * 1.5))
        final_height = min(final_height, river_surface - river_depth)
    if lake > 0.30:
        lake_surface = WATER_SURFACE_LEVEL
        lake_depth = 2 + int(round(clamp((lake - 0.30) / 0.50, 0.0, 1.0) * 2.0 + kyiv_weight))
        final_height = min(final_height, lake_surface - lake_depth)
    if final_height < SEA_LEVEL:
        if river > 0.80:
            floor = RIVER_SURFACE_LEVEL - 6
        elif river > 0.55:
            floor = RIVER_SURFACE_LEVEL - 5
        elif river > 0.30:
            floor = WATER_SURFACE_LEVEL - 4
        elif river > 0.05 or lake > 0.30 or ocean_weight > 0.07:
            floor = SEA_LEVEL - 2
        else:
            floor = final_height
        final_height = max(final_height, floor)
    return final_height

@functools.lru_cache(maxsize=262_144)
def water_level_at(wx: int, wz: int) -> int:
    ground = terrain_height(wx, wz)
    river = effective_river_strength(wx, wz)
    lake = lake_strength(wx, wz)
    weights = biome_weights(wx, wz)
    if river > 0.18:
        river_surface = RIVER_SURFACE_LEVEL if river > 0.48 or weights[KYIV_LOWLAND] > 0.18 else WATER_SURFACE_LEVEL
        boosted_surface = river_surface + WATER_LEVEL_BOOST
        if ground < boosted_surface:
            return boosted_surface

    if lake > 0.26:
        boosted_surface = WATER_SURFACE_LEVEL + WATER_LEVEL_BOOST
        if ground < boosted_surface:
            return boosted_surface

    near_water = weights[OCEAN] > 0.10 or river > 0.34 or lake > 0.26 or weights[WETLAND] > 0.40
    boosted_surface = WATER_SURFACE_LEVEL + WATER_LEVEL_BOOST
    if ground < boosted_surface and (
        ground <= SEA_LEVEL - 1
        or weights[OCEAN] > 0.18
        or river > 0.40
        or lake > 0.34
        or near_water
    ):
        return boosted_surface

    return -1

@functools.lru_cache(maxsize=262_144)
def terrain_slope(wx: int, wz: int) -> int:
    center = terrain_height(wx, wz)
    return max(
        abs(center - terrain_height(wx + 1, wz)),
        abs(center - terrain_height(wx - 1, wz)),
        abs(center - terrain_height(wx, wz + 1)),
        abs(center - terrain_height(wx, wz - 1)),
    )

def surface_block_at(wx: int, wz: int, height: int | None = None) -> int:
    if height is None:
        height = terrain_height(wx, wz)

    weights = biome_weights(wx, wz)
    river = effective_river_strength(wx, wz)
    lake = lake_strength(wx, wz)
    patch = (hash_int(wx, wz, WORLD_SEED + 44) & 0xFFFF) / 0xFFFF
    shore_zone = weights[OCEAN] > 0.07 or river > 0.20 or lake > 0.26

    if height <= SEA_LEVEL + 1 or water_level_at(wx, wz) >= 0:
        return SAND

    if shore_zone and height <= SEA_LEVEL + 8:
        return SAND

    if weights[KYIV_LOWLAND] > 0.34:
        if river > 0.18 or lake > 0.32:
            return SAND
        if patch < weights[KYIV_LOWLAND] * 0.16:
            return DIRT
        return GRASS

    if weights[WETLAND] > 0.34:
        if lake > 0.22 or patch < weights[WETLAND] * 0.35:
            return DIRT
        return GRASS

    if weights[ALPINE] > 0.36:
        if height >= SEA_LEVEL + 36 or terrain_slope(wx, wz) <= 1:
            return SNOW
        return STONE

    if weights[TAIGA] > 0.38:
        if height >= SEA_LEVEL + 16 or patch < weights[TAIGA] * 0.45:
            return SNOW
        return GRASS

    if weights[BADLANDS] > 0.22:
        if terrain_slope(wx, wz) >= 2 or patch < 0.72:
            return RED_SAND
        return CLAY

    if weights[DESERT] > 0.50 or (weights[DESERT] > 0.24 and patch < weights[DESERT] * 0.85):
        sub = value_noise(wx, wz, 32.0, WORLD_SEED + 96)
        if sub > 0.55 and terrain_slope(wx, wz) >= 1:
            return STONE
        return SAND

    if weights[SAVANNA] > 0.24 or (weights[DESERT] > 0.14 and weights[OCEAN] < 0.12):
        sub = value_noise(wx, wz, 28.0, WORLD_SEED + 97)
        if sub > 0.62:
            return SAND
        return DRY_GRASS

    if weights[HILLS] > 0.40:
        sub = value_noise(wx, wz, 36.0, WORLD_SEED + 98)
        if sub > 0.50 and (terrain_slope(wx, wz) >= 2 or height >= SEA_LEVEL + 18):
            return STONE
        if sub < -0.55:
            return DIRT
        return GRASS

    if weights[PLAINS] > 0.30:
        sub = value_noise(wx, wz, 30.0, WORLD_SEED + 99)
        if sub < -0.62:
            return DIRT
        if sub > 0.68:
            return DRY_GRASS

    if weights[FOREST] > 0.30:
        sub = value_noise(wx, wz, 26.0, WORLD_SEED + 100)
        if sub < -0.55:
            return DIRT

    if river > 0.36 and patch < river * 0.65:
        return SAND
    return GRASS

def terrain_layer_block(wx: int, y: int, wz: int, height: int | None = None) -> int:
    if height is None:
        height = terrain_height(wx, wz)
    surface = surface_block_at(wx, wz, height)

    if y == 0:
        return BEDROCK
    if y == height:
        return surface
    if surface in (SAND, RED_SAND, SNOW) and y >= height - 4:
        return surface
    if surface in (DIRT, DRY_GRASS) and y >= height - 3:
        return DIRT
    if y < height - 5 or surface == STONE:
        return STONE
    return DIRT

@functools.lru_cache(maxsize=262_144)
def grass_plant_at(wx: int, wz: int) -> int:
    height = terrain_height(wx, wz)
    if height <= SEA_LEVEL:
        return AIR
    if water_level_at(wx, wz) >= 0:
        return AIR

    weights = biome_weights(wx, wz)
    surface = surface_block_at(wx, wz, height)
    slope = terrain_slope(wx, wz)
    if slope > 3:
        return AIR

    if surface in (SAND, RED_SAND):
        desert_scrub = weights[DESERT] * 0.06 + weights[BADLANDS] * 0.20
        if hash_int(wx, wz, WORLD_SEED + 93) / 0xFFFFFFFF < desert_scrub:
            return DEAD_BUSH
        return AIR

    if surface not in (GRASS, DRY_GRASS, DIRT, SNOW):
        return AIR
    if surface == SNOW and weights[TAIGA] + weights[ALPINE] < 0.35:
        return AIR

    clearing = forest_clearing_strength(wx, wz)
    density = 0.02
    tall_bias = 0.18

    if weights[SAVANNA] > 0.22:
        density += weights[SAVANNA] * 0.38
        tall_bias = 0.82
    if weights[KYIV_LOWLAND] > 0.22:
        density += weights[KYIV_LOWLAND] * 0.16
        tall_bias = max(tall_bias, 0.42)
    if weights[WETLAND] > 0.22:
        density += weights[WETLAND] * 0.28
        tall_bias = max(tall_bias, 0.60)
    if weights[PLAINS] > 0.22:
        density += weights[PLAINS] * 0.22
        tall_bias = max(tall_bias, 0.34)
    if weights[FOREST] > 0.22:
        density += weights[FOREST] * lerp(0.04, 0.30, clearing)
        tall_bias = max(tall_bias, 0.28)
    if weights[HILLS] > 0.22:
        density += weights[HILLS] * 0.18
        tall_bias = max(tall_bias, 0.30)
    if weights[TAIGA] > 0.25 or weights[ALPINE] > 0.25:
        density *= 0.35
    if weights[DESERT] > 0.32:
        density *= 0.12
    if weights[BADLANDS] > 0.28:
        density *= 0.10

    roll = hash_int(wx, wz, WORLD_SEED + 91) / 0xFFFFFFFF
    if roll > clamp(density, 0.0, 0.65):
        return AIR
    flower_roll = hash_int(wx, wz, WORLD_SEED + 94) / 0xFFFFFFFF
    flower_density = weights[PLAINS] * 0.12 + weights[KYIV_LOWLAND] * 0.10 + weights[FOREST] * 0.04 + weights[WETLAND] * 0.06
    if surface != SNOW and flower_roll < flower_density:
        return FLOWERS
    if surface == SNOW:
        return SHORT_GRASS
    tall_roll = hash_int(wx, wz, WORLD_SEED + 92) / 0xFFFFFFFF
    return TALL_GRASS if tall_roll < tall_bias else SHORT_GRASS

@functools.lru_cache(maxsize=262_144)
def forest_clearing_strength(wx: int, wz: int) -> float:
    broad = (value_noise(wx, wz, 520.0, WORLD_SEED + 81) + 1.0) * 0.5
    meadow = (value_noise(wx, wz, 210.0, WORLD_SEED + 82) + 1.0) * 0.5
    pocket = (value_noise(wx, wz, 96.0, WORLD_SEED + 86) + 1.0) * 0.5
    return smoothstep(clamp((broad * 0.62 + meadow * 0.30 + pocket * 0.08 - 0.38) / 0.28, 0.0, 1.0))

@functools.lru_cache(maxsize=262_144)
def tree_variant(wx: int, wz: int) -> str:
    biome = dominant_biome(wx, wz)
    seed = hash_int(wx, wz, WORLD_SEED + 83)
    if biome is OCEAN:
        return "none"
    if biome is DESERT:
        return "cactus"
    if biome is BADLANDS:
        return "dead_tree" if seed % 7 == 0 else "none"
    if biome is ALPINE:
        return "alpine_pine" if seed % 6 == 0 else "none"
    if biome is TAIGA:
        return "taiga_pine"
    if biome is KYIV_LOWLAND:
        return "kyiv_poplar" if seed % 3 == 0 else ("birch" if seed % 3 == 1 else "field_oak")
    if biome is WETLAND:
        return "wetland_willow" if seed % 3 else "birch"
    if biome is SAVANNA:
        return "acacia"
    if biome is FOREST:
        roll = seed % 100
        if roll < 18:
            return "forest_pine"
        if roll < 43:
            return "birch"
        return "forest_oak"
    if biome is HILLS:
        return "alpine_pine" if seed % 4 == 0 else "forest_pine"
    if biome is PLAINS:
        return "field_oak"
    return "field_oak"

@functools.lru_cache(maxsize=262_144)
def tree_height(wx: int, wz: int) -> int:
    seed = hash_int(wx, wz, WORLD_SEED + 11)
    variant = tree_variant(wx, wz)
    if variant == "forest_pine":
        return 6 + seed % 4
    if variant == "taiga_pine":
        return 9 + seed % 7
    if variant == "alpine_pine":
        return 5 + seed % 4
    if variant == "cactus":
        return 5 + seed % 3
    if variant == "birch":
        return 5 + seed % 4
    if variant == "kyiv_poplar":
        return 8 + seed % 5
    if variant == "wetland_willow":
        return 5 + seed % 3
    if variant == "dead_tree":
        return 4 + seed % 4
    if variant == "acacia":
        return 4 + seed % 3
    if variant == "field_oak":
        return 3 + seed % 3
    if variant == "forest_oak":
        return 6 + seed % 5
    return 4 + seed % 4

@functools.lru_cache(maxsize=262_144)
def tree_should_spawn(wx: int, wz: int) -> bool:
    dominant = dominant_biome(wx, wz)
    if dominant is OCEAN:
        return False
    if tree_variant(wx, wz) == "none":
        return False
    weights = biome_weights(wx, wz)
    density = sum(weight * biome.tree_density for biome, weight in weights.items())
    if dominant is DESERT:
        density = 0.012
    if dominant is BADLANDS:
        density = 0.0035
    if dominant is KYIV_LOWLAND:
        density *= 0.45
    if dominant is WETLAND:
        density *= 0.70
    if dominant is ALPINE:
        density *= 0.35
    forest_weight = weights[FOREST]
    clearing = forest_clearing_strength(wx, wz)
    if forest_weight > 0.30 and clearing > 0.85:
        return False
    if forest_weight > 0.22:
        density *= lerp(0.95, 0.20, clearing)
    if hash_int(wx, wz, WORLD_SEED + 9) / 0xFFFFFFFF > density:
        return False

    height = terrain_height(wx, wz)
    if height <= SEA_LEVEL + 2:
        return False
    if water_level_at(wx, wz) >= 0:
        return False
    surface = surface_block_at(wx, wz, height)
    if dominant is DESERT:
        return surface == SAND and terrain_slope(wx, wz) <= 2
    if dominant is BADLANDS:
        return surface in (RED_SAND, CLAY, SAND) and terrain_slope(wx, wz) <= 2
    if dominant in (TAIGA, ALPINE):
        return surface in (GRASS, SNOW) and terrain_slope(wx, wz) <= 3
    if dominant is WETLAND:
        return surface in (GRASS, DIRT) and terrain_slope(wx, wz) <= 1
    if surface != GRASS:
        return False

    neighbor_heights = (
        terrain_height(wx + 1, wz),
        terrain_height(wx - 1, wz),
        terrain_height(wx, wz + 1),
        terrain_height(wx, wz - 1),
    )
    return max(abs(h - height) for h in neighbor_heights) <= 3

@functools.lru_cache(maxsize=1_048_576)
def tree_block_from_origin(tx: int, tz: int, wx: int, y: int, wz: int) -> int:
    base = terrain_height(tx, tz) + 1
    trunk_height = tree_height(tx, tz)
    top = base + trunk_height
    variant = tree_variant(tx, tz)
    dx = wx - tx
    dz = wz - tz
    adx = abs(dx)
    adz = abs(dz)
    seed = hash_int(tx, tz, WORLD_SEED + 85)

    if variant == "cactus":
        if wx == tx and wz == tz and base <= y < top:
            return CACTUS
        arm1_y = base + max(1, trunk_height - 4)
        arm1_dx = 1 if seed & 1 else -1
        if wx == tx + arm1_dx and wz == tz and arm1_y <= y <= arm1_y + 2:
            return CACTUS
        if seed & 2:
            arm2_y = base + max(2, trunk_height - 2)
            arm2_dz = 1 if seed & 4 else -1
            if wx == tx and wz == tz + arm2_dz and arm2_y <= y <= arm2_y + 2:
                return CACTUS
        return AIR

    if variant == "dead_tree":
        bend_dx = 1 if seed & 1 else -1
        if wx == tx and wz == tz and base <= y < top - 1:
            return ACACIA_LOG
        if y == top - 2 and wx == tx + bend_dx and wz == tz:
            return ACACIA_LOG
        if y == top - 1 and wx == tx + bend_dx * 2 and wz == tz:
            return ACACIA_LOG
        return AIR

    if variant in ("forest_pine", "taiga_pine", "alpine_pine"):
        if wx == tx and wz == tz and base <= y < top:
            return LOG
        crown_top = top + (3 if variant == "taiga_pine" else 2)
        crown_base = base + (2 if variant != "alpine_pine" else 1)
        if crown_base <= y <= crown_top:
            level_from_top = crown_top - y
            max_radius = 4 if variant == "taiga_pine" else (2 if variant == "alpine_pine" else 3)
            radius = clamp(1 + level_from_top // 2, 0, max_radius)
            radius = min(radius + 1, max_radius)
            if variant == "taiga_pine" and y < top - 5:
                radius = max_radius
            if variant == "alpine_pine" and y < top - 2:
                radius = min(radius, 2)
            if max(adx, adz) <= radius and adx + adz <= radius + 2:
                if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 11 != 0:
                    return PINE_LEAVES
        return AIR

    if variant == "acacia":
        branch_dx = 1 if seed & 1 else -1
        branch_dz = 1 if seed & 2 else -1
        if wx == tx and wz == tz and base <= y < top - 1:
            return ACACIA_LOG
        if y == top - 2 and ((wx == tx + branch_dx and wz == tz) or (wx == tx and wz == tz + branch_dz)):
            return ACACIA_LOG
        crown_x = tx + branch_dx
        crown_z = tz + branch_dz
        cdx = abs(wx - crown_x)
        cdz = abs(wz - crown_z)
        if y == top - 1 and max(cdx, cdz) <= 3 and cdx + cdz <= 4:
            if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 12 != 0:
                return ACACIA_LEAVES
        if y == top and max(cdx, cdz) <= 2 and cdx + cdz <= 3:
            if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 10 != 0:
                return ACACIA_LEAVES
        if y == top + 1 and max(cdx, cdz) <= 1:
            if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 6 != 0:
                return ACACIA_LEAVES
        return AIR

    if variant == "kyiv_poplar":
        if wx == tx and wz == tz and base <= y < top:
            return BIRCH_LOG
        crown_center = top - 1
        if crown_center - 3 <= y <= crown_center + 4:
            layer = abs(y - crown_center)
            radius = 1 if layer >= 3 else 2
            if max(adx, adz) <= radius and adx + adz <= radius + 1:
                if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 9 != 0:
                    return BIRCH_LEAVES
        return AIR

    if variant == "wetland_willow":
        if wx == tx and wz == tz and base <= y < top - 1:
            return LOG
        crown_center = top - 1
        if crown_center - 2 <= y <= crown_center + 2:
            layer = y - crown_center
            radius = 3 if layer <= 0 else 2
            if max(adx, adz) <= radius and adx + adz <= radius + 2:
                if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 10 != 0:
                    return LEAVES
        if crown_center - 4 <= y < crown_center - 1 and max(adx, adz) in (2, 3) and adx + adz <= 4:
            if hash_int(wx + y * 17, wz, WORLD_SEED + 15) % 4 != 0:
                return LEAVES
        return AIR

    if wx == tx and wz == tz and base <= y < top:
        return BIRCH_LOG if variant == "birch" else LOG

    if variant == "forest_oak":
        crown_center = top
        if crown_center - 3 <= y <= crown_center + 3:
            layer = y - crown_center
            if abs(layer) == 3:
                radius = 1
            elif abs(layer) == 2:
                radius = 2
            else:
                radius = 3
            if max(adx, adz) <= radius and adx + adz <= radius + 1:
                if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 11 != 0:
                    return LEAVES
        return AIR

    if variant == "birch":
        crown_center = top
        if crown_center - 2 <= y <= crown_center + 3:
            layer = y - crown_center
            if abs(layer) >= 3:
                radius = 1
            elif abs(layer) == 2:
                radius = 1
            else:
                radius = 2
            if max(adx, adz) <= radius and adx + adz <= radius + 1:
                if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 8 != 0:
                    return BIRCH_LEAVES
        return AIR

    crown_center = top
    if crown_center - 2 <= y <= crown_center + 2:
        layer = y - crown_center
        if abs(layer) == 2:
            radius = 1
        elif abs(layer) == 1:
            radius = 2
        else:
            radius = 2
        if max(adx, adz) <= radius and adx + adz <= radius + 1:
            if hash_int(wx + y * 31, wz, WORLD_SEED + 12) % 8 != 0:
                return LEAVES
    return AIR

@functools.lru_cache(maxsize=1_048_576)
def tree_block_at(wx: int, y: int, wz: int) -> int:
    for tx in range(wx - 4, wx + 5):
        for tz in range(wz - 4, wz + 5):
            if tree_should_spawn(tx, tz):
                block = tree_block_from_origin(tx, tz, wx, y, wz)
                if block != AIR:
                    return block
    return AIR

@functools.lru_cache(maxsize=1_048_576)
def procedural_block_at(wx: int, y: int, wz: int) -> int:
    if y < 0 or y >= WORLD_HEIGHT:
        return AIR

    height = terrain_height(wx, wz)
    if y <= height:
        return terrain_layer_block(wx, y, wz, height)
    if y <= water_level_at(wx, wz):
        return WATER

    tree_block = tree_block_at(wx, y, wz)
    if tree_block != AIR:
        return tree_block
    if y == height + 1:
        return grass_plant_at(wx, wz)
    return AIR

@functools.lru_cache(maxsize=1_048_576)
def procedural_collision_solid(wx: int, y: int, wz: int) -> bool:
    if y < 0 or y >= WORLD_HEIGHT:
        return False
    if y <= terrain_height(wx, wz):
        return True
    return False

@functools.lru_cache(maxsize=262_144)
def procedural_highest_solid_y(wx: int, wz: int) -> int:
    best = max(terrain_height(wx, wz), water_level_at(wx, wz))

    for tx in range(wx - 4, wx + 5):
        for tz in range(wz - 4, wz + 5):
            if not tree_should_spawn(tx, tz):
                continue

            base = terrain_height(tx, tz) + 1
            trunk_height = tree_height(tx, tz)
            for y in range(min(WORLD_HEIGHT - 1, base + trunk_height + 4), base - 1, -1):
                if tree_block_from_origin(tx, tz, wx, y, wz) != AIR:
                    best = max(best, y)
                    break

    return min(best, WORLD_HEIGHT - 1)

def pseudo_noise(x: int, y: int, tile: int) -> int:
    n = x * 374761393 + y * 668265263 + tile * 2246822519
    n = (n ^ (n >> 13)) * 1274126177
    return (n ^ (n >> 16)) & 255

def vary(color: tuple[int, int, int], amount: int) -> tuple[int, int, int, int]:
    r, g, b = color
    return (
        int(clamp(r + amount, 0, 255)),
        int(clamp(g + amount, 0, 255)),
        int(clamp(b + amount, 0, 255)),
        255,
    )

def atlas_color(tile: int, x: int, y: int) -> tuple[int, int, int, int]:
    noise = pseudo_noise(x, y, tile)
    wobble = (noise % 23) - 11
    blocky = (pseudo_noise(x // 4, y // 4, tile + 31) % 19) - 9

    if tile == TEX_GRASS_TOP:
        cluster = pseudo_noise(x // 2, y // 2, tile + 17)
        if cluster > 210:
            return vary((100, 173, 64), wobble // 3 + blocky)
        if cluster > 150:
            return vary((77, 154, 56), wobble // 3 + blocky)
        if cluster > 80:
            return vary((61, 132, 48), wobble // 3 + blocky)
        return vary((47, 111, 43), wobble // 3 + blocky)
    if tile == TEX_GRASS_SIDE:
        if y >= TILE_SIZE - 3 + (noise % 4):
            tip = pseudo_noise(x, y, tile + 5)
            if tip > 170:
                return vary((92, 166, 62), wobble // 2 + blocky)
            return vary((66, 143, 50), wobble // 2 + blocky)
        if y >= TILE_SIZE - 5 and noise % 4 == 0:
            return vary((70, 132, 50), wobble // 2 + blocky)
        if noise % 19 == 0:
            return vary((82, 57, 34), wobble // 2 + blocky)
        return vary((134, 96, 55), wobble // 2 + blocky)
    if tile == TEX_DIRT:
        pebble = 26 if noise > 226 else (-18 if noise < 24 else 0)
        clump = -12 if (x * 3 + y * 5) % 17 in (0, 1) else 0
        return vary((126, 86, 49), wobble // 2 + blocky + pebble + clump)
    if tile == TEX_STONE:
        crack = -36 if (x * 7 + y * 11 + noise * 3) % 43 in (0, 1) else 0
        speckle = 20 if noise > 220 else (-16 if noise < 30 else 0)
        return vary((117, 117, 117), wobble // 2 + blocky + crack + speckle)
    if tile == TEX_SAND:
        ripple = 12 if (x + y * 2) % 7 == 0 else (-8 if (x * 2 + y) % 9 == 0 else 0)
        grain = 18 if noise > 232 else 0
        return vary((219, 207, 139), wobble // 3 + blocky + ripple + grain)
    if tile == TEX_LOG_SIDE:
        stripe = 22 if x % 5 in (0, 1) else -10
        knot = -32 if (x in (5, 6, 7) and y in (5, 6, 7)) or (x in (10, 11) and y in (10, 11)) else 0
        grain = -6 if y % 4 == 0 else 0
        return vary((103, 63, 31), wobble // 2 + stripe + knot + grain)
    if tile == TEX_LOG_TOP:
        dx = x - TILE_SIZE / 2
        dy = y - TILE_SIZE / 2
        ring = int(math.hypot(dx, dy) * 2) % 5
        return vary((159, 118, 69), wobble // 2 + (18 if ring == 0 else -7))
    if tile == TEX_LEAVES:
        cluster = (pseudo_noise(x // 3, y // 3, tile + 11) + noise) // 2
        if cluster > 180:
            return vary((82, 145, 67), wobble // 2 + blocky + 8)
        if cluster > 120:
            return vary((59, 125, 55), wobble // 2 + blocky)
        if cluster > 60:
            return vary((42, 103, 42), wobble // 2 + blocky)
        return vary((29, 82, 34), wobble // 2 + blocky - 4)
    if tile == TEX_BEDROCK:
        base = (40, 41, 43) if noise > 120 else (77, 76, 78)
        return vary(base, wobble // 2)
    if tile == TEX_WATER:
        ripple = 14 if (x + y + noise) % 11 in (0, 1) else 0
        sparkle = 22 if noise > 240 else 0
        depth = -8 if (x + y) % 5 == 0 else 0
        return vary((44, 106, 184), wobble // 4 + ripple + sparkle + depth)
    if tile == TEX_SNOW:
        icy = 8 if noise % 13 in (0, 1) else 0
        return vary((224, 234, 238), wobble // 4 + icy)
    if tile == TEX_RED_SAND:
        band = 18 if y % 7 in (0, 1) else -7
        return vary((184, 92, 48), wobble // 3 + blocky + band)
    if tile == TEX_PLANKS:
        line = -34 if y % 5 == 0 else (16 if x % 8 == 0 else 0)
        return vary((157, 111, 62), wobble // 3 + blocky + line)
    if tile == TEX_DARK_PLANKS:
        line = -30 if y % 5 == 0 else (12 if x % 8 == 0 else 0)
        return vary((82, 53, 31), wobble // 3 + blocky + line)
    if tile == TEX_COBBLESTONE:
        mortar = -38 if x % 6 == 0 or y % 5 == 0 else 0
        highlight = 18 if noise > 226 else 0
        return vary((103, 105, 103), wobble // 3 + blocky + mortar + highlight)
    if tile == TEX_BRICKS:
        mortar = -48 if y % 5 == 0 or (x + (y // 5 % 2) * 4) % 8 == 0 else 0
        return vary((150, 73, 49), wobble // 4 + blocky + mortar)
    if tile == TEX_GLASS:
        streak = 28 if x == y or x + y == TILE_SIZE - 1 else 0
        return vary((126, 184, 205), wobble // 5 + streak)
    if tile == TEX_CLAY:
        band = 12 if y % 6 in (0, 1) else 0
        return vary((161, 126, 103), wobble // 3 + blocky + band)
    if tile == TEX_THATCH:
        straw = 18 if (x + noise) % 5 == 0 else -8 if y % 4 == 0 else 0
        return vary((190, 157, 73), wobble // 3 + straw)
    if tile == TEX_CACTUS:
        col = x % 5
        rib = 24 if col == 0 else (-14 if col == 2 else (-6 if col == 3 else 0))
        thorn = 38 if noise % 29 == 0 else 0
        shadow = -10 if col == 4 else 0
        rim = 16 if (y == 0 or y == TILE_SIZE - 1) and col == 0 else 0
        return vary((38, 124, 66), wobble // 3 + rib + thorn + shadow + rim)
    if tile == TEX_DRY_GRASS_TOP:
        base = (126, 139, 66) if noise > 96 else (104, 126, 61)
        return vary(base, wobble // 2)
    if tile == TEX_DRY_GRASS_SIDE:
        if y >= TILE_SIZE - 4 + (noise % 3):
            return vary((116, 132, 62), wobble // 2)
        return vary((128, 91, 48), wobble // 2)
    if tile == TEX_SHORT_GRASS:
        if y < 3:
            return 0, 0, 0, 0
        blade_hash = hash_int(x, 0, WORLD_SEED + 200)
        if (blade_hash & 0xF) < 4:
            return 0, 0, 0, 0
        blade_top = 4 + ((blade_hash >> 4) & 0x7)
        if y > blade_top + 8:
            return 0, 0, 0, 0
        t = (y - 3) / max(1, TILE_SIZE - 4)
        palette = ((58, 122, 48), (72, 138, 54), (84, 130, 50))
        base = palette[(blade_hash >> 7) % 3]
        tip = (138, 184, 92)
        r = int(base[0] * (1.0 - t) + tip[0] * t)
        g = int(base[1] * (1.0 - t) + tip[1] * t)
        b = int(base[2] * (1.0 - t) + tip[2] * t)
        jitter = ((blade_hash >> 11) % 9) - 4
        return (max(0, min(255, r + jitter)),
                max(0, min(255, g + jitter)),
                max(0, min(255, b + jitter)),
                255)
    if tile == TEX_TALL_GRASS:
        if y < 1:
            return 0, 0, 0, 0
        blade_hash = hash_int(x, 0, WORLD_SEED + 201)
        if (blade_hash & 0xF) < 3:
            return 0, 0, 0, 0
        t = (y - 1) / max(1, TILE_SIZE - 2)
        palette = ((68, 116, 52), (96, 134, 58), (112, 142, 64))
        base = palette[(blade_hash >> 7) % 3]
        tip = (172, 196, 110)
        r = int(base[0] * (1.0 - t) + tip[0] * t)
        g = int(base[1] * (1.0 - t) + tip[1] * t)
        b = int(base[2] * (1.0 - t) + tip[2] * t)
        if y >= TILE_SIZE - 3 and ((blade_hash >> 14) & 0x3) == 0:
            seed_head = (192, 178, 84)
            mix = (y - (TILE_SIZE - 3)) / 2.0
            r = int(r * (1.0 - mix) + seed_head[0] * mix)
            g = int(g * (1.0 - mix) + seed_head[1] * mix)
            b = int(b * (1.0 - mix) + seed_head[2] * mix)
        jitter = ((blade_hash >> 11) % 9) - 4
        return (max(0, min(255, r + jitter)),
                max(0, min(255, g + jitter)),
                max(0, min(255, b + jitter)),
                255)
    if tile == TEX_PINE_LEAVES:
        cluster = pseudo_noise(x // 2, y // 2, tile + 17)
        frost = 18 if noise > 224 else 0
        if cluster > 170:
            return vary((42, 104, 78), wobble // 2 + blocky + frost)
        if cluster > 90:
            return vary((32, 82, 62), wobble // 2 + blocky + frost)
        return vary((24, 62, 48), wobble // 2 + blocky)
    if tile == TEX_BIRCH_LOG_SIDE:
        bark_line = -48 if x % 5 == 0 or (noise > 226 and y % 4 in (0, 1)) else 0
        scar = -30 if (x + y * 3 + noise) % 31 in (0, 1, 2) else 0
        return vary((213, 207, 184), wobble // 4 + bark_line + scar)
    if tile == TEX_BIRCH_LOG_TOP:
        dx = x - TILE_SIZE / 2
        dy = y - TILE_SIZE / 2
        ring = int(math.hypot(dx, dy) * 2.3) % 5
        return vary((190, 164, 102), wobble // 3 + (15 if ring == 0 else -8))
    if tile == TEX_BIRCH_LEAVES:
        cluster = (pseudo_noise(x // 3, y // 3, tile + 11) + noise) // 2
        if cluster > 170:
            return vary((118, 168, 77), wobble // 2 + blocky)
        if cluster > 90:
            return vary((86, 143, 61), wobble // 2 + blocky)
        return vary((58, 116, 52), wobble // 2 + blocky)
    if tile == TEX_ACACIA_LOG_SIDE:
        stripe = 18 if x % 4 in (0, 1) else -9
        return vary((128, 75, 39), wobble // 3 + stripe + blocky)
    if tile == TEX_ACACIA_LOG_TOP:
        dx = x - TILE_SIZE / 2
        dy = y - TILE_SIZE / 2
        ring = int(math.hypot(dx, dy) * 2.1) % 4
        return vary((166, 103, 55), wobble // 3 + (16 if ring == 0 else -6))
    if tile == TEX_ACACIA_LEAVES:
        cluster = pseudo_noise(x // 3, y // 2, tile + 15)
        if cluster > 170:
            return vary((112, 137, 56), wobble // 2 + blocky)
        if cluster > 82:
            return vary((82, 113, 47), wobble // 2 + blocky)
        return vary((58, 90, 38), wobble // 2 + blocky)
    if tile == TEX_DEAD_BUSH:
        stem = abs(x - 7) <= 1 or abs((x + y // 2) - 8) <= 1 or abs((15 - x + y // 2) - 8) <= 1
        if not stem or y < 2:
            return 0, 0, 0, 0
        return vary((126, 92, 48), wobble // 3 + blocky)
    if tile == TEX_FLOWERS:
        stem = abs(x - 8) <= 1 and y < 12
        blossom = (x - 8) * (x - 8) + (y - 12) * (y - 12) <= 10
        if blossom:
            color = (224, 212, 78) if noise > 128 else (218, 112, 124)
            return vary(color, wobble // 4)
        if stem:
            return vary((48, 126, 52), wobble // 3)
        return 0, 0, 0, 0
    return 255, 0, 255, 255

def create_texture_atlas() -> pyglet.image.Texture:
    pixels = bytearray()
    for y in range(ATLAS_HEIGHT):
        for x in range(ATLAS_WIDTH):
            tile_x = x // TILE_SIZE
            tile_y = y // TILE_SIZE
            tile = tile_y * ATLAS_COLUMNS + tile_x
            local_x = x % TILE_SIZE
            local_y = y % TILE_SIZE
            pixels.extend(atlas_color(tile, local_x, local_y))

    image = pyglet.image.ImageData(ATLAS_WIDTH, ATLAS_HEIGHT, "RGBA", bytes(pixels))
    texture = image.get_texture()
    gl.glBindTexture(texture.target, texture.id)
    gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    return texture

def tile_uv(tile: int) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    col = tile % ATLAS_COLUMNS
    row = tile // ATLAS_COLUMNS
    padding = 0.35
    u0 = (col * TILE_SIZE + padding) / ATLAS_WIDTH
    v0 = (row * TILE_SIZE + padding) / ATLAS_HEIGHT
    u1 = ((col + 1) * TILE_SIZE - padding) / ATLAS_WIDTH
    v1 = ((row + 1) * TILE_SIZE - padding) / ATLAS_HEIGHT
    return (u0, v0), (u1, v0), (u1, v1), (u0, v1)

def texture_for_face(block: int, face_name: str) -> int:
    if block == GRASS:
        if face_name == "top":
            return TEX_GRASS_TOP
        if face_name == "bottom":
            return TEX_DIRT
        return TEX_GRASS_SIDE
    if block == DRY_GRASS:
        if face_name == "top":
            return TEX_DRY_GRASS_TOP
        if face_name == "bottom":
            return TEX_DIRT
        return TEX_DRY_GRASS_SIDE
    if block == DIRT:
        return TEX_DIRT
    if block == STONE:
        return TEX_STONE
    if block == SAND:
        return TEX_SAND
    if block == LOG:
        return TEX_LOG_TOP if face_name in ("top", "bottom") else TEX_LOG_SIDE
    if block == LEAVES:
        return TEX_LEAVES
    if block == PINE_LEAVES:
        return TEX_PINE_LEAVES
    if block == BIRCH_LOG:
        return TEX_BIRCH_LOG_TOP if face_name in ("top", "bottom") else TEX_BIRCH_LOG_SIDE
    if block == BIRCH_LEAVES:
        return TEX_BIRCH_LEAVES
    if block == ACACIA_LOG:
        return TEX_ACACIA_LOG_TOP if face_name in ("top", "bottom") else TEX_ACACIA_LOG_SIDE
    if block == ACACIA_LEAVES:
        return TEX_ACACIA_LEAVES
    if block == BEDROCK:
        return TEX_BEDROCK
    if block == WATER:
        return TEX_WATER
    if block == SNOW:
        return TEX_SNOW
    if block == RED_SAND:
        return TEX_RED_SAND
    if block == PLANKS:
        return TEX_PLANKS
    if block == DARK_PLANKS:
        return TEX_DARK_PLANKS
    if block == COBBLESTONE:
        return TEX_COBBLESTONE
    if block == BRICKS:
        return TEX_BRICKS
    if block == GLASS:
        return TEX_GLASS
    if block == CLAY:
        return TEX_CLAY
    if block == THATCH:
        return TEX_THATCH
    if block == CACTUS:
        return TEX_CACTUS
    if block == SHORT_GRASS:
        return TEX_SHORT_GRASS
    if block == TALL_GRASS:
        return TEX_TALL_GRASS
    if block == DEAD_BUSH:
        return TEX_DEAD_BUSH
    if block == FLOWERS:
        return TEX_FLOWERS
    return TEX_STONE

def is_plant_block(block: int) -> bool:
    return block in (SHORT_GRASS, TALL_GRASS, DEAD_BUSH, FLOWERS)

def is_occluding_block(block: int) -> bool:
    return block not in (AIR, SHORT_GRASS, TALL_GRASS, DEAD_BUSH, FLOWERS)

def is_collidable_block(block: int) -> bool:
    return block not in (AIR, WATER, SHORT_GRASS, TALL_GRASS, DEAD_BUSH, FLOWERS)

class VoxelGroup(Group):
    def __init__(self, texture: pyglet.image.Texture, program: pyglet.graphics.shader.ShaderProgram):
        super().__init__(order=0)
        self.texture = texture
        self.program = program
        self.model = Mat4()

    def set_state(self) -> None:
        self.program.use()
        self.program["model"] = self.model
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(self.texture.target, self.texture.id)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

    def unset_state(self) -> None:
        gl.glDisable(gl.GL_CULL_FACE)
        self.program.stop()

    def __hash__(self) -> int:
        return hash((self.texture.target, self.texture.id, self.program, self.order, self.parent))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VoxelGroup)
            and self.texture.target == other.texture.target
            and self.texture.id == other.texture.id
            and self.program == other.program
            and self.order == other.order
            and self.parent == other.parent
        )

def world_to_chunk(wx: int, wz: int) -> tuple[int, int, int, int]:
    cx = wx // CHUNK_SIZE
    cz = wz // CHUNK_SIZE
    return cx, cz, wx - cx * CHUNK_SIZE, wz - cz * CHUNK_SIZE

def add_quad(
    vertices: list[float],
    normals: list[float],
    texcoords: list[float],
    indices: list[int],
    corners: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    normal: tuple[float, float, float],
    tile: int,
) -> None:
    base = len(vertices) // 3
    for corner in corners:
        vertices.extend(corner)
        normals.extend(normal)
    for u, v in tile_uv(tile):
        texcoords.extend((u, v))
    indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

def add_double_sided_quad(
    vertices: list[float],
    normals: list[float],
    texcoords: list[float],
    indices: list[int],
    corners: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    normal: tuple[float, float, float],
    tile: int,
) -> None:
    base = len(vertices) // 3
    for corner in corners:
        vertices.extend(corner)
        normals.extend(normal)
    for u, v in tile_uv(tile):
        texcoords.extend((u, v))
    indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    indices.extend((base + 2, base + 1, base, base + 3, base + 2, base))

def add_plant_mesh(
    vertices: list[float],
    normals: list[float],
    texcoords: list[float],
    indices: list[int],
    wx: int,
    y: int,
    wz: int,
    block: int,
) -> int:
    tile = texture_for_face(block, "plant")
    if block == SHORT_GRASS:
        height = 0.72
    elif block == DEAD_BUSH:
        height = 0.82
    elif block == FLOWERS:
        height = 0.64
    else:
        height = 1.18
    inset = 0.08
    corners_a = (
        (wx + inset, y, wz + inset),
        (wx + 1.0 - inset, y, wz + 1.0 - inset),
        (wx + 1.0 - inset, y + height, wz + 1.0 - inset),
        (wx + inset, y + height, wz + inset),
    )
    corners_b = (
        (wx + 1.0 - inset, y, wz + inset),
        (wx + inset, y, wz + 1.0 - inset),
        (wx + inset, y + height, wz + 1.0 - inset),
        (wx + 1.0 - inset, y + height, wz + inset),
    )
    add_double_sided_quad(vertices, normals, texcoords, indices, corners_a, (0.0, 0.7, 0.7), tile)
    add_double_sided_quad(vertices, normals, texcoords, indices, corners_b, (0.7, 0.7, 0.0), tile)
    return 4

@dataclass(frozen=True)
class MeshData:
    cx: int
    cz: int
    version: int
    vertices: list[float]
    normals: list[float]
    texcoords: list[float]
    indices: list[int]
    faces: int

class Chunk:
    def __init__(self, cx: int, cz: int, edits: dict[tuple[int, int, int], int]) -> None:
        self.cx = cx
        self.cz = cz
        self.world_x = cx * CHUNK_SIZE
        self.world_z = cz * CHUNK_SIZE
        self.blocks = bytearray(CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE)
        self.vertex_list = None
        self.visible_faces = 0
        self.rendered_vertices = 0
        self.block_count = 0
        self.max_y = 0
        self.dirty = True
        self.version = 0
        self.mesh_pending = False
        self.generate(edits)
        self.version = 0

    @staticmethod
    def index(x: int, y: int, z: int) -> int:
        return (y * CHUNK_SIZE + z) * CHUNK_SIZE + x

    @staticmethod
    def in_local_bounds(x: int, y: int, z: int) -> bool:
        return 0 <= x < CHUNK_SIZE and 0 <= y < WORLD_HEIGHT and 0 <= z < CHUNK_SIZE

    def contains_world_xz(self, wx: int, wz: int) -> bool:
        return self.world_x <= wx < self.world_x + CHUNK_SIZE and self.world_z <= wz < self.world_z + CHUNK_SIZE

    def local_xz(self, wx: int, wz: int) -> tuple[int, int]:
        return wx - self.world_x, wz - self.world_z

    def get_local(self, x: int, y: int, z: int) -> int:
        if not self.in_local_bounds(x, y, z):
            return AIR
        return self.blocks[self.index(x, y, z)]

    def set_local(self, x: int, y: int, z: int, block: int) -> bool:
        if not self.in_local_bounds(x, y, z):
            return False
        idx = self.index(x, y, z)
        old = self.blocks[idx]
        if old == block:
            return False
        self.blocks[idx] = block
        if old == AIR and block != AIR:
            self.block_count += 1
            self.max_y = max(self.max_y, y)
        elif old != AIR and block == AIR:
            self.block_count -= 1
            if y == self.max_y:
                self.recompute_max_y()
        elif block != AIR:
            self.max_y = max(self.max_y, y)
        self.dirty = True
        self.version += 1
        return True

    def place_generated(self, wx: int, y: int, wz: int, block: int) -> None:
        if not self.contains_world_xz(wx, wz) or y < 0 or y >= WORLD_HEIGHT:
            return
        lx, lz = self.local_xz(wx, wz)
        if self.get_local(lx, y, lz) != AIR:
            return
        self.blocks[self.index(lx, y, lz)] = block
        self.block_count += 1
        self.max_y = max(self.max_y, y)

    def set_generated(self, wx: int, y: int, wz: int, block: int) -> None:
        if not self.contains_world_xz(wx, wz) or y < 0 or y >= WORLD_HEIGHT:
            return
        lx, lz = self.local_xz(wx, wz)
        idx = self.index(lx, y, lz)
        old = self.blocks[idx]
        if old == block:
            return
        self.blocks[idx] = block
        if old == AIR and block != AIR:
            self.block_count += 1
            self.max_y = max(self.max_y, y)
        elif old != AIR and block == AIR:
            self.block_count -= 1

    def clear_generated_column(self, wx: int, wz: int, y0: int, y1: int) -> None:
        for y in range(max(0, y0), min(WORLD_HEIGHT, y1 + 1)):
            self.set_generated(wx, y, wz, AIR)

    def generate(self, edits: dict[tuple[int, int, int], int]) -> None:
        self.block_count = 0
        self.max_y = 0
        for lx in range(CHUNK_SIZE):
            wx = self.world_x + lx
            for lz in range(CHUNK_SIZE):
                wz = self.world_z + lz
                height = terrain_height(wx, wz)
                for y in range(height + 1):
                    block = terrain_layer_block(wx, y, wz, height)
                    self.blocks[self.index(lx, y, lz)] = block
                    self.block_count += 1
                    self.max_y = max(self.max_y, y)
                water_level = water_level_at(wx, wz)
                for y in range(height + 1, water_level + 1):
                    self.blocks[self.index(lx, y, lz)] = WATER
                    self.block_count += 1
                    self.max_y = max(self.max_y, y)

        tree_x0 = self.world_x - TREE_MARGIN
        tree_x1 = self.world_x + CHUNK_SIZE + TREE_MARGIN
        tree_z0 = self.world_z - TREE_MARGIN
        tree_z1 = self.world_z + CHUNK_SIZE + TREE_MARGIN
        for tx in range(tree_x0, tree_x1):
            for tz in range(tree_z0, tree_z1):
                if tree_should_spawn(tx, tz):
                    self.add_tree_from_origin(tx, tz)

        for lx in range(CHUNK_SIZE):
            wx = self.world_x + lx
            for lz in range(CHUNK_SIZE):
                wz = self.world_z + lz
                plant = grass_plant_at(wx, wz)
                if plant != AIR:
                    self.place_generated(wx, terrain_height(wx, wz) + 1, wz, plant)

        for (wx, y, wz), block in edits.items():
            if self.contains_world_xz(wx, wz):
                lx, lz = self.local_xz(wx, wz)
                self.set_local(lx, y, lz, block)

        self.recompute_max_y()
        self.dirty = True

    def recompute_max_y(self) -> None:
        for y in range(WORLD_HEIGHT - 1, -1, -1):
            layer_offset = y * CHUNK_SIZE * CHUNK_SIZE
            if any(self.blocks[layer_offset:layer_offset + CHUNK_SIZE * CHUNK_SIZE]):
                self.max_y = y
                return
        self.max_y = 0

    def add_tree_from_origin(self, wx: int, wz: int) -> None:
        base = terrain_height(wx, wz) + 1
        trunk_height = tree_height(wx, wz)
        for lx in range(wx - 4, wx + 5):
            for lz in range(wz - 4, wz + 5):
                for ly in range(base, min(WORLD_HEIGHT, base + trunk_height + 5)):
                    block = tree_block_from_origin(wx, wz, lx, ly, lz)
                    if block != AIR:
                        self.place_generated(lx, ly, lz, block)

    def highest_solid_local(self, x: int, z: int) -> int:
        if not (0 <= x < CHUNK_SIZE and 0 <= z < CHUNK_SIZE):
            return -1
        for y in range(self.max_y, -1, -1):
            if self.get_local(x, y, z) != AIR:
                return y
        return -1

    def highest_ground_local(self, x: int, z: int) -> int:
        if not (0 <= x < CHUNK_SIZE and 0 <= z < CHUNK_SIZE):
            return -1
        for y in range(self.max_y, -1, -1):
            block = self.get_local(x, y, z)
            if block != AIR and block != WATER:
                return y
        return -1

    def build_mesh_data(self, world: World) -> tuple[list[float], list[float], list[float], list[int], int]:
        return self.build_full_mesh_data(world)

    def build_full_mesh_data(self, world: World) -> tuple[list[float], list[float], list[float], list[int], int]:
        vertices: list[float] = []
        normals: list[float] = []
        texcoords: list[float] = []
        indices: list[int] = []
        visible_faces = 0

        for y in range(self.max_y + 1):
            for lz in range(CHUNK_SIZE):
                wz = self.world_z + lz
                for lx in range(CHUNK_SIZE):
                    wx = self.world_x + lx
                    block = self.get_local(lx, y, lz)
                    if block == AIR:
                        continue
                    if is_plant_block(block):
                        visible_faces += add_plant_mesh(vertices, normals, texcoords, indices, wx, y, wz, block)
                        continue

                    for face in FACES:
                        ox, oy, oz = face.offset
                        if is_occluding_block(world.peek(wx + ox, y + oy, wz + oz)):
                            continue

                        corners = tuple(
                            (wx + corner_x, y + corner_y, wz + corner_z)
                            for corner_x, corner_y, corner_z in face.corners
                        )
                        add_quad(
                            vertices,
                            normals,
                            texcoords,
                            indices,
                            corners,
                            face.normal,
                            texture_for_face(block, face.name),
                        )
                        visible_faces += 1

        return vertices, normals, texcoords, indices, visible_faces

    def highest_in_area(self, lx0: int, lz0: int, width: int, depth: int) -> tuple[int, int]:
        best_y = -1
        best_block = AIR
        for lx in range(lx0, min(lx0 + width, CHUNK_SIZE)):
            for lz in range(lz0, min(lz0 + depth, CHUNK_SIZE)):
                y = self.highest_solid_local(lx, lz)
                if y > best_y:
                    best_y = y
                    best_block = self.get_local(lx, y, lz) if y >= 0 else AIR
        return best_y, best_block

    def rebuild_mesh(
        self,
        world: World,
        program: pyglet.graphics.shader.ShaderProgram,
        batch: Batch,
        group: VoxelGroup,
    ) -> None:
        vertices, normals, texcoords, indices, faces = self.build_mesh_data(world)
        mesh = MeshData(self.cx, self.cz, self.version, vertices, normals, texcoords, indices, faces)
        self.upload_mesh_data(program, batch, group, mesh)

    def upload_mesh_data(
        self,
        program: pyglet.graphics.shader.ShaderProgram,
        batch: Batch,
        group: VoxelGroup,
        mesh: MeshData,
    ) -> None:
        if self.vertex_list is not None:
            self.vertex_list.delete()
            self.vertex_list = None

        self.visible_faces = mesh.faces
        self.rendered_vertices = len(mesh.vertices) // 3

        if mesh.vertices:
            self.vertex_list = program.vertex_list_indexed(
                len(mesh.vertices) // 3,
                gl.GL_TRIANGLES,
                mesh.indices,
                batch=batch,
                group=group,
                POSITION=("f", mesh.vertices),
                NORMAL=("f", mesh.normals),
                TEXCOORD_0=("f", mesh.texcoords),
            )

        self.dirty = False
        self.mesh_pending = False

    def delete_mesh(self) -> None:
        if self.vertex_list is not None:
            self.vertex_list.delete()
            self.vertex_list = None
        self.mesh_pending = False

def chunk_snapshot(chunk: Chunk) -> Chunk:
    snapshot = Chunk.__new__(Chunk)
    snapshot.cx = chunk.cx
    snapshot.cz = chunk.cz
    snapshot.world_x = chunk.world_x
    snapshot.world_z = chunk.world_z
    snapshot.blocks = bytes(chunk.blocks)
    snapshot.vertex_list = None
    snapshot.visible_faces = chunk.visible_faces
    snapshot.rendered_vertices = chunk.rendered_vertices
    snapshot.block_count = chunk.block_count
    snapshot.max_y = chunk.max_y
    snapshot.dirty = chunk.dirty
    snapshot.version = chunk.version
    snapshot.mesh_pending = False
    return snapshot

class SnapshotWorld:
    def __init__(self, chunks: dict[tuple[int, int], bytes], edits: dict[tuple[int, int, int], int]) -> None:
        self.chunks = chunks
        self.edits = edits

    @staticmethod
    def in_height(y: int) -> bool:
        return 0 <= y < WORLD_HEIGHT

    def peek(self, wx: int, y: int, wz: int) -> int:
        if not self.in_height(y):
            return AIR

        cx, cz, lx, lz = world_to_chunk(wx, wz)
        blocks = self.chunks.get((cx, cz))
        if blocks is not None:
            return blocks[Chunk.index(lx, y, lz)]

        edited = self.edits.get((wx, y, wz))
        if edited is not None:
            return edited
        return procedural_block_at(wx, y, wz)

    def highest_solid_y(self, wx: int, wz: int) -> int:
        cx, cz, lx, lz = world_to_chunk(wx, wz)
        blocks = self.chunks.get((cx, cz))
        if blocks is not None:
            for y in range(WORLD_HEIGHT - 1, -1, -1):
                if blocks[Chunk.index(lx, y, lz)] != AIR:
                    return y
            return -1

        best = procedural_highest_solid_y(wx, wz)
        for (edited_x, edited_y, edited_z), block in self.edits.items():
            if edited_x == wx and edited_z == wz and block != AIR:
                best = max(best, edited_y)

        if self.edits.get((wx, best, wz)) == AIR:
            for y in range(best - 1, -1, -1):
                if self.peek(wx, y, wz) != AIR:
                    return y
            return -1
        return best

    def highest_in_area(self, wx0: int, wz0: int, width: int, depth: int) -> tuple[int, int]:
        best_y = -1
        best_block = AIR
        for wx in range(wx0, wx0 + width):
            for wz in range(wz0, wz0 + depth):
                y = self.highest_solid_y(wx, wz)
                if y > best_y:
                    best_y = y
                    best_block = self.peek(wx, y, wz) if y >= 0 else AIR
        return best_y, best_block

def generate_chunk_job(cx: int, cz: int, edits: dict[tuple[int, int, int], int]) -> Chunk:
    return Chunk(cx, cz, edits)

def build_mesh_job(chunk: Chunk, world: SnapshotWorld, version: int) -> MeshData:
    vertices, normals, texcoords, indices, faces = chunk.build_mesh_data(world)
    return MeshData(chunk.cx, chunk.cz, version, vertices, normals, texcoords, indices, faces)

def worker_ping() -> bool:
    return True

def can_use_process_workers() -> bool:
    return USE_PROCESS_WORKERS and os.path.isfile(sys.argv[0])

def make_worker_executor(max_workers: int, name: str) -> concurrent.futures.Executor:
    if can_use_process_workers():
        context = multiprocessing.get_context("spawn")
        return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=context)
    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)

class World:
    def __init__(self, program: pyglet.graphics.shader.ShaderProgram, group: VoxelGroup) -> None:
        self.program = program
        self.group = group
        self.batch = Batch()
        self.chunks: dict[tuple[int, int], Chunk] = {}
        self.edits: dict[tuple[int, int, int], int] = {}
        self.process_workers = can_use_process_workers()
        self.chunk_executor = make_worker_executor(CHUNK_WORKER_COUNT, "voxel-chunk")
        self.mesh_executor = make_worker_executor(MESH_WORKER_COUNT, "voxel-mesh")
        self._warm_futures = [
            self.chunk_executor.submit(worker_ping),
            self.mesh_executor.submit(worker_ping),
        ]
        self.pending_chunk_jobs: dict[tuple[int, int], concurrent.futures.Future[Chunk]] = {}
        self.pending_mesh_jobs: dict[tuple[int, int], concurrent.futures.Future[MeshData]] = {}
        self.visible_faces = 0
        self.rendered_vertices = 0
        self.drawn_chunks = 0
        self.drawn_faces = 0
        self.current_center = (0, 0)
        self.stream_frame = 0
        self.threaded = True
        self.render_distance = RENDER_DISTANCE
        self._render_offsets_distance = -1
        self._render_offsets: list[tuple[int, int, int]] = []

    @staticmethod
    def in_height(y: int) -> bool:
        return 0 <= y < WORLD_HEIGHT

    def ensure_chunk(self, cx: int, cz: int) -> Chunk:
        key_pos = (cx, cz)
        chunk = self.chunks.get(key_pos)
        if chunk is None:
            chunk = Chunk(cx, cz, self.edits)
            self.chunks[key_pos] = chunk
            self.mark_chunk_dirty(cx, cz)
        return chunk

    def request_chunk(self, cx: int, cz: int) -> bool:
        key_pos = (cx, cz)
        if key_pos in self.chunks or key_pos in self.pending_chunk_jobs:
            return False
        if len(self.pending_chunk_jobs) >= MAX_PENDING_CHUNK_JOBS:
            return False
        edits = dict(self.edits)
        self.pending_chunk_jobs[key_pos] = self.chunk_executor.submit(generate_chunk_job, cx, cz, edits)
        return True

    def set_render_distance(self, render_distance: int) -> None:
        self.render_distance = int(clamp(render_distance, MIN_RENDER_DISTANCE, MAX_RENDER_DISTANCE))
        self._render_offsets_distance = -1

    def render_offsets(self) -> list[tuple[int, int, int]]:
        if self._render_offsets_distance == self.render_distance:
            return self._render_offsets
        offsets: list[tuple[int, int, int]] = []
        for dz in range(-self.render_distance, self.render_distance + 1):
            for dx in range(-self.render_distance, self.render_distance + 1):
                distance = max(abs(dx), abs(dz))
                if distance <= self.render_distance:
                    offsets.append((distance, dx, dz))
        offsets.sort(key=lambda item: (item[0], item[1] * item[1] + item[2] * item[2]))
        self._render_offsets = offsets
        self._render_offsets_distance = self.render_distance
        return offsets

    def collect_completed_jobs(
        self,
        max_chunk_completions: int = CHUNK_COMPLETIONS_PER_FRAME,
        max_mesh_uploads: int = MESH_UPLOADS_PER_FRAME,
        allow_mesh_upload: bool = True,
    ) -> None:
        deadline = time.perf_counter() + STREAM_TIME_BUDGET
        completed_chunks = 0
        for key_pos, future in list(self.pending_chunk_jobs.items()):
            if not future.done():
                continue
            if completed_chunks >= max_chunk_completions:
                break
            if time.perf_counter() >= deadline:
                break
            del self.pending_chunk_jobs[key_pos]
            if key_pos in self.chunks:
                continue
            try:
                chunk = future.result()
            except Exception as exc:
                print(f"Chunk generation failed for {key_pos}: {exc}")
                continue
            self.chunks[key_pos] = chunk
            chunk.dirty = True
            completed_chunks += 1

        if not allow_mesh_upload:
            return

        uploaded = 0
        for key_pos, future in list(self.pending_mesh_jobs.items()):
            if not future.done():
                continue
            if uploaded >= max_mesh_uploads:
                break
            if time.perf_counter() >= deadline:
                break
            del self.pending_mesh_jobs[key_pos]
            try:
                mesh = future.result()
            except Exception as exc:
                print(f"Mesh build failed for {key_pos}: {exc}")
                chunk = self.chunks.get(key_pos)
                if chunk is not None:
                    chunk.mesh_pending = False
                    chunk.dirty = True
                continue

            chunk = self.chunks.get(key_pos)
            if chunk is None:
                continue
            chunk.mesh_pending = False
            if chunk.version != mesh.version:
                chunk.dirty = True
                continue
            old_faces = chunk.visible_faces
            old_vertices = chunk.rendered_vertices
            chunk.upload_mesh_data(self.program, self.batch, self.group, mesh)
            self.visible_faces += chunk.visible_faces - old_faces
            self.rendered_vertices += chunk.rendered_vertices - old_vertices
            uploaded += 1

    def create_mesh_world_snapshot(self, cx: int, cz: int) -> SnapshotWorld:
        chunks: dict[tuple[int, int], bytes] = {}
        for sx in range(cx - 1, cx + 2):
            for sz in range(cz - 1, cz + 2):
                chunk = self.chunks.get((sx, sz))
                if chunk is not None:
                    chunks[(sx, sz)] = bytes(chunk.blocks)
        return SnapshotWorld(chunks, dict(self.edits))

    def request_mesh(self, chunk: Chunk) -> bool:
        key_pos = (chunk.cx, chunk.cz)
        if chunk.mesh_pending or key_pos in self.pending_mesh_jobs:
            return False
        if len(self.pending_mesh_jobs) >= MAX_PENDING_MESH_JOBS:
            return False

        snapshot = chunk_snapshot(chunk)
        world_snapshot = self.create_mesh_world_snapshot(chunk.cx, chunk.cz)
        version = chunk.version
        chunk.mesh_pending = True
        chunk.dirty = False
        self.pending_mesh_jobs[key_pos] = self.mesh_executor.submit(build_mesh_job, snapshot, world_snapshot, version)
        return True

    def load_initial_area(self, wx: int, wz: int) -> None:
        cx, cz, _, _ = world_to_chunk(wx, wz)
        for dz in range(-INITIAL_LOAD_RADIUS, INITIAL_LOAD_RADIUS + 1):
            for dx in range(-INITIAL_LOAD_RADIUS, INITIAL_LOAD_RADIUS + 1):
                self.ensure_chunk(cx + dx, cz + dz)

    def loaded_chunk_at_world(self, wx: int, wz: int) -> Chunk | None:
        cx, cz, _, _ = world_to_chunk(wx, wz)
        return self.chunks.get((cx, cz))

    def get(self, wx: int, y: int, wz: int) -> int:
        return self.peek(wx, y, wz)

    def peek(self, wx: int, y: int, wz: int) -> int:
        if not self.in_height(y):
            return AIR

        chunk = self.loaded_chunk_at_world(wx, wz)
        if chunk is not None:
            _, _, lx, lz = world_to_chunk(wx, wz)
            return chunk.get_local(lx, y, lz)

        edited = self.edits.get((wx, y, wz))
        if edited is not None:
            return edited
        return procedural_block_at(wx, y, wz)

    def is_solid(self, wx: int, y: int, wz: int) -> bool:
        if not self.in_height(y):
            return False

        chunk = self.loaded_chunk_at_world(wx, wz)
        if chunk is not None:
            _, _, lx, lz = world_to_chunk(wx, wz)
            return is_collidable_block(chunk.get_local(lx, y, lz))

        edited = self.edits.get((wx, y, wz))
        if edited is not None:
            return is_collidable_block(edited)
        return procedural_collision_solid(wx, y, wz)

    def set(self, wx: int, y: int, wz: int, block: int) -> bool:
        if not self.in_height(y):
            return False

        cx, cz, lx, lz = world_to_chunk(wx, wz)
        chunk = self.ensure_chunk(cx, cz)
        changed = chunk.set_local(lx, y, lz, block)
        if not changed:
            return False

        original = procedural_block_at(wx, y, wz)
        if block == original:
            self.edits.pop((wx, y, wz), None)
        else:
            self.edits[(wx, y, wz)] = block

        self.mark_chunk_dirty(cx, cz)
        if lx == 0:
            self.mark_chunk_dirty(cx - 1, cz)
        elif lx == CHUNK_SIZE - 1:
            self.mark_chunk_dirty(cx + 1, cz)
        if lz == 0:
            self.mark_chunk_dirty(cx, cz - 1)
        elif lz == CHUNK_SIZE - 1:
            self.mark_chunk_dirty(cx, cz + 1)
        return True

    def mark_chunk_dirty(self, cx: int, cz: int) -> None:
        chunk = self.chunks.get((cx, cz))
        if chunk is not None:
            chunk.dirty = True

    def highest_solid_y(self, wx: int, wz: int) -> int:
        chunk = self.loaded_chunk_at_world(wx, wz)
        if chunk is not None:
            _, _, lx, lz = world_to_chunk(wx, wz)
            return chunk.highest_solid_local(lx, lz)

        best = procedural_highest_solid_y(wx, wz)
        for (edited_x, edited_y, edited_z), block in self.edits.items():
            if edited_x == wx and edited_z == wz and block != AIR:
                best = max(best, edited_y)

        if self.edits.get((wx, best, wz)) == AIR:
            for y in range(best - 1, -1, -1):
                if self.peek(wx, y, wz) != AIR:
                    return y
            return -1
        return best

    def highest_ground_y(self, wx: int, wz: int) -> int:
        chunk = self.loaded_chunk_at_world(wx, wz)
        if chunk is not None:
            _, _, lx, lz = world_to_chunk(wx, wz)
            return chunk.highest_ground_local(lx, lz)
        return terrain_height(wx, wz)

    def highest_in_area(self, wx0: int, wz0: int, width: int, depth: int) -> tuple[int, int]:
        best_y = -1
        best_block = AIR
        for wx in range(wx0, wx0 + width):
            for wz in range(wz0, wz0 + depth):
                y = self.highest_solid_y(wx, wz)
                if y > best_y:
                    best_y = y
                    best_block = self.peek(wx, y, wz) if y >= 0 else AIR
        return best_y, best_block

    def chunk_in_mesh_view(
        self,
        cx: int,
        cz: int,
        px: float,
        pz: float,
        view_direction: Vec3 | None,
    ) -> bool:
        if view_direction is None:
            return True

        center_x = cx * CHUNK_SIZE + CHUNK_SIZE / 2
        center_z = cz * CHUNK_SIZE + CHUNK_SIZE / 2
        dx = center_x - px
        dz = center_z - pz
        distance = math.hypot(dx, dz)
        if distance < CHUNK_SIZE * 2.5:
            return True
        if distance < 0.001:
            return True

        horizontal = math.hypot(view_direction.x, view_direction.z)
        if horizontal < 0.001:
            return True
        dot = (dx / distance) * (view_direction.x / horizontal) + (dz / distance) * (view_direction.z / horizontal)
        return dot > -0.20

    def update_around(
        self,
        position: Iterable[float],
        view_direction: Vec3 | None = None,
        player_moving: bool = False,
        max_new_chunks: int = CHUNKS_PER_FRAME,
        max_meshes: int = MESHES_PER_FRAME,
    ) -> None:
        self.stream_frame += 1
        stream_paused = player_moving and not STREAM_WHILE_MOVING
        allow_mesh_upload = not stream_paused and self.stream_frame % MESH_UPLOAD_INTERVAL == 0
        self.collect_completed_jobs(allow_mesh_upload=allow_mesh_upload)
        px, _, pz = position
        center = world_to_chunk(math.floor(px), math.floor(pz))[:2]
        self.current_center = center
        if stream_paused:
            return

        render_distance = self.render_distance
        desired = [(distance, center[0] + dx, center[1] + dz) for distance, dx, dz in self.render_offsets()]
        if max_new_chunks > 0:
            created = 0
            for _, cx, cz in desired:
                if (cx, cz) not in self.chunks and (cx, cz) not in self.pending_chunk_jobs:
                    if self.request_chunk(cx, cz):
                        created += 1
                        if created >= max_new_chunks:
                            break

        for key_pos, chunk in list(self.chunks.items()):
            cx, cz = key_pos
            if max(abs(cx - center[0]), abs(cz - center[1])) > render_distance + 1:
                self.visible_faces -= chunk.visible_faces
                self.rendered_vertices -= chunk.rendered_vertices
                chunk.delete_mesh()
                del self.chunks[key_pos]
        for key_pos, future in list(self.pending_chunk_jobs.items()):
            cx, cz = key_pos
            if max(abs(cx - center[0]), abs(cz - center[1])) > render_distance + 1 and future.cancel():
                del self.pending_chunk_jobs[key_pos]

        if max_meshes > 0:
            rebuilt = 0
            dirty_chunks = [
                (max(abs(cx - center[0]), abs(cz - center[1])), cx, cz, chunk)
                for (cx, cz), chunk in self.chunks.items()
                if chunk.dirty
                and max(abs(cx - center[0]), abs(cz - center[1])) <= render_distance
                and self.chunk_in_mesh_view(cx, cz, px, pz, view_direction)
            ]
            dirty_chunks.sort(key=lambda item: item[0])
            for _, cx, cz, chunk in dirty_chunks:
                if self.request_mesh(chunk):
                    rebuilt += 1
                if rebuilt >= max_meshes:
                    break

        self.visible_faces = max(0, self.visible_faces)
        self.rendered_vertices = max(0, self.rendered_vertices)

    def update_stats(self) -> None:
        self.visible_faces = sum(chunk.visible_faces for chunk in self.chunks.values())
        self.rendered_vertices = sum(chunk.rendered_vertices for chunk in self.chunks.values())

    def chunk_in_view(self, chunk: Chunk, player: Player) -> bool:
        px, _, pz = player.position
        center_x = chunk.world_x + CHUNK_SIZE / 2
        center_z = chunk.world_z + CHUNK_SIZE / 2
        dx = center_x - px
        dz = center_z - pz
        distance = math.hypot(dx, dz)
        if distance < CHUNK_SIZE * 2.5:
            return True
        if distance < 0.001:
            return True

        forward = player.forward()
        horizontal = math.hypot(forward.x, forward.z)
        if horizontal < 0.001:
            return True
        dot = (dx / distance) * (forward.x / horizontal) + (dz / distance) * (forward.z / horizontal)
        return dot > -0.35

    def draw(self, player: Player) -> None:
        visible_chunks: list[Chunk] = []
        self.drawn_chunks = 0
        self.drawn_faces = 0
        for chunk in self.chunks.values():
            if chunk.vertex_list is not None and self.chunk_in_view(chunk, player):
                visible_chunks.append(chunk)
                self.drawn_chunks += 1
                self.drawn_faces += chunk.visible_faces
        if not visible_chunks:
            return

        self.group.set_state()
        try:
            for chunk in visible_chunks:
                chunk.vertex_list.draw(gl.GL_TRIANGLES)
        finally:
            self.group.unset_state()

    def reset(self) -> None:
        for future in self.pending_chunk_jobs.values():
            future.cancel()
        for future in self.pending_mesh_jobs.values():
            future.cancel()
        self.pending_chunk_jobs.clear()
        self.pending_mesh_jobs.clear()
        for chunk in self.chunks.values():
            chunk.delete_mesh()
        self.chunks.clear()
        self.edits.clear()
        self.visible_faces = 0
        self.rendered_vertices = 0
        self.drawn_chunks = 0
        self.drawn_faces = 0

    def shutdown(self) -> None:
        self.reset()
        self.chunk_executor.shutdown(wait=False, cancel_futures=True)
        self.mesh_executor.shutdown(wait=False, cancel_futures=True)

@dataclass(frozen=True)
class Hit:
    block: tuple[int, int, int]
    previous: tuple[int, int, int] | None

class Player:
    radius = 0.30
    height = 1.78
    eye_height = 1.62
    walk_speed = 5.0
    sprint_speed = 9.0
    fly_speed = 9.0
    jump_speed = 8.2
    gravity = -24.0

    def __init__(self, world: World) -> None:
        spawn_x = CHUNK_SIZE / 2
        spawn_z = CHUNK_SIZE / 2
        world.load_initial_area(int(spawn_x), int(spawn_z))
        spawn_y = world.highest_ground_y(int(spawn_x), int(spawn_z)) + 2.0
        self.position = [spawn_x + 0.5, spawn_y, spawn_z + 0.5]
        self.velocity = [0.0, 0.0, 0.0]
        self.yaw = 135.0
        self.pitch = -12.0
        self.on_ground = False
        self.flying = False

    def eye_position(self) -> Vec3:
        return Vec3(self.position[0], self.position[1] + self.eye_height, self.position[2])

    def forward(self) -> Vec3:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        return Vec3(
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch),
        ).normalize()

    def view_matrix(self) -> Mat4:
        eye = self.eye_position()
        return Mat4.look_at(eye, eye + self.forward(), Vec3(0.0, 1.0, 0.0))

    def aabb_at(self, position: Iterable[float]) -> tuple[float, float, float, float, float, float]:
        x, y, z = position
        return (
            x - self.radius,
            y,
            z - self.radius,
            x + self.radius,
            y + self.height,
            z + self.radius,
        )

    def collides_at(self, world: World, position: Iterable[float]) -> bool:
        min_x, min_y, min_z, max_x, max_y, max_z = self.aabb_at(position)
        for x in range(math.floor(min_x), math.floor(max_x - EPSILON) + 1):
            for y in range(math.floor(min_y), math.floor(max_y - EPSILON) + 1):
                for z in range(math.floor(min_z), math.floor(max_z - EPSILON) + 1):
                    if world.is_solid(x, y, z):
                        return True
        return False

    def overlaps_block(self, block_pos: tuple[int, int, int]) -> bool:
        min_x, min_y, min_z, max_x, max_y, max_z = self.aabb_at(self.position)
        bx, by, bz = block_pos
        return (
            min_x < bx + 1
            and max_x > bx
            and min_y < by + 1
            and max_y > by
            and min_z < bz + 1
            and max_z > bz
        )

    def update(self, dt: float, keys: key.KeyStateHandler, world: World) -> None:
        dt = min(dt, 0.05)

        yaw = math.radians(self.yaw)
        forward_x = math.sin(yaw)
        forward_z = -math.cos(yaw)
        right_x = math.cos(yaw)
        right_z = math.sin(yaw)

        move_x = 0.0
        move_z = 0.0
        if keys[key.W]:
            move_x += forward_x
            move_z += forward_z
        if keys[key.S]:
            move_x -= forward_x
            move_z -= forward_z
        if keys[key.D]:
            move_x += right_x
            move_z += right_z
        if keys[key.A]:
            move_x -= right_x
            move_z -= right_z

        length = math.hypot(move_x, move_z)
        if length:
            move_x /= length
            move_z /= length

        if self.flying:
            speed = self.fly_speed
            vertical = 0.0
            if keys[key.SPACE]:
                vertical += 1.0
            if keys[key.LSHIFT] or keys[key.RSHIFT]:
                vertical -= 1.0
            self.velocity[1] = vertical * speed
        else:
            sprinting = keys[key.LCTRL] or keys[key.RCTRL] or keys[key.LSHIFT] or keys[key.RSHIFT]
            speed = self.sprint_speed if sprinting and length > 0 else self.walk_speed
            if keys[key.SPACE] and self.on_ground:
                self.velocity[1] = self.jump_speed
            self.velocity[1] += self.gravity * dt

        self.on_ground = False
        self.move_axis(world, 0, move_x * speed * dt)
        self.move_axis(world, 2, move_z * speed * dt)
        self.move_axis(world, 1, self.velocity[1] * dt)

        if self.position[1] < -15:
            spawn_y = world.highest_ground_y(CHUNK_SIZE // 2, CHUNK_SIZE // 2) + 2.0
            self.position = [CHUNK_SIZE / 2 + 0.5, spawn_y, CHUNK_SIZE / 2 + 0.5]
            self.velocity = [0.0, 0.0, 0.0]

    def move_axis(self, world: World, axis: int, amount: float) -> None:
        if abs(amount) < 0.000001:
            return

        steps = max(1, int(abs(amount) / MOVE_COLLISION_STEP) + 1)
        step = amount / steps
        for _ in range(steps):
            next_pos = self.position.copy()
            next_pos[axis] += step
            if self.collides_at(world, next_pos):
                if axis == 1:
                    if step < 0:
                        self.on_ground = True
                    self.velocity[1] = 0.0
                return
            self.position = next_pos

def raycast(world: World, origin: Vec3, direction: Vec3, reach: float = MAX_REACH) -> Hit | None:
    previous: tuple[int, int, int] | None = None
    last_cell: tuple[int, int, int] | None = None
    steps = int(reach / RAY_STEP)

    for i in range(steps + 1):
        distance = i * RAY_STEP
        point = origin + direction * distance
        cell = (math.floor(point.x), math.floor(point.y), math.floor(point.z))

        if cell == last_cell:
            continue

        if world.is_solid(*cell):
            return Hit(cell, previous)

        previous = cell
        last_cell = cell

    return None

class MinecraftWindow(pyglet.window.Window):
    def __init__(self) -> None:
        config = gl.Config(double_buffer=True, depth_size=24)
        super().__init__(
            1280,
            720,
            "Python Infinite Voxel World",
            resizable=True,
            vsync=True,
            config=config,
        )
        self.set_minimum_size(640, 400)

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

        self.selected_index = 0
        self.inventory_open = False
        self.menu_open = False
        self.render_distance_dragging = False
        self.inventory_slots: list[pyglet.shapes.Rectangle] = []
        self.inventory_labels: list[pyglet.text.Label] = []
        self.inventory_title: pyglet.text.Label | None = None
        self.menu_shapes: list[object] = []
        self.minimap_shapes: list[object] = []
        self.minimap_color_cache: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._minimap_signature: tuple[int, tuple[int, int], int] | None = None
        self.mouse_captured = False
        self.fps = 0.0
        self._fps_time = 0.0
        self._fps_frames = 0
        self._last_space_press = -10.0

        self.texture = create_texture_atlas()
        self.program = pyglet.gl.current_context.create_program((VERTEX_SHADER, "vertex"), (FRAGMENT_SHADER, "fragment"))
        self.program.use()
        self.program["atlas"] = 0
        self.program["time"] = 0.0
        self.program["camera_pos"] = (0.0, 0.0, 0.0)
        self.program.stop()
        self.voxel_group = VoxelGroup(self.texture, self.program)
        self.world = World(self.program, self.voxel_group)
        self.player = Player(self.world)
        self.world.update_around(self.player.position, self.player.forward(), max_new_chunks=0, max_meshes=64)

        self.hud_batch = Batch()
        self.inventory_batch = Batch()
        self.menu_batch = Batch()
        self.crosshair_h = pyglet.shapes.Line(0, 0, 0, 0, thickness=2, color=(245, 245, 245, 220), batch=self.hud_batch)
        self.crosshair_v = pyglet.shapes.Line(0, 0, 0, 0, thickness=2, color=(245, 245, 245, 220), batch=self.hud_batch)
        self.info_label = pyglet.text.Label(
            "",
            x=12,
            y=self.height - 18,
            anchor_x="left",
            anchor_y="center",
            font_size=11,
            color=(240, 242, 238, 255),
            batch=self.hud_batch,
        )
        self.block_label = pyglet.text.Label(
            "",
            x=12,
            y=24,
            anchor_x="left",
            anchor_y="center",
            font_size=13,
            color=(255, 255, 255, 255),
            batch=self.hud_batch,
        )
        self.create_inventory_ui()
        self.create_menu_ui()
        self.update_hud_layout()
        self.update_hud_text()

        gl.glClearColor(0.53, 0.72, 0.92, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

        pyglet.clock.schedule_interval(self.update, 1 / 60)

    @property
    def selected_block(self) -> int:
        return INVENTORY_BLOCKS[self.selected_index]

    def create_inventory_ui(self) -> None:
        self.inventory_title = pyglet.text.Label(
            "Inventory",
            font_size=16,
            color=(245, 245, 245, 255),
            anchor_x="center",
            anchor_y="center",
            batch=self.inventory_batch,
        )
        for i, block in enumerate(INVENTORY_BLOCKS):
            slot = pyglet.shapes.Rectangle(0, 0, 48, 48, color=(48, 48, 48, 235), batch=self.inventory_batch)
            label = pyglet.text.Label(
                BLOCK_NAMES[block],
                font_size=8,
                color=(245, 245, 245, 255),
                anchor_x="center",
                anchor_y="center",
                multiline=True,
                width=44,
                batch=self.inventory_batch,
            )
            self.inventory_slots.append(slot)
            self.inventory_labels.append(label)

    def create_menu_ui(self) -> None:
        self.menu_overlay = pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(0, 0, 0, 145), batch=self.menu_batch)
        self.menu_panel = pyglet.shapes.Rectangle(0, 0, 560, 460, color=(32, 35, 34, 240), batch=self.menu_batch)
        self.menu_title = pyglet.text.Label(
            "Menu",
            font_size=22,
            color=(245, 245, 245, 255),
            anchor_x="center",
            anchor_y="center",
            batch=self.menu_batch,
        )
        self.render_distance_label = pyglet.text.Label(
            "",
            font_size=13,
            color=(235, 235, 235, 255),
            anchor_x="left",
            anchor_y="center",
            batch=self.menu_batch,
        )
        self.slider_track = pyglet.shapes.Rectangle(0, 0, 1, 6, color=(92, 96, 96, 255), batch=self.menu_batch)
        self.slider_fill = pyglet.shapes.Rectangle(0, 0, 1, 6, color=(111, 174, 227, 255), batch=self.menu_batch)
        self.slider_knob = pyglet.shapes.Rectangle(0, 0, 16, 24, color=(235, 238, 238, 255), batch=self.menu_batch)
        self.minimap_title = pyglet.text.Label(
            "Minimap",
            font_size=13,
            color=(235, 235, 235, 255),
            anchor_x="left",
            anchor_y="center",
            batch=self.menu_batch,
        )
        self.minimap_bg = pyglet.shapes.Rectangle(0, 0, 260, 260, color=(12, 17, 18, 255), batch=self.menu_batch)
        self.minimap_border_top = pyglet.shapes.Line(0, 0, 0, 0, thickness=1, color=(150, 158, 158, 255), batch=self.menu_batch)
        self.minimap_border_bottom = pyglet.shapes.Line(0, 0, 0, 0, thickness=1, color=(150, 158, 158, 255), batch=self.menu_batch)
        self.minimap_border_left = pyglet.shapes.Line(0, 0, 0, 0, thickness=1, color=(150, 158, 158, 255), batch=self.menu_batch)
        self.minimap_border_right = pyglet.shapes.Line(0, 0, 0, 0, thickness=1, color=(150, 158, 158, 255), batch=self.menu_batch)
        self.player_marker = pyglet.shapes.Rectangle(0, 0, 7, 7, color=(240, 50, 50, 255), batch=self.menu_batch)
        self.menu_hint = pyglet.text.Label(
            "ESC",
            font_size=11,
            color=(205, 210, 210, 255),
            anchor_x="center",
            anchor_y="center",
            batch=self.menu_batch,
        )
        self.slider_x = 0.0
        self.slider_y = 0.0
        self.slider_width = 320.0
        self.minimap_x = 0.0
        self.minimap_y = 0.0
        self.minimap_size = 260.0

    def update_inventory_layout(self) -> None:
        columns = 5
        slot = 54
        total_width = columns * slot
        start_x = self.width / 2 - total_width / 2
        start_y = self.height / 2 + 90
        if self.inventory_title is not None:
            self.inventory_title.x = self.width / 2
            self.inventory_title.y = start_y + 38
        for i, rect in enumerate(self.inventory_slots):
            col = i % columns
            row = i // columns
            x = start_x + col * slot
            y = start_y - row * slot
            rect.x = x
            rect.y = y
            rect.color = (94, 94, 94, 245) if i == self.selected_index else (48, 48, 48, 235)
            self.inventory_labels[i].x = x + 24
            self.inventory_labels[i].y = y + 24

    def update_menu_layout(self) -> None:
        panel_width = min(620, max(460, self.width - 80))
        panel_height = min(500, max(420, self.height - 80))
        panel_x = self.width / 2 - panel_width / 2
        panel_y = self.height / 2 - panel_height / 2

        self.menu_overlay.width = self.width
        self.menu_overlay.height = self.height
        self.menu_panel.x = panel_x
        self.menu_panel.y = panel_y
        self.menu_panel.width = panel_width
        self.menu_panel.height = panel_height
        self.menu_title.x = self.width / 2
        self.menu_title.y = panel_y + panel_height - 34

        self.slider_x = panel_x + 52
        self.slider_y = panel_y + panel_height - 110
        self.slider_width = panel_width - 104
        self.render_distance_label.x = self.slider_x
        self.render_distance_label.y = self.slider_y + 36
        self.slider_track.x = self.slider_x
        self.slider_track.y = self.slider_y
        self.slider_track.width = self.slider_width
        self.slider_fill.x = self.slider_x
        self.slider_fill.y = self.slider_y

        self.minimap_size = min(300, panel_height - 190, panel_width - 104)
        self.minimap_x = self.width / 2 - self.minimap_size / 2
        self.minimap_y = panel_y + 54
        self.minimap_title.x = self.minimap_x
        self.minimap_title.y = self.minimap_y + self.minimap_size + 20
        self.minimap_bg.x = self.minimap_x
        self.minimap_bg.y = self.minimap_y
        self.minimap_bg.width = self.minimap_size
        self.minimap_bg.height = self.minimap_size
        self.minimap_border_top.x = self.minimap_x
        self.minimap_border_top.y = self.minimap_y + self.minimap_size
        self.minimap_border_top.x2 = self.minimap_x + self.minimap_size
        self.minimap_border_top.y2 = self.minimap_y + self.minimap_size
        self.minimap_border_bottom.x = self.minimap_x
        self.minimap_border_bottom.y = self.minimap_y
        self.minimap_border_bottom.x2 = self.minimap_x + self.minimap_size
        self.minimap_border_bottom.y2 = self.minimap_y
        self.minimap_border_left.x = self.minimap_x
        self.minimap_border_left.y = self.minimap_y
        self.minimap_border_left.x2 = self.minimap_x
        self.minimap_border_left.y2 = self.minimap_y + self.minimap_size
        self.minimap_border_right.x = self.minimap_x + self.minimap_size
        self.minimap_border_right.y = self.minimap_y
        self.minimap_border_right.x2 = self.minimap_x + self.minimap_size
        self.minimap_border_right.y2 = self.minimap_y + self.minimap_size
        self.menu_hint.x = panel_x + panel_width - 34
        self.menu_hint.y = panel_y + panel_height - 28
        self.update_render_distance_ui()
        self._minimap_signature = None

    def update_render_distance_ui(self) -> None:
        value = self.world.render_distance
        t = (value - MIN_RENDER_DISTANCE) / (MAX_RENDER_DISTANCE - MIN_RENDER_DISTANCE)
        fill_width = max(1, self.slider_width * t)
        self.slider_fill.width = fill_width
        self.slider_knob.x = self.slider_x + fill_width - self.slider_knob.width / 2
        self.slider_knob.y = self.slider_y - 9
        self.render_distance_label.text = f"Render Distance: {value} Chunks"

    def set_render_distance_from_mouse(self, x: float) -> None:
        t = clamp((x - self.slider_x) / max(self.slider_width, 1), 0.0, 1.0)
        value = round(MIN_RENDER_DISTANCE + t * (MAX_RENDER_DISTANCE - MIN_RENDER_DISTANCE))
        self.world.set_render_distance(value)
        self.update_render_distance_ui()
        self.update_hud_text()

    def update_minimap(self) -> None:
        if not self.menu_open:
            return
        signature = (len(self.world.chunks), self.world.current_center, self.world.render_distance)
        if signature == self._minimap_signature:
            return
        self._minimap_signature = signature

        for shape in self.minimap_shapes:
            shape.delete()
        self.minimap_shapes.clear()

        chunks = list(self.world.chunks)
        if not chunks:
            return

        center_cx, center_cz = self.world.current_center
        radius = max(self.world.render_distance, 1)
        cell = max(2.0, self.minimap_size / (radius * 2 + 1))
        map_span = cell * (radius * 2 + 1)
        origin_x = self.minimap_x + self.minimap_size / 2 - map_span / 2
        origin_y = self.minimap_y + self.minimap_size / 2 - map_span / 2
        for cx, cz in chunks:
            dx = cx - center_cx
            dz = cz - center_cz
            if abs(dx) > radius or abs(dz) > radius:
                continue
            color = self.minimap_color_cache.get((cx, cz))
            if color is None:
                wx = cx * CHUNK_SIZE + CHUNK_SIZE // 2
                wz = cz * CHUNK_SIZE + CHUNK_SIZE // 2
                color = biome_minimap_color(dominant_biome(wx, wz))
                self.minimap_color_cache[(cx, cz)] = color
            rect = pyglet.shapes.Rectangle(
                origin_x + (dx + radius) * cell,
                origin_y + (radius - dz) * cell,
                max(1, math.ceil(cell)),
                max(1, math.ceil(cell)),
                color=color + (230,),
                batch=self.menu_batch,
            )
            self.minimap_shapes.append(rect)

        marker_size = max(6, min(10, int(cell * 1.8)))
        self.player_marker.delete()
        self.player_marker = pyglet.shapes.Rectangle(
            self.minimap_x + self.minimap_size / 2 - marker_size / 2,
            self.minimap_y + self.minimap_size / 2 - marker_size / 2,
            marker_size,
            marker_size,
            color=(240, 50, 50, 255),
            batch=self.menu_batch,
        )

    def inventory_slot_at(self, x: float, y: float) -> int | None:
        for i, rect in enumerate(self.inventory_slots):
            if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
                return i
        return None

    def set_mouse_capture(self, captured: bool) -> None:
        self.mouse_captured = captured
        self.set_exclusive_mouse(captured)

    def set_menu_open(self, open_menu: bool) -> None:
        self.menu_open = open_menu
        self.render_distance_dragging = False
        if open_menu:
            self.inventory_open = False
            self.set_mouse_capture(False)
            self.update_menu_layout()
            self.update_minimap()
        self.update_hud_text()

    def set_flying(self, flying: bool) -> None:
        self.player.flying = flying
        self.player.velocity[1] = 0.0
        if flying:
            self.player.on_ground = False

    def update(self, dt: float) -> None:
        self._fps_time += dt
        self._fps_frames += 1
        if self._fps_time >= 0.25:
            self.fps = self._fps_frames / self._fps_time
            self._fps_time = 0.0
            self._fps_frames = 0

        if not self.menu_open:
            self.player.update(dt, self.keys, self.world)
        player_moving = (
            not self.menu_open
            and (
            self.keys[key.W]
            or self.keys[key.A]
            or self.keys[key.S]
            or self.keys[key.D]
            or (self.player.flying and (self.keys[key.SPACE] or self.keys[key.LSHIFT] or self.keys[key.RSHIFT]))
            )
        )
        self.world.update_around(self.player.position, self.player.forward(), player_moving=player_moving)
        self.update_hud_text()
        self.update_minimap()

    def update_hud_layout(self) -> None:
        cx = self.width / 2
        cy = self.height / 2
        size = 9
        self.crosshair_h.x = cx - size
        self.crosshair_h.y = cy
        self.crosshair_h.x2 = cx + size
        self.crosshair_h.y2 = cy
        self.crosshair_v.x = cx
        self.crosshair_v.y = cy - size
        self.crosshair_v.x2 = cx
        self.crosshair_v.y2 = cy + size

        self.info_label.y = self.height - 18
        self.update_inventory_layout()
        self.update_menu_layout()

    def update_hud_text(self) -> None:
        lock_text = "Menu" if self.menu_open else ("ESC menu" if self.mouse_captured else "Click to capture mouse")
        fly_text = "Fly" if self.player.flying else "Walk"
        px = math.floor(self.player.position[0])
        pz = math.floor(self.player.position[2])
        biome = biome_name_at(px, pz)
        self.info_label.text = (
            f"FPS {self.fps:.0f} | Chunks {self.world.drawn_chunks}/{len(self.world.chunks)} | "
            f"Faces {self.world.drawn_faces}/{self.world.visible_faces} | "
            f"RD {self.world.render_distance} | "
            f"Jobs {len(self.world.pending_chunk_jobs)}/{len(self.world.pending_mesh_jobs)} | "
            f"{fly_text} | {lock_text}"
        )
        slot = self.selected_index + 1 if self.selected_index < HOTBAR_SIZE else "E"
        self.block_label.text = f"{slot}: {BLOCK_NAMES[self.selected_block]} | {biome}"

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.update_hud_layout()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        if self.menu_open:
            if self.render_distance_dragging:
                self.set_render_distance_from_mouse(x)
            return
        if not self.mouse_captured:
            return
        sensitivity = 0.14
        self.player.yaw += dx * sensitivity
        self.player.pitch = clamp(self.player.pitch + dy * sensitivity, -89.0, 89.0)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if self.menu_open and self.render_distance_dragging:
            self.set_render_distance_from_mouse(x)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.menu_open:
            self.render_distance_dragging = False

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.menu_open:
            slider_hit = (
                self.slider_x - 12 <= x <= self.slider_x + self.slider_width + 12
                and self.slider_y - 18 <= y <= self.slider_y + 24
            )
            if slider_hit:
                self.render_distance_dragging = True
                self.set_render_distance_from_mouse(x)
            return

        if self.inventory_open:
            slot = self.inventory_slot_at(x, y)
            if slot is not None:
                self.selected_index = slot
                self.update_inventory_layout()
                self.update_hud_text()
            return

        if not self.mouse_captured:
            self.set_mouse_capture(True)
            return

        hit = raycast(self.world, self.player.eye_position(), self.player.forward())
        if hit is None:
            return

        if button == mouse.LEFT:
            bx, by, bz = hit.block
            block = self.world.get(bx, by, bz)
            if block not in (BEDROCK, WATER):
                self.world.set(bx, by, bz, AIR)
        elif button == mouse.RIGHT and hit.previous is not None:
            px, py, pz = hit.previous
            place_pos = (px, py, pz)
            if self.world.in_height(py) and not self.player.overlaps_block(place_pos):
                self.world.set(px, py, pz, self.selected_block)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if self.menu_open or self.inventory_open:
            return
        if scroll_y > 0:
            self.selected_index = (min(self.selected_index, HOTBAR_SIZE - 1) - 1) % HOTBAR_SIZE
        elif scroll_y < 0:
            self.selected_index = (min(self.selected_index, HOTBAR_SIZE - 1) + 1) % HOTBAR_SIZE
        self.update_hud_text()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.ESCAPE:
            self.set_menu_open(not self.menu_open)
            return
        if self.menu_open:
            return
        if symbol == key.F:
            self.set_flying(not self.player.flying)
        elif symbol == key.E:
            self.inventory_open = not self.inventory_open
            if self.inventory_open:
                self.set_mouse_capture(False)
                self.update_inventory_layout()
        elif symbol == key.SPACE:
            now = time.perf_counter()
            if now - self._last_space_press <= 0.35:
                self.set_flying(not self.player.flying)
                self._last_space_press = -10.0
            else:
                self._last_space_press = now
        elif symbol == key.R:
            self.world.reset()
            self.player = Player(self.world)
            self._last_space_press = -10.0
            self.world.update_around(self.player.position, self.player.forward(), max_new_chunks=0, max_meshes=64)
        elif symbol in (key._1, key._2, key._3, key._4, key._5, key._6, key._7, key._8, key._9):
            self.selected_index = symbol - key._1
            self.update_inventory_layout()
            self.update_hud_text()

    def on_close(self) -> None:
        self.world.shutdown()
        super().on_close()

    def set_3d(self) -> None:
        gl.glEnable(gl.GL_DEPTH_TEST)
        aspect = max(self.width / max(self.height, 1), 0.1)
        far_plane = max(320.0, (self.world.render_distance + 3) * CHUNK_SIZE * 1.55)
        self.projection = Mat4.perspective_projection(aspect, 0.05, far_plane, 70.0)
        self.view = self.player.view_matrix()

    def set_2d(self) -> None:
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        self.projection = Mat4.orthogonal_projection(0, self.width, 0, self.height, -1, 1)
        self.view = Mat4()

    def on_draw(self) -> None:
        self.clear()
        eye = self.player.eye_position()
        self.program.use()
        self.program["time"] = time.perf_counter()
        self.program["camera_pos"] = (eye.x, eye.y, eye.z)
        self.program.stop()
        self.set_3d()
        self.world.draw(self.player)
        self.set_2d()
        self.hud_batch.draw()
        if self.inventory_open:
            self.inventory_batch.draw()
        if self.menu_open:
            self.menu_batch.draw()

def main() -> None:
    multiprocessing.freeze_support()
    MinecraftWindow()
    pyglet.app.run()

if __name__ == "__main__":
    main()
