# Minecraft_clone_python


```bash
   pip install pyglet

```

3. **Run the engine:**
```bash
python main.py


```
![Screenshot der Voxel Engine - Fluss und Landschaft](https://github.com/mareksdfgh/Minecraft_clone_python/blob/main/Bildschirmfoto%202026-05-11%20um%2016.56.22.png)
)

---

# Python Infinite Voxel Engine

A high-performance, infinite procedural voxel sandbox game written entirely in Python. Utilizing the `pyglet` library for windowing and OpenGL bindings, this project features a custom GLSL shader pipeline, multi-process chunk streaming, and complete procedural texture generation.

## Features

* **Infinite Procedural Terrain:** Utilizes value noise and multi-layered heightmaps to generate a seamless, endless world.
* **Complex Biome System:** Features 11 distinct biomes with unique generation rules, flora, and topography:
  * *Plains, Forest, Desert, Ocean, Savanna, Epic Hills,East Lowlands, Snowy Taiga, Badlands, Alpine Peaks, and Wetland.*
* **Multiprocessing Architecture:** Chunk data generation and mesh building are offloaded to asynchronous worker processes/threads, ensuring smooth framerates while the player moves through the world.
* **Procedural Texturing:** Zero external image assets are required. The game generates a complete texture atlas at runtime using algorithmic pseudo-noise.
* **Physics & Interaction:** 
  * AABB collision detection.
  * Accurate raycasting for breaking and placing blocks.
  * Walk, sprint, jump, and fly mechanics.
* **User Interface:**
  * Real-time dynamic minimap colored by biome data.
  * Interactive ESC menu with an adjustable Render Distance slider.
  * Inventory and hotbar system with block selection.

## Requirements

* **Python 3.10** or higher
* **pyglet**

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mareksdfgh/Minecraft_clone_python.git




2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install pyglet

```


3. **Run the engine:**
```bash
python main.py

```



## Controls

| Action | Key / Input |
| --- | --- |
| **Move** | `W`, `A`, `S`, `D` |
| **Jump** | `Space` |
| **Sprint** | Hold `Shift` or `Ctrl` |
| **Toggle Fly Mode** | `F` |
| **Ascend / Descend (Fly)** | `Space` / `Shift` |
| **Break Block** | `Left Mouse Button` |
| **Place Block** | `Right Mouse Button` |
| **Select Block** | `1` - `9` or `Mouse Scroll Wheel` |
| **Open Inventory** | `E` |
| **Open Menu / Free Mouse** | `ESC` |

## Architecture Overview

* **Chunking:** The world is divided into `16x180x16` chunks. Only visible faces are added to the chunk meshes (Face Culling) to drastically reduce the polygon count.
* **Rendering:** Meshes are compiled into Pyglet `Batch` and `VertexList` objects for efficient drawing. Frustum culling logic is applied to prevent rendering chunks behind the camera.
* **Threading/Processing:** The `World` class manages job queues. The `CHUNK_WORKER_COUNT` and `MESH_WORKER_COUNT` are dynamically scaled based on the system's available CPU cores to balance background generation and main-thread rendering.


```

```
