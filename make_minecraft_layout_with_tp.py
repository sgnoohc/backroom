#!/usr/bin/env python3
# make_minecraft_layout_with_plate_finish_spawn_on_start.py
# Arena with clustered walls, adjustable walk height, Wardens spawned on start_run,
# start TP, clear function, timer (start/tick), and a plate-triggered finish.

import argparse
import random
import math
from typing import List, Tuple

# ----------------- Layout generation (clustered obstacles) -----------------
def in_bounds(n: int, r: int, c: int, margin: int = 1) -> bool:
    return margin <= r < n - margin and margin <= c < n - margin

def stamp_blob(grid: List[List[str]], r: int, c: int, radius: int, mark: str = '#') -> None:
    n = len(grid)
    for rr in range(r - radius, r + radius + 1):
        for cc in range(c - radius, c + radius + 1):
            if in_bounds(n, rr, cc, margin=1) and max(abs(rr - r), abs(cc - c)) <= radius:
                grid[rr][cc] = mark

def generate_open_layout_clustered(
    n: int,
    rng: random.Random,
    density: float = 0.12,
    cluster_radius: int = 1,
    walk_steps: int = 1,
    min_exit_frac: float = 0.8,
    min_exit_blocks: int = 0,
) -> List[List[str]]:
    grid = [[' ' for _ in range(n)] for _ in range(n)]
    start = (1, 0)   # start on left border
    grid[start[0]][start[1]] = 'S'

    # --- clustered walls via short random walks ---
    interior_area = (n - 2) * (n - 2)
    avg_blob = (2 * cluster_radius + 1) ** 2
    target = max(1, int(density * interior_area))
    seeds = max(1, int(target / max(1, avg_blob // 2)))

    for _ in range(seeds):
        r = rng.randint(0, n - 2)
        c = rng.randint(0, n - 2)
        for _ in range(walk_steps):
            # stamp a small blob
            for rr in range(r - cluster_radius, r + cluster_radius + 1):
                for cc in range(c - cluster_radius, c + cluster_radius + 1):
                    if 1 <= rr < n - 1 and 1 <= cc < n - 1 and max(abs(rr - r), abs(cc - c)) <= cluster_radius:
                        grid[rr][cc] = '#'
            # move step
            r = max(1, min(n - 2, r + rng.choice([-1, 0, 1])))
            c = max(1, min(n - 2, c + rng.choice([-1, 0, 1])))

    # keep start clear
    grid[start[0]][start[1]] = 'S'

    # --- place exit at least 60% across the board from start (Chebyshev distance) ---
    def cheb(a, b): return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    # Enforce BOTH a fractional and absolute minimum
    dist_thresh = max(
        1,
        math.ceil(max(min_exit_frac * (n - 2), float(min_exit_blocks)))
    )

    # open tiles that satisfy distance
    candidates = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                  if grid[r][c] == ' ' and cheb((r, c), start) >= dist_thresh]

    if not candidates:
        # fallback to the farthest open tile (if layout is very dense)
        all_open = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1) if grid[r][c] == ' ']
        if not all_open:
            # extreme edge case: no open cell—just reopen one far corner
            fallback = (n - 2, n - 2)
            grid[fallback[0]][fallback[1]] = ' '
            candidates = [fallback]
        else:
            farthest = max(all_open, key=lambda p: cheb(p, start))
            candidates = [farthest]

    er, ec = rng.choice(candidates)
    grid[er][ec] = 'E'
    return grid

def generate_open_layout_mazeish(
    n: int,
    rng: random.Random,
    min_exit_frac: float = 0.8,
    extra_openings: int = None,
    room_attempts: int = None,
    room_radius_range: Tuple[int,int] = (1, 2),
    min_exit_blocks: int = 0,
) -> List[List[str]]:
    """
    Maze-first, then open up: 1-thick walls that naturally form ─, L, and T shapes,
    with added loops and small rooms so it's not only tight hallways.

    Legend: ' ' open, '#' wall, 'S' start (left border [1,0]), 'E' exit (≥ min_exit_frac across).
    """
    if n < 7:
        raise ValueError("mazeish generation works best with n >= 7")

    # 1) Start with all walls
    grid = [['#' for _ in range(n)] for _ in range(n)]

    # 2) Make maze cells at odd coordinates open
    for r in range(1, n-1, 2):
        for c in range(1, n-1, 2):
            grid[r][c] = ' '

    # 3) Randomized DFS/Backtracker over odd cells; carve 1-thick walls between cells
    def neighbors_odd(rr, cc):
        for dr, dc in ((-2,0),(2,0),(0,-2),(0,2)):
            nr, nc = rr+dr, cc+dc
            if 1 <= nr < n-1 and 1 <= nc < n-1:
                yield nr, nc

    start_cell = (1, 1)
    stack = [start_cell]
    seen = {start_cell}
    while stack:
        r, c = stack[-1]
        nbrs = [(nr, nc) for (nr, nc) in neighbors_odd(r, c) if (nr, nc) not in seen]
        if nbrs:
            nr, nc = rng.choice(nbrs)
            # carve wall between (r,c) and (nr,nc)
            wr, wc = (r + nr)//2, (c + nc)//2
            grid[wr][wc] = ' '
            seen.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # 4) Add extra random openings to create loops and T/L junctions
    if extra_openings is None:
        extra_openings = max(2, n)        # heuristic
    for _ in range(extra_openings):
        rr = rng.randrange(1, n-1)
        cc = rng.randrange(1, n-1)
        # Only break walls that touch at least two opens (to bias T/L shapes)
        if grid[rr][cc] == '#':
            open_count = 0
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                r2, c2 = rr+dr, cc+dc
                if 1 <= r2 < n-1 and 1 <= c2 < n-1 and grid[r2][c2] == ' ':
                    open_count += 1
            if open_count >= 1 and rng.random() < 0.9:
                grid[rr][cc] = ' '

    # 5) Carve a few small "rooms" (square/Chebyshev blobs) to soften the maze
    if room_attempts is None:
        room_attempts = max(1, n//2)
    rmin, rmax = room_radius_range
    rmin = max(1, rmin); rmax = max(rmin, rmax)
    for _ in range(room_attempts):
        rr = rng.randrange(2, n-2)
        cc = rng.randrange(2, n-2)
        rad = rng.randint(rmin, rmax)
        for ar in range(rr-rad, rr+rad+1):
            for ac in range(cc-rad, cc+rad+1):
                if 1 <= ar < n-1 and 1 <= ac < n-1 and max(abs(ar-rr), abs(ac-cc)) <= rad:
                    grid[ar][ac] = ' '

    # 6) Start + Exit
    grid[1][0] = 'S'      # left border
    grid[1][1] = ' '      # ensure doorway into maze

    # place exit far from start
    start = (1, 0)
    def cheb(a,b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    thresh = max(
        1,
        math.ceil(max(min_exit_frac * (n - 2), float(min_exit_blocks)))
    )

    candidates = [(r,c) for r in range(1,n-1) for c in range(1,n-1)
                  if grid[r][c] == ' ' and cheb((r,c), start) >= thresh]
    if not candidates:
        # fallback: farthest open tile
        opens = [(r,c) for r in range(1,n-1) for c in range(1,n-1) if grid[r][c] == ' ']
        if not opens:
            opens = [(n-2, n-2)]
            grid[n-2][n-2] = ' '
        candidates = [max(opens, key=lambda p: cheb(p, start))]

    er, ec = rng.choice(candidates)
    grid[er][ec] = 'E'
    return grid


# ----------------- Helpers -----------------
def mc_pos(origin_x: int, origin_y: int, origin_z: int, r: int, c: int) -> Tuple[int,int,int]:
    return origin_x + c, origin_y, origin_z + r  # (x,y,z)

def find_start(grid: List[List[str]]) -> Tuple[int,int]:
    n = len(grid)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'S':
                return (r, c)
    return (1, 0)

def collect_open_tiles(grid: List[List[str]]) -> List[Tuple[int,int]]:
    n = len(grid)
    return [(r, c) for r in range(1, n-1) for c in range(1, n-1) if grid[r][c] == ' ']

TILE = 181  # 181*181 = 32761 < 32768

def tile_fill_plane(lines, x0, z0, x1, z1, y, block_id):
    xi, xj = sorted((x0, x1))
    zi, zj = sorted((z0, z1))
    x = xi
    while x <= xj:
        xr = min(xj, x + TILE - 1)
        z = zi
        while z <= zj:
            zr = min(zj, z + TILE - 1)
            lines.append(f"fill {x} {y} {z} {xr} {y} {zr} {block_id}")
            z = zr + 1
        x = xr + 1

def tile_clear_volume(lines, x0, z0, x1, z1, y_lo, y_hi, air_id="minecraft:air"):
    # Clear per layer so each fill is still ≤ 181×181
    for y in range(min(y_lo, y_hi), max(y_lo, y_hi) + 1):
        tile_fill_plane(lines, x0, z0, x1, z1, y, air_id)

# ----------------- Build function (no Warden summons here) -----------------
def write_build_function(
    grid: List[List[str]],
    origin_x: int, origin_y: int, origin_z: int,
    out_path: str,
    *,
    walk_height: int = 2,           # interior air height
    floor_axis: str = "x",          # "x" or "z"
    namespace: str = "build",       # used by the exit command block to call finish_plate
) -> Tuple[int,int,int]:
    """
    Writes build_layout.mcfunction and returns recommended TP coords (x,y,z) for start (feet level).
    Places a command block at the exit with a pressure plate on top that runs <namespace>:finish_plate.
    """
    n = len(grid)
    OAK_FLOOR = f"minecraft:stripped_oak_log[axis={floor_axis}]"
    BIRCH_WALL = "minecraft:stripped_birch_log"
    SEA_LANTERN = "minecraft:sea_lantern"
    AIR = "minecraft:air"

    # Vertical layout
    y_floor = origin_y
    y_clear_lo = origin_y + 1
    y_clear_hi = origin_y + walk_height
    y_ceiling = origin_y + walk_height + 1

    x0, z0 = origin_x, origin_z
    x1, z1 = origin_x + n - 1, origin_z + n - 1

    lines = []
    lines.append(f"# Build clustered layout at ({origin_x}, {origin_y}, {origin_z}), size {n}x{n}")
    lines.append(f"# Walk height: {walk_height} (air {y_clear_lo}..{y_clear_hi}); Ceiling at {y_ceiling}")
    lines.append("# Floor: stripped oak; Perimeter & clusters: stripped birch; Ceiling: sea lanterns")
    lines.append("")

    # Floor (tiled)
    tile_fill_plane(lines, x0, z0, x1, z1, y_floor, OAK_FLOOR)

    # Ceiling (tiled)
    tile_fill_plane(lines, x0, z0, x1, z1, y_ceiling, SEA_LANTERN)

    # Clear walk volume (tiled per layer)
    tile_clear_volume(lines, x0, z0, x1, z1, y_clear_lo, y_clear_hi, AIR)


    def column(x, z, y1, y2, block_id):
        lines.append(f"fill {x} {y1} {z} {x} {y2} {z} {block_id}")

    # Perimeter
    for c in range(n):
        x, _, zt = mc_pos(origin_x, origin_y, origin_z, 0, c)
        x, _, zb = mc_pos(origin_x, origin_y, origin_z, n-1, c)
        column(x, zt, y_clear_lo, y_clear_hi, BIRCH_WALL)
        column(x, zb, y_clear_lo, y_clear_hi, BIRCH_WALL)
    for r in range(1, n - 1):
        xl, _, z = mc_pos(origin_x, origin_y, origin_z, r, 0)
        xr, _, z = mc_pos(origin_x, origin_y, origin_z, r, n - 1)
        column(xl, z, y_clear_lo, y_clear_hi, BIRCH_WALL)
        column(xr, z, y_clear_lo, y_clear_hi, BIRCH_WALL)

    # Interior clusters
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if grid[r][c] == '#':
                x, _, z = mc_pos(origin_x, origin_y, origin_z, r, c)
                column(x, z, y_clear_lo, y_clear_hi, BIRCH_WALL)

    # Start doorway (ensure walk space open)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'S':
                x, _, z = mc_pos(origin_x, origin_y, origin_z, r, c)
                lines.append(f"fill {x} {y_clear_lo} {z} {x} {y_clear_hi} {z} {AIR}")

    # Exit: place command block + pressure plate (no hole)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'E':
                ex, _, ez = mc_pos(origin_x, origin_y, origin_z, r, c)
                # clear headroom
                lines.append(f"fill {ex} {y_clear_lo} {ez} {ex} {y_clear_hi} {ez} {AIR}")
                # command block (Needs Redstone); safer to set command with data modify
                lines.append(f"setblock {ex} {y_floor} {ez} minecraft:command_block[facing=up]{{auto:0b}}")
                lines.append(
                    f'data modify block {ex} {y_floor} {ez} Command set value '
                    f'"execute as @p run function {namespace}:finish_plate"'
                )
                # pressure plate on top to trigger it
                lines.append(f"setblock {ex} {y_clear_lo} {ez} minecraft:stone_pressure_plate")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # TP spot (one block inside from the left-wall start)
    sr, sc = find_start(grid)
    sxx, _, szz = mc_pos(origin_x, origin_y, origin_z, sr, sc)
    tx = sxx + (1 if sc == 0 else 0)
    tz = szz + (1 if sr == 0 else 0)
    ty = y_clear_lo
    return (tx, ty, tz)

# ----------------- TP function -----------------
def write_tp_function(tp_path: str, tp_selector: str, tp_coords: Tuple[int,int,int]) -> None:
    tx, ty, tz = tp_coords
    lines = [
        "# Teleport players to the start point (safe carve + TP)",
        f"fill {tx} {ty} {tz} {tx} {ty+1} {tz} minecraft:air",
        f"tp {tp_selector} {tx} {ty} {tz}",
    ]
    with open(tp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------- CLEAR function -----------------
def write_clear_function(
    out_path: str,
    origin_x: int, origin_y: int, origin_z: int,
    n: int,
    clear_height: int
) -> None:
    if clear_height < 1:
        raise ValueError("clear_height must be >= 1")

    x0, y0, z0 = origin_x, origin_y, origin_z
    x1, y1, z1 = origin_x + n - 1, origin_y + clear_height - 1, origin_z + n - 1
    AIR = "minecraft:air"
    MAX_PER = 32768

    lines = [f"# Clear n={n} x n={n} x h={clear_height} at origin ({origin_x}, {origin_y}, {origin_z})"]
    total_vol = n * n * clear_height

    if total_vol <= MAX_PER:
        lines.append(f"fill {x0} {y0} {z0} {x1} {y1} {z1} {AIR}")
    else:
        TILE = 181  # 181*181 = 32761 < 32768
        for y in range(y0, y1 + 1):
            xi = x0
            while xi <= x1:
                xj = min(x1, xi + TILE - 1)
                zi = z0
                while zi <= z1:
                    zj = min(z1, zi + TILE - 1)
                    lines.append(f"fill {xi} {y} {zi} {xj} {y} {zj} {AIR}")
                    zi = zj + 1
                xi = xj + 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------- TIMER functions -----------------
def write_timer_functions(
    namespace: str,
    start_path: str,
    tick_path: str,
    finish_plate_path: str,
    tp_selector: str,
    tp_coords: Tuple[int,int,int],
    warden_world_positions: List[Tuple[int,int,int]],
    warden_persist: bool,
    arena_box: Tuple[int,int,int,int,int,int],   # <— NEW
) -> None:
    sx, sy, sz = tp_coords
    nbt = "{PersistenceRequired:1b}" if warden_persist else ""

    # start_run: TP players, tag runners, zero time, schedule ticking, SUMMON WARDENS
    # start_lines = [
    #     "# Start a timed run: TP players, tag as runners, summon Wardens, start ticking",
    #     "scoreboard objectives add runTime dummy",
    #     f"tag {tp_selector} add runner",
    #     f"scoreboard players set {tp_selector} runTime 0",
    #     f"fill {sx} {sy} {sz} {sx} {sy+1} {sz} minecraft:air",
    #     f"tp {tp_selector} {sx} {sy} {sz}",
    #     f"kill @e[type=minecraft:warden]",
    # ]
    start_lines = [
            "# Start a timed run: rebuild arena, TP players, tag as runners, summon Wardens, start ticking",
            # Rebuild the arena first
            f"function {namespace}:clear_area",
            f"function {namespace}:build_layout",

            # Reset/kick old wardens just in case
            "kill @e[type=minecraft:warden]",

            # Timer setup + TP
            "scoreboard objectives add runTime dummy",
            f"tag {tp_selector} add runner",
            f"scoreboard players set {tp_selector} runTime 0",
            ]
    for i in range(5):
        for j in range(3):
            for k in range(3):
                start_lines += [
                            f"fill {sx} {sy} {sz} {sx+i} {sy+j} {sz+k} minecraft:air",
                        ]
    start_lines += [
            "kill @e[type=minecraft:item]",
            f"tp {tp_selector} {sx} {sy} {sz}",
        ]

    # Summon Wardens now (ensure spawn spots are clear)
    for (wx, wy, wz) in warden_world_positions:
        start_lines.append(f"setblock {wx} {wy} {wz} minecraft:air")
        start_lines.append(f"summon minecraft:warden {wx} {wy} {wz} {nbt}".rstrip())
    # Start ticking
    start_lines.append(f"schedule function {namespace}:timer_tick 1t replace")

    with open(start_path, "w", encoding="utf-8") as f:
        f.write("\n".join(start_lines))

    # timer_tick: +1 tick for each runner; reschedule while any remain
    tick_lines = [
        "# Increment timer for all runners each tick",
        "scoreboard objectives add runTime dummy",
        "scoreboard players add @a[tag=runner] runTime 1",
        f"execute if entity @a[tag=runner] run schedule function {namespace}:timer_tick 1t replace",
        'title @a[tag=runner] actionbar {"text":"Time: "}',
        'title @a[tag=runner] actionbar [{"text":"Time: "},{"score":{"name":"*","objective":"runTime"}},{"text":" ticks"}]',

    ]
    with open(tick_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tick_lines))

    # finish_plate: called by the command block at exit
    x0, y0, z0, x1, y1, z1 = arena_box

    finish_lines = [
        "# Finish triggered by exit plate",
        "scoreboard objectives add runTime dummy",
        'tellraw @s [{"text":"Finished! Time: ","color":"yellow"},'
        '{"score":{"name":"*","objective":"runTime"}},{"text":" ticks (~","color":"gray"},'
        '{"text":"seconds = ticks/20","color":"gray"},{"text":")"}]',
        # stop timer and wipe runners
        "tag @a remove runner",
        "scoreboard players reset @a runTime",
        # remove Wardens
        "kill @e[type=minecraft:warden]",
        # remove dropped sculk catalyst items
        'kill @e[type=minecraft:item,nbt={Item:{id:"minecraft:sculk_catalyst"}}]',
        # scrub placed sculk family blocks in the arena prism
        f"fill {x0} {y0} {z0} {x1} {y1} {z1} air replace minecraft:sculk",
        f"fill {x0} {y0} {z0} {x1} {y1} {z1} air replace minecraft:sculk_vein",
        f"fill {x0} {y0} {z0} {x1} {y1} {z1} air replace minecraft:sculk_catalyst",
        f"fill {x0} {y0} {z0} {x1} {y1} {z1} air replace minecraft:sculk_sensor",
        f"fill {x0} {y0} {z0} {x1} {y1} {z1} air replace minecraft:sculk_shrieker",
    ]

    with open(finish_plate_path, "w", encoding="utf-8") as f:
        f.write("\n".join(finish_lines))

def print_grid(grid: List[List[str]]) -> None:
    for row in grid:
        print(''.join(row))

# ----------------- CLI & main -----------------
def main():
    p = argparse.ArgumentParser(description="Emit build, tp, clear, and timer mcfunctions with a plate-finish exit; Wardens spawn on start_run.")
    p.add_argument("-n", "--size", type=int, required=True, help="grid size n (n x n)")
    p.add_argument("--gen", choices=["clustered", "mazeish"], default="clustered",
               help="which layout generator to use")
    p.add_argument("--min-exit-frac", type=float, default=1.0,
               help="minimum exit distance as a fraction of interior span (Chebyshev)")
    p.add_argument("--min-exit-blocks", type=int, default=0,
               help="absolute minimum Chebyshev distance (in blocks) between S and E")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--density", type=float, default=0.12, help="cluster coverage (approx)")
    p.add_argument("--cluster-radius", type=int, default=1, help="blob radius (0=1x1, 1≈3x3, 2≈5x5...)")
    p.add_argument("--walk-steps", type=int, default=1, help="stamping steps per seed (default 1)")
    p.add_argument("--origin", type=int, nargs=3, metavar=("X","Y","Z"), default=(100, 200, 100),
                   help="world origin (x y z) — default 100 200 100")
    p.add_argument("--floor-axis", choices=["x", "z"], default="x",
                   help="floorboard direction (stripped oak axis)")
    p.add_argument("--walk-height", type=int, default=2,
                   help="interior walkable height; ceiling at Y+walk_height+1")

    # Wardens (spawned in start_run)
    p.add_argument("--wardens", type=int, default=0, help="number of random Warden spawns (on start)")
    p.add_argument("--warden-persist", action="store_true", help="Wardens get {PersistenceRequired:1b}")

    # Outputs & namespace
    p.add_argument("--namespace", type=str, default="build",
                   help="datapack namespace (also used by the exit command block)")
    p.add_argument("--out-build", type=str, default="build_layout.mcfunction",
                   help="output mcfunction for building the arena")
    p.add_argument("--out-tp", type=str, default="tp_start.mcfunction",
                   help="output mcfunction for teleporting to start")
    p.add_argument("--out-clear", type=str, default="clear_area.mcfunction",
                   help="output mcfunction for clearing the n×n×height volume")
    p.add_argument("--out-start-run", type=str, default="start_run.mcfunction",
                   help="output mcfunction that teleports, tags runners, summons Wardens, and starts the timer")
    p.add_argument("--out-timer-tick", type=str, default="timer_tick.mcfunction",
                   help="output mcfunction that increments run time and keeps ticking")
    p.add_argument("--out-finish-plate", type=str, default="finish_plate.mcfunction",
                   help="output mcfunction called by the exit plate to stop timer & kill wardens")

    # clear height & selectors
    p.add_argument("--clear-height", type=int, default=8,
                   help="height to clear upward from origin Y (>=1)")
    p.add_argument("--tp-selector", type=str, default="@a",
                   help="who to teleport in tp_start/start_run (e.g., @a or @p)")
    args = p.parse_args()

    if args.size < 7:
        raise SystemExit("Choose n >= 7 for nicer clusters with borders.")
    if not (0.0 <= args.density <= 0.9):
        raise SystemExit("density must be between 0.0 and 0.9")
    if args.cluster_radius < 0:
        raise SystemExit("cluster-radius must be >= 0")
    if args.walk_height < 1:
        raise SystemExit("--walk-height must be >= 1")
    if args.clear_height < 1:
        raise SystemExit("--clear-height must be >= 1")

    rng = random.Random(args.seed)

    if args.gen == "mazeish":
        grid = generate_open_layout_mazeish(
                args.size, rng,
                min_exit_frac=args.min_exit_frac,
                extra_openings=2,
                room_attempts=2800,
                min_exit_blocks=args.min_exit_blocks,

                )
    else:
        grid = generate_open_layout_clustered(
            args.size, rng,
            density=args.density,
            cluster_radius=args.cluster_radius,
            walk_steps=args.walk_steps,
            min_exit_blocks=args.min_exit_blocks,
            )

    print_grid(grid)

    ox, oy, oz = args.origin
    # Build function (no Wardens spawned here)
    tp_coords = write_build_function(
        grid, ox, oy, oz, args.out_build,
        walk_height=args.walk_height,
        floor_axis=args.floor_axis,
        namespace=args.namespace,
    )
    # TP helper
    write_tp_function(args.out_tp, args.tp_selector, tp_coords)
    # CLEAR helper
    write_clear_function(args.out_clear, ox, oy, oz, args.size, args.clear_height)

    # Precompute Warden world positions for start_run
    warden_world_positions: List[Tuple[int,int,int]] = []
    if args.wardens > 0:
        # collect candidate tiles and pick some
        candidates = collect_open_tiles(grid)
        rng.shuffle(candidates)
        picks = candidates[:min(args.wardens, len(candidates))]
        # Use walk layer y for spawns
        wy = oy + 1
        for (rr, cc) in picks:
            wx, _, wz = mc_pos(ox, oy, oz, rr, cc)
            warden_world_positions.append((wx, wy, wz))

    x0, y0, z0 = ox, oy, oz
    x1, y1, z1 = ox + args.size - 1, oy + args.clear_height - 1, oz + args.size - 1
    arena_box = (x0, y0, z0, x1, y1, z1)

    # Timer-related functions (start_run summons Wardens)
    write_timer_functions(
        args.namespace,
        args.out_start_run,
        args.out_timer_tick,
        args.out_finish_plate,
        args.tp_selector,
        tp_coords,
        warden_world_positions,
        args.warden_persist,
        arena_box,
    )


    print(f"Wrote: {args.out_build}, {args.out_tp}, {args.out_clear}, {args.out_start_run}, {args.out_timer_tick}, {args.out_finish_plate}")
    print("Place them in data/<namespace>/functions/, then:")
    print(f"  /function {args.namespace}:build_layout   # builds arena (no Wardens)")
    print(f"  /function {args.namespace}:tp_start       # teleports to start")
    print(f"  /function {args.namespace}:clear_area     # wipes the n×n×H area")
    print(f"  /function {args.namespace}:start_run      # TP + summon Wardens + start timer")
    print("Step on the exit pressure plate to finish, stop timer, and kill Wardens.")

if __name__ == "__main__":
    main()

