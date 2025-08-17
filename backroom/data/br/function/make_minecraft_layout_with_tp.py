#!/usr/bin/env python3
# make_minecraft_layout_with_tp.py
# Adds a 'mazeish-wide' generator: wider halls + user-selected halls (rows/cols/segments),
# keeps your build/clear/timer/warden features.

import argparse
import random
import math
from typing import List, Tuple

def carve_segmented_row(grid, r_center: int, band_width: int,
                        seg_len_rng: Tuple[int,int], gap_len_rng: Tuple[int,int],
                        rng: random.Random) -> None:
    """Open a horizontal band in segments (open...gap...open...), inside the interior."""
    n = len(grid)
    if n < 3: return
    k = max(0, (band_width - 1)//2)
    c = 1
    while c < n-1:
        seg = rng.randint(seg_len_rng[0], seg_len_rng[1])
        for cc in range(c, min(n-1, c+seg)):
            for rr in range(max(1, r_center-k), min(n-1, r_center+k+1)):
                grid[rr][cc] = ' '
        c += seg
        gap = rng.randint(gap_len_rng[0], gap_len_rng[1])
        c += gap

def carve_segmented_col(grid, c_center: int, band_width: int,
                        seg_len_rng: Tuple[int,int], gap_len_rng: Tuple[int,int],
                        rng: random.Random) -> None:
    """Open a vertical band in segments (open...gap...open...), inside the interior."""
    n = len(grid)
    if n < 3: return
    k = max(0, (band_width - 1)//2)
    r = 1
    while r < n-1:
        seg = rng.randint(seg_len_rng[0], seg_len_rng[1])
        for rr in range(r, min(n-1, r+seg)):
            for cc in range(max(1, c_center-k), min(n-1, c_center+k+1)):
                grid[rr][cc] = ' '
        r += seg
        gap = rng.randint(gap_len_rng[0], gap_len_rng[1])
        r += gap


def _count_open_neighbors4(grid, r, c):
    n = len(grid)
    cnt = 0
    if r > 1 and grid[r-1][c] == ' ': cnt += 1
    if r < n-2 and grid[r+1][c] == ' ': cnt += 1
    if c > 1 and grid[r][c-1] == ' ': cnt += 1
    if c < n-2 and grid[r][c+1] == ' ': cnt += 1
    return cnt

def erode_walls(grid, rng, prob: float = 0.08, min_open_neighbors: int = 2):
    """Randomly delete interior wall cells that already border open space, to connect areas."""
    if prob <= 0: return
    n = len(grid)
    to_open = []
    for r in range(1, n-1):
        for c in range(1, n-1):
            if grid[r][c] == '#':
                if _count_open_neighbors4(grid, r, c) >= min_open_neighbors and rng.random() < prob:
                    to_open.append((r, c))
    for r, c in to_open:
        grid[r][c] = ' '

def carve_random_rooms(grid, rng, attempts: int, radius_range: Tuple[int,int]):
    """Carve Chebyshev blobs (square-radius rooms) inside the maze."""
    if attempts <= 0: return
    n = len(grid)
    rmin, rmax = radius_range
    rmin = max(1, rmin); rmax = max(rmin, rmax)
    for _ in range(attempts):
        rr = rng.randrange(2, n-2)
        cc = rng.randrange(2, n-2)
        rad = rng.randint(rmin, rmax)
        for ar in range(rr-rad, rr+rad+1):
            for ac in range(cc-rad, cc+rad+1):
                if 1 <= ar < n-1 and 1 <= ac < n-1 and max(abs(ar-rr), abs(ac-cc)) <= rad:
                    grid[ar][ac] = ' '

def punch_extra_doors(grid, rng, hall_width: int, count: int):
    """
    Punch extra doorways of width=hall_width through 1-thick walls.
    We look for wall cells with open on both sides (horizontal *or* vertical seams).
    """
    if count <= 0: return
    n = len(grid)
    k = max(0, (hall_width - 1) // 2)

    # Collect candidate seam cells
    vertical_seams = []   # open-left / wall / open-right → open a vertical slit
    horizontal_seams = [] # open-up   / wall / open-down  → open a horizontal slit

    for r in range(1, n-1):
        for c in range(1, n-1):
            if grid[r][c] != '#':
                continue
            if grid[r][c-1] == ' ' and grid[r][c+1] == ' ':
                vertical_seams.append((r, c))
            if grid[r-1][c] == ' ' and grid[r+1][c] == ' ':
                horizontal_seams.append((r, c))

    rng.shuffle(vertical_seams)
    rng.shuffle(horizontal_seams)
    vi = hi = 0

    for _ in range(count):
        # Alternate orientations if possible
        use_vert = (rng.random() < 0.5)
        if use_vert and vi < len(vertical_seams):
            r, c = vertical_seams[vi]; vi += 1
            r0, r1 = max(1, r-k), min(n-2, r+k)
            for rr in range(r0, r1+1):
                grid[rr][c] = ' '
        elif hi < len(horizontal_seams):
            r, c = horizontal_seams[hi]; hi += 1
            c0, c1 = max(1, c-k), min(n-2, c+k)
            for cc in range(c0, c1+1):
                grid[r][cc] = ' '
        elif vi < len(vertical_seams):
            r, c = vertical_seams[vi]; vi += 1
            r0, r1 = max(1, r-k)


# ----------------- Helpers for bounds & carving -----------------
def in_bounds(n: int, r: int, c: int, margin: int = 1) -> bool:
    return margin <= r < n - margin and margin <= c < n - margin

def clamp(v, lo, hi): return max(lo, min(hi, v))

def parse_csv_ints(s: str) -> List[int]:
    if not s: return []
    out = []
    for part in s.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out

def parse_segments(seg_list: List[str]) -> List[Tuple[int,int,int,int]]:
    out = []
    for s in seg_list:
        parts = [p for p in s.replace(' ', '').split(',') if p != '']
        if len(parts) != 4:
            continue
        r0, c0, r1, c1 = map(int, parts)
        if r0 == r1 or c0 == c1:  # axis-aligned only
            out.append((r0, c0, r1, c1))
    return out

def carve_row_band(grid: List[List[str]], r: int, width: int) -> None:
    n = len(grid)
    k = (width - 1) // 2
    for rr in range(r - k, r + k + 1):
        if 1 <= rr < n - 1:
            for c in range(1, n - 1):
                grid[rr][c] = ' '

def carve_col_band(grid: List[List[str]], c: int, width: int) -> None:
    n = len(grid)
    k = (width - 1) // 2
    for cc in range(c - k, c + k + 1):
        if 1 <= cc < n - 1:
            for r in range(1, n - 1):
                grid[r][cc] = ' '

def carve_segment_band(grid: List[List[str]], r0: int, c0: int, r1: int, c1: int, width: int) -> None:
    n = len(grid)
    k = (width - 1) // 2
    r0, r1 = sorted((r0, r1))
    c0, c1 = sorted((c0, c1))
    r0 = clamp(r0, 1, n - 2); r1 = clamp(r1, 1, n - 2)
    c0 = clamp(c0, 1, n - 2); c1 = clamp(c1, 1, n - 2)
    if r0 == r1:
        rr = r0
        for c in range(c0, c1 + 1):
            for ar in range(rr - k, rr + k + 1):
                if 1 <= ar < n - 1:
                    grid[ar][c] = ' '
    elif c0 == c1:
        cc = c0
        for r in range(r0, r1 + 1):
            for ac in range(cc - k, cc + k + 1):
                if 1 <= ac < n - 1:
                    grid[r][ac] = ' '

def widen_all_open_cells(grid: List[List[str]], width: int) -> None:
    """Chebyshev-dilate every open cell by radius k to get ~width-wide halls."""
    if width <= 1: return
    n = len(grid)
    k = (width - 1) // 2
    to_open = []
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if grid[r][c] == ' ':
                for rr in range(r - k, r + k + 1):
                    for cc in range(c - k, c + k + 1):
                        if 1 <= rr < n - 1 and 1 <= cc < n - 1 and max(abs(rr - r), abs(cc - c)) <= k:
                            to_open.append((rr, cc))
    for rr, cc in to_open:
        grid[rr][cc] = ' '

# ----------------- Layout generation (clustered obstacles) -----------------
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
    start = (1, 0)
    grid[start[0]][start[1]] = 'S'

    interior_area = (n - 2) * (n - 2)
    avg_blob = (2 * cluster_radius + 1) ** 2
    target = max(1, int(density * interior_area))
    seeds = max(1, int(target / max(1, avg_blob // 2)))

    for _ in range(seeds):
        r = rng.randint(0, n - 2)
        c = rng.randint(0, n - 2)
        for _ in range(walk_steps):
            for rr in range(r - cluster_radius, r + cluster_radius + 1):
                for cc in range(c - cluster_radius, c + cluster_radius + 1):
                    if 1 <= rr < n - 1 and 1 <= cc < n - 1 and max(abs(rr - r), abs(cc - c)) <= cluster_radius:
                        grid[rr][cc] = '#'
            r = max(1, min(n - 2, r + rng.choice([-1, 0, 1])))
            c = max(1, min(n - 2, c + rng.choice([-1, 0, 1])))

    grid[start[0]][start[1]] = 'S'

    def cheb(a, b): return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    dist_thresh = max(1, math.ceil(max(min_exit_frac * (n - 2), float(min_exit_blocks))))

    candidates = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                  if grid[r][c] == ' ' and cheb((r, c), start) >= dist_thresh]

    if not candidates:
        all_open = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1) if grid[r][c] == ' ']
        if not all_open:
            fallback = (n - 2, n - 2)
            grid[fallback[0]][fallback[1]] = ' '
            candidates = [fallback]
        else:
            farthest = max(all_open, key=lambda p: cheb(p, start))
            candidates = [farthest]

    er, ec = rng.choice(candidates)
    grid[er][ec] = 'E'
    return grid

def generate_open_layout_mazeish_halls(
    n: int,
    rng: random.Random,
    *,
    hall_width: int = 3,                # base maze corridor width (true wide maze)
    # segmented macro-hall controls:
    hall_band_width: int = None,        # width of the macro bands (defaults to hall_width)
    row_step: int = None,               # how often to place a horizontal band
    col_step: int = None,               # how often to place a vertical band
    row_offset: int = 2,                # starting offset for horizontal bands (grid rows)
    col_offset: int = 2,                # starting offset for vertical bands (grid cols)
    seg_len: Tuple[int,int] = (6, 12),  # open segment length range
    gap_len: Tuple[int,int] = (3, 7),   # gap length range between segments
    # exit distance rules:
    min_exit_frac: float = 0.8,
    min_exit_blocks: int = 0,
) -> List[List[str]]:
    """
    Builds a wide-corridor maze, then deletes rows/cols of walls in segmented strips
    (so they don't span the whole map), yielding a big connected hall grid overlaying
    smaller maze pockets.
    """
    # Base: true wide maze (rooms of hall_width separated by 1-thick walls)
    grid = generate_open_layout_mazeish_wide(
        n, rng,
        hall_width=hall_width,
        widen_all=True,
        rows_to_carve=[],
        cols_to_carve=[],
        segments_to_carve=[],
        min_exit_frac=min_exit_frac,
        min_exit_blocks=min_exit_blocks,
        extra_doors=0, erode_pct=0.0, room_attempts=0
    )

    # Macro-band parameters
    Wb = hall_band_width if hall_band_width and hall_band_width > 0 else hall_width
    # Reasonable default spacing: about one band every (hall_width + 3)
    rs = row_step if row_step and row_step > 0 else (hall_width + 3)
    cs = col_step if col_step and col_step > 0 else (hall_width + 3)

    # Horizontal segmented halls
    for r_center in range(1 + row_offset, n-1, rs):
        carve_segmented_row(grid, r_center, Wb, seg_len, gap_len, rng)

    # Vertical segmented halls
    for c_center in range(1 + col_offset, n-1, cs):
        carve_segmented_col(grid, c_center, Wb, seg_len, gap_len, rng)

    # Re-stamp start in case a band overwrote marker
    grid[1][0] = 'S'
    if 1 <= 1 < n-1 and 1 <= 1 < n-1:
        grid[1][1] = ' '

    # Ensure exit is far enough and on open
    # (wipe any existing 'E' and re-place according to distance rule)
    for r in range(1, n-1):
        for c in range(1, n-1):
            if grid[r][c] == 'E':
                grid[r][c] = ' '

    start = (1, 0)
    def cheb(a,b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    thresh = max(1, math.ceil(max(min_exit_frac * (n - 2), float(min_exit_blocks))))
    candidates = [(r, c) for r in range(1, n-1) for c in range(1, n-1)
                  if grid[r][c] == ' ' and cheb((r, c), start) >= thresh]
    if not candidates:
        opens = [(r, c) for r in range(1, n-1) for c in range(1, n-1) if grid[r][c] == ' ']
        if not opens:
            opens = [(n - 2, n - 2)]; grid[n - 2][n - 2] = ' '
        candidates = [max(opens, key=lambda p: cheb(p, start))]
    er, ec = rng.choice(candidates)
    grid[er][ec] = 'E'
    return grid


# ----------------- Maze-ish (1-thick) -----------------
def generate_open_layout_mazeish(
    n: int,
    rng: random.Random,
    min_exit_frac: float = 0.8,
    extra_openings: int = None,
    room_attempts: int = None,
    room_radius_range: Tuple[int,int] = (1, 2),
    min_exit_blocks: int = 0,
) -> List[List[str]]:
    if n < 7:
        raise ValueError("mazeish generation works best with n >= 7")

    grid = [['#' for _ in range(n)] for _ in range(n)]
    for r in range(1, n-1, 2):
        for c in range(1, n-1, 2):
            grid[r][c] = ' '

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
            wr, wc = (r + nr)//2, (c + nc)//2
            grid[wr][wc] = ' '
            seen.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    if extra_openings is None:
        extra_openings = max(2, n)
    for _ in range(extra_openings):
        rr = rng.randrange(1, n-1)
        cc = rng.randrange(1, n-1)
        if grid[rr][cc] == '#':
            open_count = 0
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                r2, c2 = rr+dr, cc+dc
                if 1 <= r2 < n-1 and 1 <= c2 < n-1 and grid[r2][c2] == ' ':
                    open_count += 1
            if open_count >= 1 and rng.random() < 0.9:
                grid[rr][cc] = ' '

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

    grid[1][0] = 'S'
    grid[1][1] = ' '

    start = (1, 0)
    def cheb(a,b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    thresh = max(1, math.ceil(max(min_exit_frac * (n - 2), float(min_exit_blocks))))
    candidates = [(r,c) for r in range(1,n-1) for c in range(1,n-1)
                  if grid[r][c] == ' ' and cheb((r,c), start) >= thresh]
    if not candidates:
        opens = [(r,c) for r in range(1,n-1) for c in range(1,n-1) if grid[r][c] == ' ']
        if not opens:
            opens = [(n-2, n-2)]; grid[n-2][n-2] = ' '
        candidates = [max(opens, key=lambda p: cheb(p, start))]
    er, ec = random.choice(candidates)
    grid[er][ec] = 'E'
    return grid

# ----------------- NEW: Maze-ish (WIDE) with user-selected halls -----------------
def generate_open_layout_mazeish_wide(
    n: int,
    rng: random.Random,
    hall_width: int = 3,
    widen_all: bool = True,  # kept for CLI compat; not used (already wide)
    rows_to_carve: List[int] = None,
    cols_to_carve: List[int] = None,
    segments_to_carve: List[Tuple[int,int,int,int]] = None,
    min_exit_frac: float = 0.8,
    min_exit_blocks: int = 0,
    # NEW open-up knobs:
    extra_doors: int = 0,
    erode_pct: float = 0.0,
    room_attempts: int = 0,
    room_radius_range: Tuple[int,int] = (2, 4),
) -> List[List[str]]:
    """
    Wide-corridor maze on a coarse grid (rooms of width=hall_width with 1-thick walls),
    then 'open up' by punching extra doors, eroding walls, and carving rooms.
    """
    # --- base wide maze (same as before) ---
    W = max(1, int(hall_width))
    Nint = n - 2
    P = W + 1  # room pitch (W) + 1-thick wall
    C = max(1, (Nint + 1) // P)

    grid = [['#' for _ in range(n)] for _ in range(n)]
    r0 = c0 = 1

    def room_top(ri, cj): return (r0 + ri * P, c0 + cj * P)

    for ri in range(C):
        rt = r0 + ri * P
        for cj in range(C):
            ct = c0 + cj * P
            for rr in range(rt, min(rt + W, n - 1)):
                for cc in range(ct, min(ct + W, n - 1)):
                    grid[rr][cc] = ' '

    def door_east(rt, ct):
        x_wall = ct + W
        if x_wall >= n - 1: return
        for rr in range(rt, min(rt + W, n - 1)):
            grid[rr][x_wall] = ' '

    def door_south(rt, ct):
        y_wall = rt + W
        if y_wall >= n - 1: return
        for cc in range(ct, min(ct + W, n - 1)):
            grid[y_wall][cc] = ' '

    def neighbors(ri, cj):
        for dri, dcj in ((-1,0),(1,0),(0,-1),(0,1)):
            nri, ncj = ri + dri, cj + dcj
            if 0 <= nri < C and 0 <= ncj < C:
                yield nri, ncj, dri, dcj

    start_cell = (0, 0)
    stack = [start_cell]
    seen = {start_cell}
    while stack:
        ri, cj = stack[-1]
        cand = [(nri, ncj, dri, dcj) for (nri, ncj, dri, dcj) in neighbors(ri, cj) if (nri, ncj) not in seen]
        if cand:
            nri, ncj, dri, dcj = rng.choice(cand)
            rt, ct = room_top(ri, cj)
            if dri == 0 and dcj == 1:        # east
                door_east(rt, ct)
            elif dri == 0 and dcj == -1:     # west
                nrt, nct = room_top(nri, ncj)
                door_east(nrt, nct)
            elif dri == 1 and dcj == 0:      # south
                door_south(rt, ct)
            else:                              # north
                nrt, nct = room_top(nri, ncj)
                door_south(nrt, nct)
            seen.add((nri, ncj))
            stack.append((nri, ncj))
        else:
            stack.pop()

    # A few loops for nicer connectivity
    for _ in range(max(0, C)):
        ri = rng.randrange(C); cj = rng.randrange(C)
        rt, ct = room_top(ri, cj)
        if rng.random() < 0.5 and cj + 1 < C:
            door_east(rt, ct)
        elif ri + 1 < C:
            door_south(rt, ct)

    # Forced user corridors (bands / segments)
    rows_to_carve = rows_to_carve or []
    cols_to_carve = cols_to_carve or []
    segments_to_carve = segments_to_carve or []
    for rr in rows_to_carve:
        carve_row_band(grid, rr, W)
    for cc in cols_to_carve:
        carve_col_band(grid, cc, W)
    for (rA, cA, rB, cB) in segments_to_carve:
        carve_segment_band(grid, rA, cA, rB, cB, W)

    # --- OPEN-UP PASSES ---
    punch_extra_doors(grid, rng, hall_width=W, count=max(0, int(extra_doors)))
    erode_walls(grid, rng, prob=max(0.0, float(erode_pct)), min_open_neighbors=2)
    carve_random_rooms(grid, rng, attempts=max(0, int(room_attempts)), radius_range=room_radius_range)

    # Start at left border aligned to first band
    start_row = r0
    grid[start_row][0] = ' '
    grid[start_row][0] = 'S'
    for rr in range(start_row, min(start_row + W, n - 1)):
        grid[rr][1] = ' '

    # Ensure exit far from start (re-place if an open-up pass removed it)
    start = (start_row, 0)
    def cheb(a,b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    thresh = max(1, math.ceil(max(min_exit_frac * (n - 2), float(min_exit_blocks))))
    # clear prior E
    for r in range(1, n-1):
        for c in range(1, n-1):
            if grid[r][c] == 'E': grid[r][c] = ' '
    candidates = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                  if grid[r][c] == ' ' and cheb((r, c), start) >= thresh]
    if not candidates:
        opens = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1) if grid[r][c] == ' ']
        if not opens:
            opens = [(n - 2, n - 2)]; grid[n - 2][n - 2] = ' '
        candidates = [max(opens, key=lambda p: cheb(p, start))]
    er, ec = rng.choice(candidates)
    grid[er][ec] = 'E'
    return grid



# ----------------- World coord helpers -----------------
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

# ---- tiling helpers to avoid /fill 32,768 block limit ----
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
    for y in range(min(y_lo, y_hi), max(y_lo, y_hi) + 1):
        tile_fill_plane(lines, x0, z0, x1, z1, y, air_id)

# ----------------- Build function (unchanged walls/floor/exit logic) -----------------
def write_build_function(
    grid: List[List[str]],
    origin_x: int, origin_y: int, origin_z: int,
    out_path: str,
    *,
    walk_height: int = 2,
    floor_axis: str = "x",
    namespace: str = "br",
) -> Tuple[int,int,int]:
    n = len(grid)
    OAK_FLOOR = f"minecraft:stripped_oak_log[axis={floor_axis}]"
    BIRCH_WALL = "minecraft:stripped_birch_log"
    SEA_LANTERN = "minecraft:sea_lantern"
    AIR = "minecraft:air"

    y_floor = origin_y
    y_clear_lo = origin_y + 1
    y_clear_hi = origin_y + walk_height
    y_ceiling = origin_y + walk_height + 1

    x0, z0 = origin_x, origin_z
    x1, z1 = origin_x + n - 1, origin_z + n - 1

    lines = []
    lines.append(f"# Build layout at ({origin_x}, {origin_y}, {origin_z}), size {n}x{n}")
    lines.append(f"# Walk height: {walk_height} (air {y_clear_lo}..{y_clear_hi}); Ceiling at {y_ceiling}")
    lines.append("# Floor: stripped oak; Walls: stripped birch; Ceiling: sea lanterns")
    lines.append("")
    tile_fill_plane(lines, x0, z0, x1, z1, y_floor, OAK_FLOOR)
    tile_fill_plane(lines, x0, z0, x1, z1, y_ceiling, SEA_LANTERN)
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

    # Interior walls
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if grid[r][c] == '#':
                x, _, z = mc_pos(origin_x, origin_y, origin_z, r, c)
                column(x, z, y_clear_lo, y_clear_hi, BIRCH_WALL)

    lines.append("# doorway")

    # # Ensure start doorway clear
    # for r in range(n):
    #     for c in range(n):
    #         if grid[r][c] == 'S':
    #             x, _, z = mc_pos(origin_x, origin_y, origin_z, r, c)
    #             lines.append(f"fill {x} {y_clear_lo} {z} {x} {y_clear_hi} {z} {AIR}")

    # Exit command block + plate
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'E':
                ex, _, ez = mc_pos(origin_x, origin_y, origin_z, r, c)
                lines.append(f"fill {ex} {y_clear_lo} {ez} {ex} {y_clear_hi} {ez} {AIR}")
                lines.append(f"setblock {ex} {y_floor} {ez} minecraft:command_block[facing=up]{{auto:0b}}")
                lines.append(
                    f'data modify block {ex} {y_floor} {ez} Command set value '
                    f'"execute as @p run function {namespace}:finish_plate"'
                )
                lines.append(f"setblock {ex} {y_clear_lo} {ez} minecraft:stone_pressure_plate")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    sr, sc = find_start(grid)
    sxx, _, szz = mc_pos(origin_x, origin_y, origin_z, sr, sc)
    tx = sxx + (1 if sc == 0 else 0)
    tz = szz + (1 if sr == 0 else 0)
    ty = y_clear_lo
    return (tx, ty, tz)

# ----------------- TP / CLEAR / TIMER (unchanged) -----------------
def write_tp_function(tp_path: str, tp_selector: str, tp_coords: Tuple[int,int,int]) -> None:
    tx, ty, tz = tp_coords
    lines = [
        "# Teleport players to the start point (safe carve + TP)",
        f"fill {tx} {ty} {tz} {tx} {ty+1} {tz} minecraft:air",
        f"tp {tp_selector} {tx} {ty} {tz}",
    ]
    with open(tp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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
        TILE = 181
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

def write_timer_functions(
    namespace: str,
    start_path: str,
    tick_path: str,
    finish_plate_path: str,
    tp_selector: str,
    tp_coords: Tuple[int,int,int],
    warden_world_positions: List[Tuple[int,int,int]],
    warden_persist: bool,
    arena_box: Tuple[int,int,int,int,int,int],
) -> None:
    sx, sy, sz = tp_coords
    nbt = "{PersistenceRequired:1b}" if warden_persist else ""
    start_lines = [
        "# Start a timed run: rebuild arena, TP players, tag as runners, summon Wardens, start ticking",
        f"function {namespace}:clear_area",
        f"function {namespace}:build_layout",
        "kill @e[type=minecraft:warden]",
        "scoreboard objectives add runTime dummy",
        'scoreboard objectives modify runTime displayname {"text":"Run Time (ticks)"}',
        f"tag {tp_selector} add runner",
        f"scoreboard players set {tp_selector} runTime 0",
    ]
    # carve a little headroom
    for i in range(10):
        for j in range(3):
            for k in range(10):
                start_lines.append(f"fill {sx} {sy} {sz} {sx+i} {sy+j} {sz+k} minecraft:air")
    start_lines += [
        "kill @e[type=minecraft:item]",
        f"tp {tp_selector} {sx} {sy} {sz}",
        "gamemode adventure @a[tag=runner]",
        '# reset everyone’s visible score so only runners show',
        'scoreboard players reset @a[tag=runner] runTime',
        'scoreboard players set @a[tag=runner] runTime 0',
        '# show the timer on the sidebar (choose sort order)',
        'scoreboard objectives setdisplay sidebar runTime',
        'team leave @a[tag=runner]',
        # "team join alive @a[tag=runner]   # optional—only if you want an 'alive' team",
    ]
    for (wx, wy, wz) in warden_world_positions:
        start_lines.append(f"setblock {wx} {wy} {wz} minecraft:air")
        start_lines.append(f"summon minecraft:warden {wx} {wy} {wz} {nbt}".rstrip())
    start_lines.append(f"schedule function {namespace}:timer_tick 1t replace")
    with open(start_path, "w", encoding="utf-8") as f:
        f.write("\n".join(start_lines))

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

    x0, y0, z0, x1, y1, z1 = arena_box
    finish_lines = [
        "# Finish triggered by exit plate",
        "scoreboard objectives add runTime dummy",
        'tellraw @s [{"text":"Finished! Time: ","color":"yellow"},'
        '{"score":{"name":"*","objective":"runTime"}},{"text":" ticks (~","color":"gray"},'
        '{"text":"seconds = ticks/20","color":"gray"},{"text":")"}]',
        "tag @a remove runner",
        "scoreboard players reset @a runTime",
        "kill @e[type=minecraft:warden]",
        'kill @e[type=minecraft:item,nbt={Item:{id:"minecraft:sculk_catalyst"}}]',
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

# --- Grid → Image / PDF -----------------------------------------------
from typing import List, Tuple, Dict, Optional
try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Please install Pillow first:  pip install pillow")

def grid_to_image(
    grid: List[List[str]],
    cell: int = 12,
    margin: int = 8,
    *,
    colors: Optional[Dict[str, Tuple[int,int,int]]] = None,
    grid_lines: bool = False,
    max_size_px: int = 4096
) -> Image.Image:
    """
    Render an ASCII grid to a PIL Image.

    grid: list of list of chars. ' ' = open, '#' = wall, 'S' = start, 'E' = exit
    cell: pixel size of each grid cell
    margin: pixel padding around the grid
    colors: mapping override; keys: floor, wall, start, exit, grid
    grid_lines: draw faint grid lines if True
    max_size_px: scale cell size down automatically if the image would exceed this width/height
    """
    n = len(grid)
    if n == 0 or len(grid[0]) != n:
        raise ValueError("grid must be an n×n list of lists")

    # default palette (soft background, dark walls)
    if colors is None:
        colors = {
            "floor": (245, 245, 240),
            "wall":  (40, 44, 52),
            "start": (51, 136, 255),
            "exit":  (234, 67, 53),
            "grid":  (210, 210, 210),
        }

    # auto-scale if too large
    total_w = 2*margin + n*cell
    total_h = 2*margin + n*cell
    if total_w > max_size_px or total_h > max_size_px:
        scale = min(max_size_px / total_w, max_size_px / total_h)
        cell = max(1, int(cell * scale))
        total_w = 2*margin + n*cell
        total_h = 2*margin + n*cell

    img = Image.new("RGB", (total_w, total_h), colors["floor"])
    d = ImageDraw.Draw(img)

    # draw cells
    for r in range(n):
        for c in range(n):
            x0 = margin + c*cell
            y0 = margin + r*cell
            x1 = x0 + cell
            y1 = y0 + cell
            ch = grid[r][c]
            if ch == '#':
                d.rectangle([x0, y0, x1-1, y1-1], fill=colors["wall"])
            elif ch == 'S':
                d.rectangle([x0, y0, x1-1, y1-1], fill=colors["start"])
            elif ch == 'E':
                d.rectangle([x0, y0, x1-1, y1-1], fill=colors["exit"])
            # spaces (' ') are already floor-colored background

    # optional grid lines (subtle)
    if grid_lines and cell >= 4:
        for i in range(n+1):
            y = margin + i*cell
            d.line([margin, y, margin + n*cell, y], fill=colors["grid"])
            x = margin + i*cell
            d.line([x, margin, x, margin + n*cell], fill=colors["grid"])

    return img

def save_grid_png_pdf(
    grid: List[List[str]],
    png_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    **kwargs
) -> Tuple[Optional[str], Optional[str]]:
    """
    Render the grid and save it as PNG and/or PDF (single-page).

    kwargs are passed to grid_to_image (cell, margin, colors, grid_lines, max_size_px).
    Returns (png_path or None, pdf_path or None).
    """
    if not png_path and not pdf_path:
        raise ValueError("Provide at least one of png_path or pdf_path")

    img = grid_to_image(grid, **kwargs)

    if png_path:
        img.save(png_path, format="PNG")

    if pdf_path:
        # Pillow can write a single image directly to PDF
        img.convert("RGB").save(pdf_path, format="PDF", resolution=300)

    return png_path, pdf_path


# ----------------- CLI & main -----------------
def main():
    p = argparse.ArgumentParser(description="Emit build/tp/clear/timer mcfunctions; generators include clustered, mazeish, and mazeish-wide (wider halls + user-selected halls).")
    p.add_argument("-n", "--size", type=int, required=True, help="grid size n (n x n)")

    # Generators & exit distance
    p.add_argument("--gen", choices=["clustered", "mazeish", "mazeish-wide", "mazeish-halls"], default="clustered", help="which layout generator to use")

    p.add_argument("--min-exit-frac", type=float, default=1.0, help="minimum exit distance as a fraction of interior span (Chebyshev)")
    p.add_argument("--min-exit-blocks", type=int, default=0, help="absolute minimum Chebyshev distance (in blocks) between S and E")

    # Layout knobs (clustered)
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--density", type=float, default=0.12, help="cluster coverage (approx)")
    p.add_argument("--cluster-radius", type=int, default=1, help="blob radius")
    p.add_argument("--walk-steps", type=int, default=1, help="stamping steps per seed (default 1)")

    # Mazeish-wide options
    p.add_argument("--hall-width", type=int, default=3, help="odd width for halls in mazeish-wide (e.g., 3,5,7)")
    p.add_argument("--widen-all", action="store_true", help="if set, widen every corridor in the base maze")
    p.add_argument("--rows", type=str, default="", help="comma-separated interior row indices (1..n-2) to force as halls")
    p.add_argument("--cols", type=str, default="", help="comma-separated interior column indices (1..n-2) to force as halls")
    p.add_argument("--seg", action="append", default=[], help="repeatable; each value is r0,c0,r1,c1 (axis-aligned corridor segment)")

    # World & materials
    p.add_argument("--origin", type=int, nargs=3, metavar=("X","Y","Z"), default=(100, 200, 100), help="world origin (x y z)")
    p.add_argument("--floor-axis", choices=["x", "z"], default="x", help="floorboard axis")
    p.add_argument("--walk-height", type=int, default=2, help="interior walkable height; ceiling at Y+walk_height+1")

    # Wardens
    p.add_argument("--wardens", type=int, default=0, help="number of random Warden spawns (on start)")
    p.add_argument("--warden-persist", action="store_true", help="Wardens get {PersistenceRequired:1b}")

    # Outputs & namespace
    p.add_argument("--namespace", type=str, default="br", help="datapack namespace")
    p.add_argument("--out-build", type=str, default="build_layout.mcfunction", help="output for building the arena")
    p.add_argument("--out-tp", type=str, default="tp_start.mcfunction", help="output for teleporting to start")
    p.add_argument("--out-clear", type=str, default="clear_area.mcfunction", help="output for clearing the n×n×height volume")
    p.add_argument("--out-start-run", type=str, default="start_run.mcfunction", help="output to rebuild+tp+summon+start timer")
    p.add_argument("--out-timer-tick", type=str, default="timer_tick.mcfunction", help="output for incrementing run time")
    p.add_argument("--out-finish-plate", type=str, default="finish_plate.mcfunction", help="output for plate to stop timer/cleanup")

    p.add_argument("--clear-height", type=int, default=8, help="height to clear upward from origin Y (>=1)")
    p.add_argument("--tp-selector", type=str, default="@a", help="who to teleport in tp_start/start_run")

    # Mazeish-wide "open-up" controls
    p.add_argument("--extra-doors", type=int, default=0,
                   help="punch this many additional hall_width-wide doorways through 1-thick walls")
    p.add_argument("--erode-pct", type=float, default=0.0,
                   help="probability to delete a wall cell that touches ≥2 open neighbors (e.g., 0.08)")
    p.add_argument("--room-attempts", type=int, default=0,
                   help="number of random room carves (Chebyshev blobs)")
    p.add_argument("--room-radius-range", type=int, nargs=2, metavar=("RMIN","RMAX"), default=[2,4],
                   help="radius range for random rooms (Chebyshev)")

    # mazeish-halls (macro hall overlay) controls
    p.add_argument("--hall-band-width", type=int, default=None,
                   help="width of macro hall bands over the maze (default: hall_width)")
    p.add_argument("--hall-row-step", type=int, default=None,
                   help="row spacing between horizontal macro bands")
    p.add_argument("--hall-col-step", type=int, default=None,
                   help="col spacing between vertical macro bands")
    p.add_argument("--hall-row-offset", type=int, default=2,
                   help="starting row offset for bands (interior index)")
    p.add_argument("--hall-col-offset", type=int, default=2,
                   help="starting col offset for bands (interior index)")
    p.add_argument("--seg-len", type=int, nargs=2, metavar=("MIN","MAX"), default=[6,12],
                   help="open segment length range along each band")
    p.add_argument("--gap-len", type=int, nargs=2, metavar=("MIN","MAX"), default=[3,7],
                   help="gap length range between open segments")



    args = p.parse_args()

    if args.size < 7:
        raise SystemExit("Choose n >= 7 for nicer mazes with borders.")
    if not (0.0 <= args.density <= 0.9):
        raise SystemExit("density must be between 0.0 and 0.9")
    if args.cluster_radius < 0:
        raise SystemExit("cluster-radius must be >= 0")
    if args.walk_height < 1:
        raise SystemExit("--walk-height must be >= 1")
    if args.clear_height < 1:
        raise SystemExit("--clear-height must be >= 1")

    rng = random.Random(args.seed)

    if args.gen == "mazeish-wide":
        rows = parse_csv_ints(args.rows)
        cols = parse_csv_ints(args.cols)
        segs = parse_segments(args.seg)
        grid = generate_open_layout_mazeish_wide(
            args.size, rng,
            hall_width=args.hall_width,
            widen_all=True,  # already wide
            rows_to_carve=rows,
            cols_to_carve=cols,
            segments_to_carve=segs,
            min_exit_frac=args.min_exit_frac,
            min_exit_blocks=args.min_exit_blocks,
            extra_doors=args.extra_doors,
            erode_pct=args.erode_pct,
            room_attempts=args.room_attempts,
            room_radius_range=tuple(args.room_radius_range),
        )
    elif args.gen == "mazeish":
        grid = generate_open_layout_mazeish(
            args.size, rng,
            min_exit_frac=args.min_exit_frac,
            min_exit_blocks=args.min_exit_blocks,
            extra_openings=max(2, args.size),
            room_attempts=max(1, args.size//2),
            room_radius_range=(1,2),
        )
    elif args.gen == "mazeish-halls":
        rows = parse_csv_ints(args.rows)    # optional: still available but unused by halls
        cols = parse_csv_ints(args.cols)
        segs = parse_segments(args.seg)
        grid = generate_open_layout_mazeish_halls(
            args.size, rng,
            hall_width=args.hall_width,
            hall_band_width=args.hall_band_width if args.hall_band_width else args.hall_width,
            row_step=args.hall_row_step, col_step=args.hall_col_step,
            row_offset=args.hall_row_offset, col_offset=args.hall_col_offset,
            seg_len=tuple(args.seg_len), gap_len=tuple(args.gap_len),
            min_exit_frac=args.min_exit_frac, min_exit_blocks=args.min_exit_blocks,
        )
    else:
        grid = generate_open_layout_clustered(
            args.size, rng,
            density=args.density,
            cluster_radius=args.cluster_radius,
            walk_steps=args.walk_steps,
            min_exit_blocks=args.min_exit_blocks,
        )


    # Optional dump to console to visualize
    print_grid(grid)

    # Save both versions
    save_grid_png_pdf(
        grid,
        png_path="maze_layout.png",
        pdf_path="maze_layout.pdf",
        cell=10,           # bigger = higher resolution
        margin=12,
        grid_lines=True    # turn off if you prefer a clean look
    )

    ox, oy, oz = args.origin
    tp_coords = write_build_function(
        grid, ox, oy, oz, args.out_build,
        walk_height=args.walk_height,
        floor_axis=args.floor_axis,
        namespace=args.namespace,
    )
    write_tp_function(args.out_tp, args.tp_selector, tp_coords)
    write_clear_function(args.out_clear, ox, oy, oz, args.size, args.clear_height)

    # Precompute Warden spawns
    warden_world_positions: List[Tuple[int,int,int]] = []
    if args.wardens > 0:
        candidates = collect_open_tiles(grid)
        rng.shuffle(candidates)
        picks = candidates[:min(args.wardens, len(candidates))]
        wy = oy + 1
        for (rr, cc) in picks:
            wx, _, wz = mc_pos(ox, oy, oz, rr, cc)
            warden_world_positions.append((wx, wy, wz))

    x0, y0, z0 = ox, oy, oz
    x1, y1, z1 = ox + args.size - 1, oy + args.clear_height - 1, oz + args.size - 1
    arena_box = (x0, y0, z0, x1, y1, z1)

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
    print(f"  /function {args.namespace}:build_layout")
    print(f"  /function {args.namespace}:tp_start")
    print(f"  /function {args.namespace}:clear_area")
    print(f"  /function {args.namespace}:start_run   # clear->build->tp->summon->timer")
    print("Step on the exit plate to finish & cleanup.")

if __name__ == "__main__":
    main()

