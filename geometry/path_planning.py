import numpy as np
import heapq

def build_grid(size, obstacles):
    grid = np.ones((size, size), dtype=np.uint8)
    for poly in obstacles:
        minx, miny, maxx, maxy = map(int, poly.bounds)
        grid[miny:maxy, minx:maxx] = 0
    return grid

def astar(grid, start, goal):
    h, w = grid.shape
    pq = [(0, start)]
    came = {start: None}
    cost = {start: 0}

    moves = [(1,0),(-1,0),(0,1),(0,-1)]

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            break

        for dx, dy in moves:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny,nx] == 1:
                new_cost = cost[current] + 1
                if (nx,ny) not in cost or new_cost < cost[(nx,ny)]:
                    cost[(nx,ny)] = new_cost
                    priority = new_cost + abs(nx-goal[0]) + abs(ny-goal[1])
                    heapq.heappush(pq, (priority, (nx,ny)))
                    came[(nx,ny)] = current

    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = came.get(cur)
    return path[::-1]
