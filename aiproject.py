import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# ---------- Heuristic Function ----------
def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ---------- A* Algorithm ----------
def astar_visual(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = []

    while open_list:
        current = heapq.heappop(open_list)[1]
        visited.append(current)

        if current == goal:
            # Reconstruct final path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited

        # Explore 4-directional neighbors
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[neighbor[0]][neighbor[1]] == 1:
                    continue  # Skip walls

                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return None, visited  # No path found

# ---------- Random Maze Generator ----------
def generate_maze(rows, cols, wall_prob=0.3):
    """Generate a random maze with 0s (paths) and 1s (walls)"""
    maze = [[1 if random.random() < wall_prob else 0 for _ in range(cols)] for _ in range(rows)]
    maze[0][0] = 0  # Start
    maze[rows - 1][cols - 1] = 0  # Goal
    return maze

# ---------- Auto-Solvable Maze ----------
def generate_solvable_maze(rows, cols, wall_prob=0.3):
    """Keep generating random mazes until one is solvable"""
    print("Generating solvable maze...")
    attempts = 0
    while True:
        maze = generate_maze(rows, cols, wall_prob)
        path, visited = astar_visual(maze, (0, 0), (rows - 1, cols - 1))
        attempts += 1
        if path:  # Path found
            print(f"✅ Solvable maze generated")
            return maze, path, visited

# ---------- Maze Setup ----------
rows, cols = 12, 12
maze, path, visited = generate_solvable_maze(rows, cols, wall_prob=0.28)
start = (0, 0)
goal = (rows - 1, cols - 1)

# ---------- Visualization Setup ----------
color_map = {
    0: [1, 1, 1],  # White (open cell)
    1: [0, 0, 0],  # Black (wall)
}

fig, ax = plt.subplots()
img = np.array([[color_map[val] for val in row] for row in maze], dtype=float)
im = ax.imshow(img)

def update(frame):
    """Update frame for animation"""
    temp = np.array([[color_map[val] for val in row] for row in maze], dtype=float)

    # Mark visited cells
    for (x, y) in visited[:frame]:
        temp[x][y] = [0.3, 0.5, 1.0]  # Blue (visited)

    # Mark final path
    if path:
        for (x, y) in path:
            temp[x][y] = [1.0, 1.0, 0.3]  # Yellow (final path)

    # Mark start & goal
    sx, sy = start
    gx, gy = goal
    temp[sx][sy] = [0.0, 1.0, 0.0]  # Green (start)
    temp[gx][gy] = [1.0, 0.0, 0.0]  # Red (goal)

    im.set_array(temp)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(visited) + 20,
                              interval=180, repeat=False)

plt.title("Maze Solving Problem With A* Algorithm")
plt.axis("off")
plt.show()