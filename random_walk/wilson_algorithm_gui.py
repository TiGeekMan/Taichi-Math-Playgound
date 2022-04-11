import taichi as ti
import numpy as np
import random
from taichi.tools.image import imresize


N = 80
margin = 2
width = 2 * (N + margin) - 1
height = 2 * (N + margin) - 1
anim_speed = 30
scale = 5
WALL = [0, 0, 0]
TREE = [200, 200, 200]
PATH = [255, 0, 255]
grid = np.zeros((height, width, 3), dtype=np.uint8)
pixels = np.zeros((scale * height, scale * width, 3), dtype=np.uint8)


cells = []
for y in range(margin, height - margin, 2):
    for x in range(margin, width - margin, 2):
        cells.append((x, y))


def neighborhood(cell):
    x, y = cell
    neighbors = []
    if x >= margin + 2:
        neighbors.append((x - 2, y))
    if y >= margin + 2:
        neighbors.append((x, y - 2))
    if x <= width - 3 - margin:
        neighbors.append((x + 2, y))
    if y <= height - 3 - margin:
        neighbors.append((x, y + 2))
    return neighbors


graph = {v: neighborhood(v) for v in cells}


def get_neighbors(cell):
    return graph[cell]


def mark_cell(cell, value):
    x, y = cell
    grid[x, y] = value


def mark_space(c1, c2, value):
    c = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
    mark_cell(c, value)


def mark_path(path, value):
    for cell in path:
        mark_cell(cell, value)
    for c1, c2 in zip(path[1:], path[:-1]):
        mark_space(c1, c2, value)


def barrier(c1, c2):
    x = (c1[0] + c2[0]) // 2
    y = (c1[1] + c2[1]) // 2
    return all(grid[x, y] == WALL)


def is_wall(cell):
    x, y = cell
    return all(grid[x, y] == WALL)


def in_tree(cell):
    x, y = cell
    return all(grid[x, y] == TREE)


def in_path(cell):
    x, y = cell
    return all(grid[x, y] == PATH)


def add_cell_to_path(path, cell):
    mark_cell(cell, PATH)
    mark_space(path[-1], cell, PATH)
    path.append(cell)


def erase_loop(path, cell):
    index = path.index(cell)
    mark_path(path[index:], WALL)
    mark_cell(path[index], PATH)
    return path[: index + 1]


def resize_grid():
    return imresize(grid, scale * width, scale * height)


gui = ti.GUI("Wilson algorithm", res=(scale * width, scale * height))
root = cells[0]
mark_cell(root, TREE)
count = 0
while gui.running:
    for cell in cells:
        if not in_tree(cell):
            lerw = [cell]
            mark_cell(cell, PATH)
            current_cell = cell

            while not in_tree(current_cell):
                next_cell = random.choice(get_neighbors(current_cell))
                if in_path(next_cell):
                    lerw = erase_loop(lerw, next_cell)

                elif in_tree(next_cell):
                    add_cell_to_path(lerw, next_cell)
                    mark_cell(next_cell, TREE)

                else:
                    add_cell_to_path(lerw, next_cell)

                current_cell = next_cell
                if count % anim_speed == 0:
                    gui.set_image(resize_grid())
                    gui.show()

                count += 1

            mark_path(lerw, TREE)

    gui.set_image(resize_grid())
    gui.show()
