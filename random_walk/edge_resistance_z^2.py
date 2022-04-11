import taichi as ti
from taichi.math import ivec2

ti.init(dynamic_index=True)

num_rounds = 100000
size = 1000

tree = ti.Vector.field(2, ti.i32, shape=(size, size))
success = ti.field(ti.i32, shape=())


@ti.func
def get_random_neighbor(v):
    neighbours = ti.Matrix.zero(ti.i32, 4, 2)
    x, y = v
    count = 0
    if x < size - 1:
        neighbours[0, 0] = x + 1
        neighbours[0, 1] = y
        count = 1
    if y < size - 1:
        neighbours[1, 0] = x
        neighbours[1, 1] = y + 1
        count = 2
    if x > 0:
        neighbours[2, 0] = x - 1
        neighbours[2, 1] = y
        count = 3
    if y > 0:
        neighbours[3, 0] = x
        neighbours[3, 1] = y - 1
        count = 4
    ind = ti.cast(ti.floor(ti.random() * count), int)
    x, y = neighbours[ind, 0], neighbours[ind, 1]
    return ivec2(x, y)


root = ivec2(size // 2, size // 2)
start = root + ivec2(1, 0)


@ti.kernel
def main():
    if True:
        for _ in range(num_rounds):
            pos = start
            while not all(pos == root):
                neighbour = get_random_neighbor(pos)
                tree[pos[0], pos[1]] = neighbour
                pos = neighbour

            success[None] += int(all(tree[start[0], start[1]] == root))

main()
print(success[None] / num_rounds)
