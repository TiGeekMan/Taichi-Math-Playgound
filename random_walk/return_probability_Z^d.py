"""
Compute the return probability of a simple random walk on the n-D square grid.
"""
import taichi as ti
ti.init(arch=ti.gpu, kernel_profiler=True)

dim = 3
num_rounds = 10000000
max_steps = 1000000
success_count = ti.field(ti.i32, shape=())

vec = ti.types.vector(dim, ti.i32)
origin = vec([0] * dim)


def get_grid_neighbours():
    def make_direction(k, val):
        li = [0] * dim
        li[k] = val
        return li
    
    result = ti.Vector.field(dim, ti.i32, shape=(2 * dim,))
    for k in range(dim):
        result[2 * k] = make_direction(k, 1)
        result[2 * k + 1] = make_direction(k, -1)
    return result


neighbours = get_grid_neighbours()


@ti.func
def get_random_direction():
    rnd = ti.cast(ti.floor(ti.random() * 2 * dim), int)
    return neighbours[rnd]


@ti.kernel
def do_experiments():
    for _ in range(num_rounds):
        pos = origin
        for step in range(max_steps):
            pos += get_random_direction()
            if all(pos == origin):
                success_count[None] += 1
                break


do_experiments()
print(success_count[None] / num_rounds)
ti.profiler.print_kernel_profiler_info()
