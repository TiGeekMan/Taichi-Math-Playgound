from glob import escape
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, kernel_profiler=True)

dim = 2
vec = ti.types.vector(dim, ti.i32)
num_rounds = 1000000
max_steps = 1000000
success_count = ti.field(ti.i32, shape=())
origin = ti.Vector([0] * dim)
escape_radius = 50


def get_grid_neighbours():
    def make_direction(k, val):
        li = [0] * dim
        li[k] = val
        return li
    
    result = ti.Vector.field(dim, ti.i32, shape=(2*dim,))
    for k in range(dim):
        result[2 * k] = make_direction(k, 1)
        result[2 * k + 1] = make_direction(k, -1)
    return result


neighbours = get_grid_neighbours()


def win_probability(R):
    return 1 / (2.0 / np.pi * np.log(R) + 1.0293737)


@ti.func
def choose_random_direction():
    rnd = ti.cast(ti.floor(ti.random() * 2 * dim), int)
    return neighbours[rnd]


@ti.kernel
def do_experiments():
    for _ in range(num_rounds):
        pos = origin
        for step in range(max_steps):
            pos += choose_random_direction()
            dist2 = pos.norm_sqr()
            if dist2 == 0:
                break
            if dist2 > ti.static(escape_radius ** 2):
                success_count[None] += 1
                break

do_experiments()
print("True probability:", win_probability(escape_radius))
print("Estimated probability:", success_count[None] / num_rounds)
ti.profiler.print_kernel_profiler_info()
