import taichi as ti
ti.init(dynamic_index=True)

num_rounds = 100000
max_steps = 10000
success = ti.field(ti.i32, shape=())

dim = 3
vec = ti.types.vector(dim, ti.i32)

start = vec([1] * dim)
end = vec([-1] * dim)


@ti.func
def get_random_move(vert):
    index = int(ti.floor(ti.random() * dim))
    neighbour = vert
    neighbour[index] *= -1
    return neighbour


@ti.func
def vector_equal(v1, v2):
    return all(v1 == v2)


@ti.kernel
def main():
    for _ in range(num_rounds):
        prev = start
        for step in range(max_steps):
            next = get_random_move(prev)
            if vector_equal(next, end):
                success[None] += 1
                break
            if vector_equal(next, start):
                break

            prev = next


main()
prob = success[None] / num_rounds
print(1.0 / (dim * prob))
