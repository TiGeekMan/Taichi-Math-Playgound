"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Taichi animation of polyhedra mirrors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import taichi as ti
import numpy as np
from polytopes import models

ti.init(arch=ti.cpu)  # set arch=ti.gpu if you have cuda backend


edge_thickness = 0.05
max_reflections = 10
resolution = (800, 600)  # window resolution
pixels = ti.Vector.field(3, ti.f32, shape=resolution)  # image pixels


@ti.func
def vec2(x, y):
    return ti.Vector([float(x), float(y)])


@ti.func
def vec3(x, y, z):
    return ti.Vector([float(x), float(y), float(z)])


@ti.func
def clamp(v, v_min, v_max):
    return ti.max(ti.min(v, v_max), v_min)


@ti.func
def orient(p, a, b, n):
    z = (a - p).cross(b - p)
    return z.dot(n)


@ti.func
def reflect(v, n):
    return v - 2. * v.dot(n) * n


@ti.func
def mix(a, b, t):
    return a * (1.0 - t) + b * t


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def camera_matrix(location, lookat, up):
    forward = (lookat - location).normalized()
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    return ti.Matrix.cols([right, up, -forward])


def load_texture(image_path):
    image = ti.tools.imread(image_path).astype(float) / 255.0
    H, W = image.shape[:2]
    texture = ti.Vector.field(3, ti.f32, shape=(H, W))
    texture.from_numpy(image)
    return texture


woodTex = load_texture("./glslhelpers/textures/wood.jpg")
skyboxTex = load_texture("./glslhelpers/textures/skybox.jpg")


def generate_polytope_data(
        coxeter_diagram,
        trunc_type,
        extra_relations=(),
        snub=False,
        dual=False
):
    """Generate data for polyhedra vertex coordinates and face indices.
    """
    if snub:
        P = models.Snub(coxeter_diagram, extra_relations=extra_relations)
    else:
        P = models.Polyhedra(coxeter_diagram, trunc_type, extra_relations)

    if dual:
        P = models.Catalan3D(P)

    P.build_geometry()
    vertices_coords = P.vertices_coords

    def get_taichi_field_for_faces(face_group):
        """Put all vertex coordinates of faces of the same type into a
        large (m, n, 3) taichi field.
        """
        m = len(face_group)
        n = len(face_group[0])
        faces_data = ti.Vector.field(3, ti.f32, shape=(m, n))
        faces_data_numpy = np.zeros((m, n, 3), dtype=float)
        for ind, face in enumerate(face_group):
            face_coords = [vertices_coords[ind] for ind in face]
            face_center = sum(face_coords) / len(face)
            v0, v1, v2 = face_coords[:3]
            normal = np.cross(v1 - v0, v2 - v0)
            if np.dot(face_center, normal) < 0:
                face_coords = face_coords[::-1]

            faces_data_numpy[ind, :, :] = face_coords

        faces_data.from_numpy(faces_data_numpy)
        return faces_data

    facesEnabled = ti.Vector.field(3, ti.i32, shape=())
    facesEnabled = [True, False, False]
    facesA = get_taichi_field_for_faces(P.face_indices[0])
    if len(P.face_indices) > 1:
        facesB = get_taichi_field_for_faces(P.face_indices[1])
        facesEnabled[1] = True
    else:
        facesB = facesA
    if len(P.face_indices) > 2:
        facesC = get_taichi_field_for_faces(P.face_indices[2])
        facesEnabled[2] = True
    else:
        facesC = facesA
    return facesEnabled, facesA, facesB, facesC


@ti.func
def nohit():
    """
    t: distance travelled along the ray.
    ed: distance to the edge of the polygon face.
    normal: normal vector of the face.
    bary: bary coordinates of the intersection point about the first three face vertices.
    """
    return ti.Struct(t=1000.0,
                     ed=0.0,
                     normal=vec3(0, 0, 0),
                     bary=vec3(0, 0, 0))


@ti.func
def get_ray_intersection_for_given_faces(ray_origin, ray_dir, faces):
    """Check the ray-face intersection for faces of the same shape.
    """
    m, n = faces.shape
    hit = nohit()
    # for each face we check if the ray has intersection with it
    for k in ti.ndrange(m):
        # firstly we check if the ray intersects with the plane.
        # we need only three vertices to verify this
        v0 = faces[k, 0]
        v1 = faces[k, 1]
        v2 = faces[k, 2]
        v10 = v1 - v0
        v20 = v2 - v0

        q = ray_origin - v0
        d20 = ray_dir.cross(v20)
        q10 = q.cross(v10)
        det = v10.dot(d20)
        h = q10.dot(v20)
        u = q.dot(d20) / det
        v = ray_dir.dot(q10) / det
        w = 1.0 - u - v
        t = h / det
        count = 0
        if t > 1e-3:
            # then we check if the intersection point lies within the face
            p = ray_origin + t * ray_dir
            normal = v10.cross(v20).normalized()
            ed_sqr = (p - v0).norm_sqr()
            for i in ti.ndrange(n):
                j = (i - 1) % n
                e = faces[k, j] - faces[k, i]
                ip = p - faces[k, i]
                b = ip - e * clamp(ip.dot(e) / e.norm_sqr(), 0, 1)
                ed_sqr = ti.min(ed_sqr, b.norm_sqr())
                count += orient(p, faces[k, j], faces[k, i], normal) >= 0.0

            if (count == n or count == -n) and t < hit.t:
                hit = ti.Struct(t=t,
                                ed=ti.sqrt(ed_sqr),
                                normal=normal,
                                bary=vec3(u, v, w))
    return hit


@ti.func
def get_ray_intersection(ray_origin, ray_dir):
    """Check the ray-face intersection for all faces.
    """
    result = get_ray_intersection_for_given_faces(ray_origin, ray_dir, facesA)
    hitB = nohit()
    hitC = nohit()
    if ti.static(facesEnabled[1]):
        hitB = get_ray_intersection_for_given_faces(ray_origin, ray_dir, facesB)
    if ti.static(facesEnabled[2]):
        hitC = get_ray_intersection_for_given_faces(ray_origin, ray_dir, facesC)
    if hitB.t < result.t:
        result = hitB
    if hitC.t < result.t:
        result = hitC
    return result


@ti.func
def hit_from_outside(ray_origin, ray_dir):
    """Check if a ray intersects the faces from the outside.
    """
    result = nohit()
    hit = get_ray_intersection(ray_origin, ray_dir)
    if hit.t <= 10.0 and ray_dir.dot(hit.normal) < 0.:
        result = hit
    return result


@ti.func
def hit_from_inside(ray_origin, ray_dir):
    """Check if a ray intersects the faces from the inside.
    """
    result = nohit()
    hit = get_ray_intersection(ray_origin, ray_dir)
    if hit.t <= 10.0:
        result = hit
    return result


@ti.func
def texture(tex, uv):
    """Sample from a texture image.
    """
    tex_w, tex_h = tex.shape
    uv = uv * ti.Vector([tex_w - 1, tex_h - 1])
    k = int(uv)
    t = fract(uv)
    k[0] = k[0] % tex_w
    k[1] = k[1] % tex_h
    return (
        (
            (tex[k] * (1 - t[0])) + tex[k + ti.Vector([1, 0])] * t[0]
        ) * (1 - t[1]) +
        (
            tex[k + ti.Vector([1, 0])] * (1 - t[0]) + tex[k + ti.Vector([1, 1])] * t[0]
        ) * t[1]
    )


@ti.func
def cubemap_coord(dir):
    """Sample from a cubemap texture.
    """
    eps = 1e-7
    coor = vec2(0, 0)
    if dir.z >= 0 and dir.z >= abs(dir.y) - eps and dir.z >= abs(dir.x) - eps:
        coor = vec2(3 / 8, 1 / 2) + vec2(dir.x / 8, dir.y / 6) / dir.z

    if dir.z <= 0 and -dir.z >= abs(dir.y) - eps and -dir.z >= abs(dir.x) - eps:
        coor = vec2(7 / 8, 1 / 2) + vec2(-dir.x / 8, dir.y / 6) / -dir.z

    if dir.x <= 0 and -dir.x >= abs(dir.y) - eps and -dir.x >= abs(dir.z) - eps:
        coor = vec2(1 / 8, 1 / 2) + vec2(dir.z / 8, dir.y / 6) / -dir.x

    if dir.x >= 0 and dir.x >= abs(dir.y) - eps and dir.x >= abs(dir.z) - eps:
        coor = vec2(5 / 8, 1 / 2) + vec2(-dir.z / 8, dir.y / 6) / dir.x

    if dir.y >= 0 and dir.y >= abs(dir.x) - eps and dir.y >= abs(dir.z) - eps:
        coor = vec2(3 / 8, 5 / 6) + vec2(dir.x / 8, -dir.z / 6) / dir.y

    if dir.y <= 0 and -dir.y >= abs(dir.x) - eps and -dir.y >= abs(dir.z) - eps:
        coor = vec2(3 / 8, 1 / 6) + vec2(dir.x / 8, dir.z / 6) / -dir.y

    return coor


@ti.func
def get_wall_color(ray_dir, hit):
    albedo = texture(woodTex, vec2(hit.bary[0], hit.bary[1]) * 2.0)
    albedo = ti.pow(albedo, vec3(2.2, 2.2, 2.2)) * 0.5
    lighting = 0.2 + ti.max(hit.normal.dot(vec3(0.8, 0.5, 0)), 0)
    result = ti.Vector([0., 0., 0., 0.])
    if ray_dir.dot(hit.normal) < 0:
        f = clamp(hit.ed * 1e3 - 3, 0, 1)
        albedo = mix(vec3(0.01, 0.01, 0.01), albedo, f) * lighting
        result = ti.Vector([albedo[0], albedo[1], albedo[2], f])
    else:
        m = ti.max(hit.bary[0], ti.max(hit.bary[1], hit.bary[2]))
        a = fract(vec2(hit.ed, m) * 40.6) - 0.5
        b = 0.2 / (a.norm_sqr() + 0.2)
        lightShape = 1 - clamp(hit.ed * 100.0 - 2., 0, 1)
        lightShape *= b
        emissive = vec3(3.5, 1.8, 1)
        emissive = mix(albedo * lighting, emissive, lightShape)
        result = ti.Vector([emissive[0], emissive[1], emissive[2], 0])

    return result


@ti.func
def get_background_color(ray_dir):
    uv = cubemap_coord(-ray_dir)
    color = texture(skyboxTex, uv)
    color = ti.pow(color, vec3(2.2, 2.2, 2.2))
    luma = color.dot(vec3(0.2126, 0.7152, 0.0722)) * 0.7
    return 2.5 * color / (1 - luma)


@ti.func
def get_ray_color(ray_origin, ray_dir):
    color = vec3(0, 0, 0)
    hit = hit_from_outside(ray_origin, ray_dir)
    if hit.t > 10.0:
        color = get_background_color(ray_dir)
    else:
        ref_dir = reflect(ray_dir, hit.normal)
        bg_color = get_background_color(ref_dir)
        bg_color = ti.pow(bg_color, vec3(1, 1, 1))
        fresnel = 0.05 + 0.95 * ti.pow(1 - max(0, ray_dir.dot(-hit.normal)), 5.0)
        color += bg_color * fresnel

        if hit.ed < edge_thickness:
            wc = get_wall_color(ray_dir, hit)
            color = color * wc[3] + vec3(wc[0], wc[1], wc[2])

        else:
            ray_origin = ray_origin + hit.t * ray_dir
            hit = hit_from_inside(ray_origin, ray_dir)
            transmit = vec3(1, 1, 1)
            paint = True
            for _ in ti.static(range(max_reflections)):
                if paint:
                    if hit.ed < edge_thickness:
                        wc = get_wall_color(ray_dir, hit)
                        color += transmit * vec3(wc[0], wc[1], wc[2])
                        paint = False
                    else:
                        ray_origin += hit.t * ray_dir
                        ray_dir = reflect(ray_dir, hit.normal)
                        ray_origin += ray_dir * 0.001
                        transmit *= vec3(0.4, 0.7, 0.7)
                        hit = hit_from_inside(ray_origin, ray_dir)

    return color


@ti.func
def post_process(color):
    """Gamma correction.
    """
    x, y, z = color
    x /= (0.5 + 0.5 * x)
    y /= (0.5 + 0.5 * y)
    z /= (0.5 + 0.5 * z)
    return ti.pow(vec3(x, y, z), 1/2.2)


@ti.kernel
def render(time: ti.f32):
    W, H = resolution
    mx = time * 0.2
    my = ti.sin(time * 0.2) * 0.5
    camera_location = 2 * vec3(ti.cos(mx) * ti.cos(my),
                               ti.sin(my),
                               ti.sin(mx) * ti.cos(my))
    lookat = vec3(0, 0, 0)
    M = camera_matrix(camera_location, lookat, vec3(0, 1, 0))
    for ix, iy in pixels:
        uv = ti.Vector([(ix / H - 0.5 * W / H), 0.5 - iy / H])
        screen_ray = vec3(uv[0], uv[1], -1).normalized()
        camera_ray = (M @ screen_ray).normalized()
        color = get_ray_color(camera_location, camera_ray)
        pixels[ix, iy] = post_process(color)


if __name__ == "__main__":
    facesEnabled, facesA, facesB, facesC = generate_polytope_data(
        (4, 2, 3),
        (0, 1, 0),
        snub=False,
        dual=True
    )
    gui = ti.GUI("Test", res=resolution)
    for i in range(1000000):
        render(i * 0.03)
        gui.set_image(pixels)
        gui.show()
