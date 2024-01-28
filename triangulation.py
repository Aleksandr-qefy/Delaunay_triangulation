from typing import Union, Tuple, List, Dict, Set
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import random

Digit = Union[int, float]
Color = Tuple[int, int, int]
Triangle = Tuple[int, int, int]
Graph = Dict[int, Set[int]]

POINTS_TO_GEN = 100

CANVAS_WIDTH = 1000
CANVAS_PADDING = 100
CANVAS_M_PADDING = CANVAS_WIDTH - CANVAS_PADDING * 2
COLOR: Color = (0, 255, 0)
GRAY: Color = (245, 245, 245)

EPS = 1e-10


class Point:
    def __init__(self, x: Digit = 0, y: Digit = 0, z: Digit = 0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __ne__(self, other):
        return (self.x, self.y, self.z) != (other.x, other.y, other.z)


# алгоритм Брезенхэма рисования отрезка по двум заданным точкам
def Bresenhams_line_algorithm(canvas_matrix: np.array, p1: Tuple[Digit, Digit], p2: Tuple[Digit, Digit], table=None) -> None:
    x0, y0 = p1
    x1, y1 = p2
    c_a, c_b = canvas_matrix[p1[1], p1[0]], canvas_matrix[p2[1], p2[0]]
    c_a, c_b = np.array(c_a), np.array(c_b)
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = abs(y1 - y0)
    dx2 = dx * 2
    dy2 = dy * 2
    d = -dx
    y_step = 1 if y0 < y1 else -1
    d_p1_p2 = abs(dx)
    x = x0
    y = y0
    while x < x1:
        x_, y_ = (x, y) if steep else (y, x)
        d_px_p2 = abs(x - x0)
        t = d_px_p2 / d_p1_p2 if d_p1_p2 != 0 else 0
        canvas_matrix[x_, y_] = np.array((1 - t) * c_a + t * c_b, dtype='uint8')
        if table is not None:
            if y_ not in table:
                table[y_] = []
            if x_ not in table[y_]:
                table[y_].append(x_)
        d += dy2
        if d > 0:
            y = y + y_step
            d -= dx2
        x = x + 1


def triangulate(points: List[Point]) -> List[Tuple[int, int, int]]:
    """
    Алгоритм триангуляции Делоне методом заметающей прямой

    Аргументы:
      points
        список точек (объектов Point), отсортированных в порядке возрастания по парам (x, y)

    Результат
        Список из кортежей размером 3 из индексов точек из переданного списка,
        которые олицетворяют треугольники
    """

    # ф-я для создания вектора numpy
    def get_vector(p1: Point, p2: Point) -> np.array:
        return np.array([p2.x - p1.x, p2.y - p1.y])

    # ф-я для получения обратного вектора
    def reverse_vector(vector: np.array) -> np.array:
        return -vector

    # ф-я для нахождения Евклидовой нормы вектора
    def norm(vector: np.array) -> float:
        return np.sum(vector ** 2) ** 0.5

    # ф-я для нахождения синуса угла между 2-мя векторами numpy
    def vectors_sin(v1: np.array, v2: np.array) -> float:
        return (v1[0] * v2[1] - v1[1] * v2[0]) / (norm(v1) * norm(v2))

    # ф-я для нахождения индекса точки, которая находится по другую сторону
    # от указанной точки и ребра (ребро задаётся двумя точками)
    def get_other_corner_point(G: Graph, edge_p1_idx: int, edge_p2_idx: int, m_idx: int) -> Union[int, None]:
        intersection = G[edge_p1_idx] & G[edge_p2_idx]
        intersection.remove(m_idx)
        if not len(intersection):
            return None
        return max(intersection)

    # ф-я для нахождения евклидова расстояния
    def rho(p1: Point, p2: Point) -> float:
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

    # ф-я для нахождения центра окружности, проходящей через 3 переданные точки
    def circle_centre(p1: Point, p2: Point, p3: Point) -> Union[Point, None]:
        x1, y1, x2, y2, x3, y3 = p1.x, p1.y, p2.x, p2.y, p3.x, p3.y
        c = (x1 - x2) ** 2 + (y1 - y2) ** 2
        a = (x2 - x3) ** 2 + (y2 - y3) ** 2
        b = (x3 - x1) ** 2 + (y3 - y1) ** 2
        s = 2 * (a * b + b * c + c * a) - (a * a + b * b + c * c)
        if not s:
            return None
        px = (a * (b + c - a) * x1 + b * (c + a - b) * x2 + c * (a + b - c) * x3) / s
        py = (a * (b + c - a) * y1 + b * (c + a - b) * y2 + c * (a + b - c) * y3) / s
        cp = Point(px, py)
        return cp

    # предикат (ф-я), проверяющий выполнение условия Делоне для двух 3-угольных полигонов, имеющих смежное ребро
    def check_Delaunay_condition(edge_p1: Point, edge_p2: Point, p1: Point, p2: Point) -> bool:
        cp = circle_centre(edge_p1, edge_p2, p1)
        if cp is None:
            return False
        return rho(cp, p2) > rho(cp, p1)

    # ф-я, которая возвращает переданные индексы в порядке возрастания
    def in_order(*idxs) -> Union[Tuple[int, int], Triangle]:
        idxs = list(idxs)
        idxs.sort()
        return tuple(idxs)

    # задаём таблицу смежности графа
    G = {i: set() for i in range(len(points))}
    # зададим словарь из рёбер, образующих МВО, заданный в виде
    #   outer_edges[(i, j)] = {логическое значение направления обхода ребра (i, j)}
    outer_edges = {}
    # зададим словарь из вершин из МВО, заданный в виде point_outer_edges[i] = {множество смежных вершин из МВО}
    point_outer_edges = {i: set() for i in range(len(points))}
    # проверим краевое условие, в котором первые вершины множества точек располагаются на одной абсциссе
    if points[0].x == points[1].x == points[2].x:
        x_ = points[0].x
        n = 1
        while n < len(points) and x_ == points[n].x:
            n += 1
        assert n != len(points), "Все точки располагаются на одной прямой!"

        for i in range(1, n + 1):
            outer_edges[(i-1, i)] = True
            G[i-1].add(i); G[i].add(i-1)
            point_outer_edges[i-1].add(i); point_outer_edges[i].add(i-1)
        outer_edges[(0, n)] = False
        G[0].add(n); G[n].add(0)
        point_outer_edges[0].add(n); point_outer_edges[n].add(0)

        for i in range(1, n):
            G[i].add(n); G[n].add(i)
    else:
        # если краевое условие не подтвердилось, то зададим граф и МВО на первых трёх точках
        G[0].add(1); G[1].add(0)
        G[1].add(2); G[2].add(1)
        G[0].add(2); G[2].add(0)
        point_outer_edges[0] |= {1, 2}
        point_outer_edges[1] |= {0, 2}
        point_outer_edges[2] |= {0, 1}
        v1, v2 = get_vector(points[0], points[1]), get_vector(points[0], points[2])
        outer_edges[(0, 1)] = EPS < -vectors_sin(v1, v2) <= 1
        outer_edges[(1, 2)] = outer_edges[(0, 1)]
        outer_edges[(0, 2)] = not outer_edges[(0, 1)]
        n = 3

    for i in range(n, len(points)):
        # Добавим треугольники, образованные видимыми ребрами и самой точкой
        # (то есть добавим ребра из рассматриваемой точки во все концы видимых ребер).
        new_edges = [(i - 1, i)]
        hidden_edges = []
        start_adjacent_vertices = list(point_outer_edges[i - 1])

        for curr_node, next_node in [(i - 1, start_adjacent_vertices[1]), (i - 1, start_adjacent_vertices[0])]:
            flag = False
            while True:
                idx1, idx2 = in_order(curr_node, next_node)
                p1, p2, m = points[idx1], points[idx2], points[i]
                vector_m_p1 = get_vector(m, p1)
                vector_p1_p2 = get_vector(p1, p2)
                if not outer_edges[(idx1, idx2)]:
                    vector_p1_p2 = reverse_vector(vector_p1_p2)
                _sin = vectors_sin(vector_m_p1, vector_p1_p2)
                if not (EPS < _sin <= 1):
                    point_outer_edges[curr_node].add(i)
                    point_outer_edges[i].add(curr_node)
                    outer_edges[(curr_node, i)] = (
                        outer_edges[(idx1, idx2)] if (curr_node == idx2)
                        else not outer_edges[(idx1, idx2)])
                    break

                # print(f"Вершина {i} видит ребро {idx1, idx2}")
                new_edges.append((next_node, i))
                del outer_edges[(idx1, idx2)]
                point_outer_edges[idx1].remove(idx2)
                point_outer_edges[idx2].remove(idx1)
                hidden_edges.append((idx1, idx2))

                if len(outer_edges) == 1:
                    flag = True
                    idx = list(point_outer_edges[next_node])[0]
                    idx1, idx2 = in_order(idx, next_node)
                    new_edges.append((idx, i))
                    point_outer_edges[next_node].add(i)
                    point_outer_edges[i].add(next_node)
                    outer_edges[(next_node, i)] = (
                        outer_edges[(idx1, idx2)] if (next_node == idx2)
                        else not outer_edges[(idx1, idx2)])
                    point_outer_edges[idx].add(i)
                    point_outer_edges[i].add(idx)
                    outer_edges[(idx, i)] = (
                        outer_edges[(idx1, idx2)] if (idx == idx2)
                        else not outer_edges[(idx1, idx2)])
                    break

                curr_node, next_node = next_node, list(point_outer_edges[next_node] - {i-1})[0]

            if flag:
                break

        for idx, _ in new_edges:
            G[idx].add(i)
            G[i].add(idx)

        # для проверки условия Делоне создадим очередь, в которую будем добавлять
        # кортежи по 3 точки из 4-х угольника
        queue = deque([(i, idx1, idx2) for idx1, idx2 in hidden_edges])
        while queue:
            idx1, ed_idx1, ed_idx2 = queue.popleft()
            p1, edge_p1, edge_p2 = points[idx1], points[ed_idx1], points[ed_idx2]
            idx2 = get_other_corner_point(G, ed_idx1, ed_idx2, i)

            if idx2 is None:
                continue

            p2 = points[idx2]
            if not check_Delaunay_condition(edge_p1, edge_p2, p1, p2):
                # если условие Делоне не выпоняется, то меняем расположение внутреннего ребра в 4-х угольнике
                # и добавим в очередь на проверку смежные 3-х угольники, смежные с получившимися 3-х угольниками,
                # которые образуют 4-х угольники
                G[ed_idx1].remove(ed_idx2)
                G[ed_idx2].remove(ed_idx1)
                G[idx1].add(idx2)
                G[idx2].add(idx1)
                queue.append((idx1, *in_order(idx2, ed_idx1)))
                queue.append((idx1, *in_order(idx2, ed_idx2)))

    # по получившейся таблице смежности G получим треугольники и добавим их во множество треугольников
    triangles = set()
    # будем сохранять множество точек, которые уже были просмотрены
    visited_points = set()
    # зададим очередь для точек, которые надо посетить
    # при правильной работе программы, граф получится связанным
    points_to_visit = deque([0])
    while points_to_visit:
        a = points_to_visit.popleft()
        for b in G[a]:
            if b not in visited_points:
                points_to_visit.append(b)
            for c in G[a] & G[b]:
                triangles.add(in_order(a, b, c))
        visited_points.add(a)

    return list(triangles)


if __name__ == '__main__':
    def get_rand_coordinate():
        return CANVAS_PADDING + random.random() * CANVAS_M_PADDING

    points = [Point(get_rand_coordinate(),
                    get_rand_coordinate(),
                    get_rand_coordinate()) for _ in range(POINTS_TO_GEN)]
    points.sort(key=lambda point: (point.x, point.y))

    points_unique = [points[0]]
    for i in range(1, len(points)):
        if points[i] != points[i - 1]:
            points_unique.append(points[i])

    points = points_unique
    print("Уникальных точек:", len(points))

    triangles = triangulate(points)
    print("Число треугольников в полученной триангуляции:", len(triangles))

    # Выведем получившуюся триангуляцию
    vp = np.full((CANVAS_WIDTH, CANVAS_WIDTH, 3), GRAY, dtype='uint8')
    for point in points:
        x, y = int(point.x), int(point.y)
        vp[y - 2:y + 3, x - 2:x + 3] = COLOR
    for idx1, idx2, idx3 in triangles:
        x1, y1 = int(points[idx1].x), int(points[idx1].y)
        x2, y2 = int(points[idx2].x), int(points[idx2].y)
        x3, y3 = int(points[idx3].x), int(points[idx3].y)
        vp[y1, x1] = COLOR
        vp[y2, x2] = COLOR
        Bresenhams_line_algorithm(vp, (x1, y1), (x2, y2))
        Bresenhams_line_algorithm(vp, (x1, y1), (x3, y3))
        Bresenhams_line_algorithm(vp, (x2, y2), (x3, y3))
    plt.axis('off')
    plt.imshow(vp)
    plt.show()
