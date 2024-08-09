import matplotlib.pyplot as plt
import random


def draw_sierpinski_triangle(num_points, weights, filename):
    # Координаты вершин треугольника
    vertices = [(0, 0), (1, 0), (0.5, 0.866)]

    # Начальная точка (может быть любой внутри треугольника)
    x, y = 0.5, 0.5

    # Создание списка для хранения точек
    points = [(x, y)]

    # Генерация точек треугольника Серпинского
    for _ in range(num_points):
        chosen_vertex = random.choices(vertices, weights=weights, k=1)[0]
        x = (x + chosen_vertex[0]) / 2
        y = (y + chosen_vertex[1]) / 2
        points.append((x, y))

    # Разделение точек на координаты x и y для визуализации
    x_coords, y_coords = zip(*points)

    # Визуализация треугольника Серпинского
    plt.figure(figsize=(8, 8), dpi=2048)
    plt.scatter(x_coords, y_coords, s=0.1, color='black')
    plt.axis('off')  # Отключение осей для лучшей визуализации
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


# Количество точек для генерации
num_points = 15000

# Набор весов для различных вариаций треугольника Серпинского
weights_list = [
    [1 / 3, 1 / 3, 1 / 3],  # Стандартный треугольник
    [0.5, 0.25, 0.25],
    [0.6, 0.3, 0.1],
    [0.15, 0.7, 0.15],
    [0.15, 0.8, 0.05],
    [0.05, 0.05, 0.9],
    [0.3, 0.1, 0.6]
]

# Генерация и сохранение изображений
for i, weights in enumerate(weights_list):
    filename = f'C:/Users/22354/PycharmProjects/Diplom_itog/triangle/{i+1}/triangle_{i+1}.png'
    draw_sierpinski_triangle(num_points, weights, filename)
    print(f'Saved {filename}')
