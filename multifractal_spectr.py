import os

import cv2
import numpy as np

import exponent_heldera
import fractal_dimension


def divide_into_regions(matrix, n=10):
    # Проверка, что матрица не пуста
    if not matrix.any():
        raise ValueError("Пустая матрица")

    thresholds = np.linspace(matrix.min(), matrix.max(), num=n)
    regions = np.zeros_like(matrix, dtype=np.uint8)

    for i in range(1, len(thresholds)):
        lower_threshold = thresholds[i - 1]
        upper_threshold = thresholds[i]
        regions[(matrix >= lower_threshold) & (matrix < upper_threshold)] = i

    return (regions, thresholds)


def calculate_multifractal_spectr(image, matrix_heldera, path_folder =""):
    # разделяем изображение
    matrix_region, thresholds = divide_into_regions(matrix_heldera)

    unique_labels = np.unique(matrix_region)
    fractal_dimensions = np.zeros(len(unique_labels))

    # Рассчет фрактальной размерности для каждой области
    for i, label in enumerate(unique_labels):
        if label!=0:
            mask = (matrix_region == label)
            mask_image = image.copy()
            mask_image[~mask] = 0

            if (path_folder != ""):
                full_path = path_folder + f'mask_{i}.png'
                cv2.imwrite(full_path, mask_image)

            if i < len(fractal_dimensions):
                frcatal_image = fractal_dimension.ImageFractalDimension(mask_image)
                fractal_dimensions[i] = frcatal_image.fractal_dim
    return (fractal_dimensions, thresholds)