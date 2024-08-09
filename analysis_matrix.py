import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import stats


def calculate_err(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Векторы должны быть одинаковой длины.")
    np_vector1 = np.array(vector1)
    np_vector2 = np.array(vector2)
    return np.sqrt(np.sum((np_vector1 - np_vector2) ** 2))



if __name__ == '__main__':

    # # path_folder = 'C:/Users/22354/PycharmProjects/Diplom_itog/result/output2.txt'
    # # with open(path_folder, 'w', encoding='utf-8') as f:
    # #     sys.stdout = f
    # path_result3 = f'C:/Users/22354/PycharmProjects/Diplom_itog/result/norms_matrix.txt'
    # norms_matrix = np.loadtxt(path_result3)
    # norms_matrix = norms_matrix[5:]
    #
    # # mean = np.mean(norms_matrix)
    # # std_dev = np.std(norms_matrix)
    # # normalized_norms = (norms_matrix - mean) / std_dev
    # #
    # min_val = np.min(norms_matrix)
    # max_val = np.max(norms_matrix)
    # normalized_norms = (norms_matrix - min_val) / (max_val - min_val)
    #
    # # normalized_norms, lambda_param = stats.boxcox(norms_matrix)
    # # normalized_norms = np.arcsinh(norms_matrix)
    # # normalized_norms = np.log(normalized_norms)
    #
    # # normalized_norms = normalized_norms*100
    #
    # # plt.figure(figsize=(15, 6))
    # #
    # matrix3 = np.empty((0, ))
    # for i in range(1,21):
    #     path_result = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_{i}.3'
    #     path_result2 = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_labls_{i}.2'
    #     matrix = np.loadtxt(path_result)
    #     matrix2 = np.loadtxt(path_result2)
    #     deviations = []
    #     first_row = matrix[0, :]
    #     for i in range(6, matrix.shape[0]):
    #         deviation = calculate_err(first_row, matrix[i, :])
    #         deviations.append(deviation)
    #     plt.scatter(normalized_norms, deviations, color='blue')
    #
    # plt.xlabel('Норма матрицы')
    # plt.ylabel('Отклонение от оригинального изображения')
    # plt.title(f'Точечный график зависимости отклонения мультифрактального спектра от нормы матрицы проектвного искажения')
    # filtered_numbers = normalized_norms[normalized_norms >= 0.04]
    # plt.xticks(filtered_numbers, rotation = 'vertical')
    # plt.grid(True)
    # plt.show()

#ПРОСТО РАССКОММЕНТИРОВАТЬ ТО ЧТО НИЖЕ
    for i in range(1,27):

        # ГРАФИКИ РАЗМЕРНОСТЬ ОТ ЭКСПОНЕНТ ГЕЛЬДЕРА
        path_result = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_{i}.3'
        path_result2 = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_labls_{i}.3'
        matrix = np.loadtxt(path_result)
        matrix2 = np.loadtxt(path_result2)
        # matrix = [row[1:] for row in matrix]
        # matrix2 = [row[1:] for row in matrix2]
        fig = plt.figure(figsize=(1900 / 100, 933 / 100))  # Размеры в дюймах, переведенные в сотые доли

        for j in range(1, len(matrix)):
            vector = np.zeros(len(matrix2[i-1]) - 1)
            for x in range(len(vector)-1):
                vector[x] = (matrix2[i-1][x] + matrix2[i-1][x + 1]) / 2
            plt.scatter(matrix2[0], matrix[j], color='blue')
        plt.scatter(matrix2[0], matrix[0], color='red',
                    label="Значения мультифрактального спектра исходного изображения")

        plt.legend()
        plt.xlabel('Экспонента Гельдера\n', fontsize=16)
        plt.ylabel('\nФрактальная размерность', fontsize=16)
        weights_list = [
            [1 / 3, 1 / 3, 1 / 3],  # Стандартный треугольник
            [0.5, 0.25, 0.25],
            [0.6, 0.3, 0.1],
            [0.15, 0.7, 0.15],
            [0.15, 0.8, 0.05],
            [0.05, 0.05, 0.9],
            [0.3, 0.1, 0.6]
        ]
        if i == 1:
            plt.title(f'Точечный график для образца {i}', fontsize=16)
        else:
            plt.title(f'Точечный график для для образца {i}',fontsize=16)
        plt.xticks(matrix2[0])
        plt.gca().add_patch(patches.Rectangle((0.85, 0.75), 0.1, 0.1, fill=True, color='red', alpha=0.5))
        # Показать график
        plt.grid(True)
        plt.savefig(f'C:/Users/22354/PycharmProjects/Diplom_itog/result/graphs/График_obrazew_{i}.png', dpi=192)

#ХЕРНЯ СНИЗУ

    #
    #         matrix = [row[1:] for row in matrix]
    #         max_non_zero_count=0
    #         for row in matrix:
    #             non_zero_count = sum(1 for element in row if element != 0)
    #             max_non_zero_count = max(max_non_zero_count, non_zero_count)
    #
    #         for i in range(len(matrix)):
    #             matrix[i] = matrix[i][:max_non_zero_count]
    #
    #         np.set_printoptions(linewidth=np.inf)
    #
    #         print(f'Мультифрактальный спектр исходного изображения {i}')
    #         print(matrix[0])
    #         print('Мультифрактальные спектры проективных искажений')
    #         for j in range (1, len(matrix)):
    #             row_j = matrix[j]
    #             #non_zero_elements = row_j[row_j != 0]
    #             print(matrix[j])
    #
    #         max_err = float('-inf')
    #         min_err = float('inf')
    #         for x in range(len(matrix) - 1):
    #             for y in range(x + 1, len(matrix)):
    #                 err = calculate_err(matrix[x], matrix[y])
    #                 if err > max_err:
    #                     max_err = err
    #                     max_i, max_j = x, y
    #                 if err < min_err:
    #                     min_err = err
    #                     min_i, min_j = x, y
    #
    #         print(f'Максимальное отличие между мультифрактальными спектрами = {max_err} . Это между векторами {max_i} и {max_j}')
    #         print(f'Минимальное отличие между мульттифракатльными спектрами = {min_err} . Это между векторами {min_i} и {min_j}')
    #
    #         print("")
    # sys.stdout = sys.__stdout__