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
    path_folder = 'C:/Users/22354/PycharmProjects/Diplom_itog/result/output3.txt'
    with open(path_folder, 'w', encoding='utf-8') as f:
        sys.stdout = f
        glob_max_err = float('-inf')
        glob_min_err = float('inf')

        for i in range (1,21):
            path_result = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_{i}.2'
            # path_result2 = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_labls_{i}'
            matrix = np.loadtxt(path_result)

            matrix = [row[1:] for row in matrix]
            max_non_zero_count=0
            for row in matrix:
                non_zero_count = sum(1 for element in row if element != 0)
                max_non_zero_count = max(max_non_zero_count, non_zero_count)

            # for i in range(len(matrix)):
            #     matrix[i] = matrix[i][:max_non_zero_count]

            np.set_printoptions(linewidth=np.inf)
            #
            print(f'Мультифрактальный спектр исходного изображения {i}')
            print(matrix[0])
            print('Мультифрактальные спектры проективных искажений')
            for j in range (1, len(matrix)):
                row_j = matrix[j]
                #non_zero_elements = row_j[row_j != 0]
                print(matrix[j])
            #
            max_err = float('-inf')
            min_err = float('inf')
            for x in range(len(matrix) - 1):
                for y in range(x + 1, len(matrix)):
                    err = calculate_err(matrix[x], matrix[y])
                    if err > max_err:
                        max_err = err
                        max_i, max_j = x, y
                    if err < min_err and err!=0:
                        min_err = err
                        min_i, min_j = x, y
            if max_err > glob_max_err:
                glob_max_err = max_err
            if min_err < glob_min_err:
                glob_min_err = min_err

            print(f'Максимальное отличие между мультифрактальными спектрами = {max_err} . Это между векторами {max_i} и {max_j}')
            print(f'Минимальное отличие между мульттифракатльными спектрами = {min_err} . Это между векторами {min_i} и {min_j}')

            print("")
            print(glob_max_err)
            print(glob_min_err)
    sys.stdout = sys.__stdout__