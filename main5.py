import os

import numpy as np
import cv2
import exponent_heldera
import multifractal_spectr


def projective_transformation (image_path, path_folder):
    image = cv2.imread(image_path)
    src_points = np.float32( [[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
    dst_points_list = [
        np.float32([[650, 200], [image.shape[1] - 100, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 100, 900], [0, image.shape[0] - 90], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 0], [350, image.shape[0] - 900], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 700], [0, image.shape[0] - 1], [image.shape[1] - 350, image.shape[0] - 600]]),
        np.float32([[280, 100], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 650, image.shape[0] - 480]]),

        np.float32([[0, 0], [image.shape[1] - 1, 0], [350, image.shape[0] - 900], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 0], [250, image.shape[0] - 450], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 0], [200, image.shape[0] - 100], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[20, 20], [image.shape[1], 250], [0, image.shape[0] - 50], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 300], [0, image.shape[0] - 300], [image.shape[1] - 1, image.shape[0] - 300]]),
        np.float32([[0, 0], [image.shape[1] - 50, 450], [0, image.shape[0] - 45], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 150], [0, image.shape[0] - 1], [image.shape[1] - 150, image.shape[0] - 300]]),
        np.float32([[0, 0], [image.shape[1] - 100, 900], [0, image.shape[0] - 90], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[0, 0], [image.shape[1] - 1, 500], [0, image.shape[0] - 1], [image.shape[1] - 150, image.shape[0] - 400]]),
        np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 650, image.shape[0] - 480]]),

        np.float32([[50, 30], [image.shape[1] - 1, 0], [50, image.shape[0] - 1], [image.shape[1] - 100, image.shape[0] - 1]]),
        np.float32(
            [[150, 110], [image.shape[1] - 1, 50], [100, image.shape[0] - 200], [image.shape[1] - 1, image.shape[0] - 1]]),
        np.float32([[100, 100], [image.shape[1] - 50, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 50]]),
        np.float32(
            [[100, 50], [image.shape[1] - 50, 20], [50, image.shape[0] - 10], [image.shape[1] - 20, image.shape[0] - 50]]),
        np.float32(
            [[50, 200], [image.shape[1] - 1, 100], [0, image.shape[0] - 50], [image.shape[1] - 1, image.shape[0] - 200]]),

        np.float32([[50, 150], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 0]]),
        np.float32(
            [[150, 0], [image.shape[1] - 170, 50], [50, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 100]]),
        np.float32([[90, 0], [image.shape[1] - 1, 0], [75, image.shape[0] - 180], [image.shape[1] - 135, image.shape[0] - 1]]),
        np.float32([[0, 125], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 170, image.shape[0] - 50]]),
        np.float32([[50, 0], [image.shape[1] - 200, 100], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]]),

    ]

    result = np.empty(25)
    for i, dst_points in enumerate(dst_points_list):
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        norm_matrix = np.linalg.norm(matrix)
        result[i]=norm_matrix
        print (norm_matrix)
        # result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
        # output_path = path_folder + f'/transformed_{i}.jpg'
        # cv2.imwrite(output_path, result)
    np.savetxt(f'C:/Users/22354/PycharmProjects/Diplom_itog/result/norms_matrix.txt', result, fmt='%.18e')


def matrix_heldera_saved(path_image, output_path):
    image = cv2.imread(path_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(gray_image, (64, 64))
    matrix_heldera = exponent_heldera.exp_heldera(small_image)
    np.savetxt(output_path, matrix_heldera, fmt='%.18e')

if __name__ == '__main__':
    for i in range (1, 27):
        print(f'Рассчет мультифрактальный спектров для образца {i}')
        # Создаем семейство проективных искажений для каждого исходного изображения
        path_image = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/{i}.PNG'
        path_folder = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}'
        # projective_transformation(path_image, path_folder)

        # Рассчитываем эскпоненты Гельдера
        #
        #
        # for j in range (14,26):
        #     print(j)
        #     path_image_transformed = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/transformed_{j}.jpg'
        #     output_path = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/matrix_heldera_trasformed_{j}'
        #     matrix_heldera_saved(path_image_transformed, output_path)

        # Рассчитываем мультифрактальный спектр
        result = np.empty((0, 10))
        result2 = np.empty((0, 10))
        matrix_heldera = np.loadtxt(f"C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/matrix_heldera_{i}.txt")
        path_mask = f"C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/mask"

        if not os.path.exists(path_mask+'/original/'):
            os.makedirs(path_mask+'/original/')

        image = cv2.imread(path_image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small_image = cv2.resize(gray_image, (64, 64))

        multifractal, lables = multifractal_spectr.calculate_multifractal_spectr(small_image, matrix_heldera,path_mask + '/original/')
        result = np.vstack((result, multifractal))
        result2 = np.vstack((result2, lables))

        for j in range(1,26):
            path_image_transformed = f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/transformed_{j}.jpg'
            image = cv2.imread(path_image_transformed)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            small_image = cv2.resize(gray_image, (64, 64))
            # matrix_heldera = np.loadtxt(f"C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/matrix_heldera_trasformed_{j}")
            if not os.path.exists(path_mask + f'/trasformed{j}/'):
                os.makedirs(path_mask + f'/trasformed{j}/')
            multifractal, lables = multifractal_spectr.calculate_multifractal_spectr(small_image, matrix_heldera,
                                                                                     path_mask + f'/trasformed{j}/')
            result = np.vstack((result, multifractal))
            result2 = np.vstack((result2, lables))
        result = np.nan_to_num(result, nan=0.0)

        print('Мультифракталььный спектр для оригинального изображения')
        print(result[0])
        print('Мультифракатльные спектры искаженных изображений')
        for j in range(1, len(result)):
            row_j = result[j]
            #non_zero_elements = row_j[row_j != 0]
            print(result[j])

        # np.savetxt(f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_{i}.3', result, fmt='%.18e')
        # np.savetxt(f'C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_labls_{i}.3', result2, fmt='%.18e')






