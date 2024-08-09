def projective_transformation (image_path, path_folder):
    image = cv2.imread(image_path)
    src_points = np.float32( [[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
    dst_points_list = [
        np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
        # np.float32([[650, 200], [image.shape[1] - 100, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 100, 900], [0, image.shape[0] - 90], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 0], [350, image.shape[0] - 900], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 700], [0, image.shape[0] - 1], [image.shape[1] - 350, image.shape[0] - 600]]),
        # np.float32([[280, 100], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 650, image.shape[0] - 480]]),
        #
        # np.float32([[0, 0], [image.shape[1] - 1, 0], [350, image.shape[0] - 900], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 0], [250, image.shape[0] - 450], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 0], [200, image.shape[0] - 100], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[20, 20], [image.shape[1], 250], [0, image.shape[0] - 50], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 300], [0, image.shape[0] - 300], [image.shape[1] - 1, image.shape[0] - 300]]),
        # np.float32([[0, 0], [image.shape[1] - 50, 450], [0, image.shape[0] - 45], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 150], [0, image.shape[0] - 1], [image.shape[1] - 150, image.shape[0] - 300]]),
        # np.float32([[0, 0], [image.shape[1] - 100, 900], [0, image.shape[0] - 90], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 500], [0, image.shape[0] - 1], [image.shape[1] - 150, image.shape[0] - 400]]),
        # np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 650, image.shape[0] - 480]]),
        #
        # np.float32([[50, 30], [image.shape[1] - 1, 0], [50, image.shape[0] - 1], [image.shape[1] - 100, image.shape[0] - 1]]),
        # np.float32(
        #     [[150, 110], [image.shape[1] - 1, 50], [100, image.shape[0] - 200], [image.shape[1] - 1, image.shape[0] - 1]]),
        # np.float32([[100, 100], [image.shape[1] - 50, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 50]]),
        # np.float32(
        #     [[100, 50], [image.shape[1] - 50, 20], [50, image.shape[0] - 10], [image.shape[1] - 20, image.shape[0] - 50]]),
        # np.float32(
        #     [[50, 200], [image.shape[1] - 1, 100], [0, image.shape[0] - 50], [image.shape[1] - 1, image.shape[0] - 200]]),
        #
        # np.float32([[50, 150], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 0]]),
        # np.float32(
        #     [[150, 0], [image.shape[1] - 170, 50], [50, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 100]]),
        # np.float32([[90, 0], [image.shape[1] - 1, 0], [75, image.shape[0] - 180], [image.shape[1] - 135, image.shape[0] - 1]]),
        # np.float32([[0, 125], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 170, image.shape[0] - 50]]),
        # np.float32([[50, 0], [image.shape[1] - 200, 100], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]]),

    ]

    # result = np.empty(25)
    for i, dst_points in enumerate(dst_points_list):
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print(matrix)
        norm_matrix = np.linalg.norm(matrix)
        # result[i]=norm_matrix
        print (norm_matrix)
        result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
        cv2.imshow('uihkj',result )
        cv2.waitKey(0)

        # output_path = path_folder + f'/transformed_{i+1}.jpg'
        # cv2.imwrite(output_path, result)
    # np.savetxt(f'C:/Users/22354/PycharmProjects/Diplom_itog/result/norms_matrix.txt', result, fmt='%.18e')
