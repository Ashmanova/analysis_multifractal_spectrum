# Работа с новыми данными
import numpy as np
import matplotlib.pyplot as plt

def calculate_err(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Векторы должны быть одинаковой длины.")
    np_vector1 = np.array(vector1)
    np_vector2 = np.array(vector2)
    return np.sqrt(np.sum((np_vector1 - np_vector2) ** 2))

# Инициализация списка для хранения векторов различий
difference_vectors = []

path_result3 = f'C:/Users/22354/PycharmProjects/Diplom_itog/result/norms_matrix3.txt'
norms_matrix = np.loadtxt(path_result3)
print(len(norms_matrix))
plt.figure(figsize=(10, 5))

print(norms_matrix)
# norms_matrix = norms_matrix[1:]
# norms_matrix = np.delete(norms_matrix, 3)
# norms_matrix = np.delete(norms_matrix, 14)



min_val = np.min(norms_matrix)
max_val = np.max(norms_matrix)
normalized_norms2 = (norms_matrix - min_val) / (max_val - min_val)
print(min_val, max_val)

max_err = float('-inf')
min_err = float('inf')
max_i, max_j = 0, 0
min_i, min_j = 0, 0

alpha = 0.2
normalized_norms = np.power(norms_matrix, alpha)
print(normalized_norms)

all_current_differences = []


# Цикл по файлам
for i in range(1, 27):  #
    file_path = f"C:/Users/22354/PycharmProjects/Diplom_itog/image/{i}/data/result_{i}.3"

    matrix_multifractal = np.loadtxt(file_path)
    # Вектор различий для текущей матрицы
    current_differences = []
    # Первая строка матрицы
    first_row = matrix_multifractal[0, :]
    # Сравнение каждой строки с первой строкой
    j=1
    for row in matrix_multifractal[1:, :]:  # Пропускаем первую строку
        # Вычисление разности и добавление результата в список
        j += 1
        difference = calculate_err(first_row, row)
        current_differences.append(difference)

        if difference<=min_err:
            min_err = difference
            min_i, min_j = i, j
        if difference >= max_err:
            max_err = difference
            max_i, max_j = i, j


    difference_vectors.append(current_differences)
    # current_differences = current_differences[1:]
    # current_differences = np.delete(current_differences, 3)
    # current_differences = np.delete(current_differences, 14)
    all_current_differences.extend(current_differences)
    plt.scatter(normalized_norms, current_differences, label=f'Образец № {i}')

plt.grid(True, zorder=0)

all_current_differences.sort()

# filtered_numbers = normalized_norms[normalized_norms >= 0.020] #5.72664284
normalized_norms = np.delete(normalized_norms, 7)
normalized_norms = np.delete(normalized_norms, 10)
norms_matrix = np.delete(norms_matrix, 7)
norms_matrix = np.delete(norms_matrix, 10)
plt.xticks(normalized_norms,labels=np.round(norms_matrix, 3), rotation = 'vertical')
plt.axvline(x=1.18740386, color='darkgray', linestyle='-', linewidth=0.8)



threshold = 0.07
filtered_differences = [all_current_differences[0]]
for value in all_current_differences[1:]:
    if value - filtered_differences[-1] > threshold:
        filtered_differences.append(value)

plt.yticks(filtered_differences)



plt.ylabel('Значение отклонения от эталонного\nмультифракатльного спектра\n', fontsize=14)
plt.xlabel('\nНорма матрицы проективного искажения', fontsize=14)
plt.title('Точечная диаграмма отклонения\nмультифракатального спектра в зависимости от\nпроективного искажения для кристаллических образцов', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




print(f'Максимальное отличие между мультифрактальными спектрами = {max_err} для образца {max_i} между оригинальным образцом и искажением {max_j}')
print(f'Минимальное отличие между мульттифракатльными спектрами = {min_err} для образца {min_i} между оригинальным образцом и искажением {min_j}')


