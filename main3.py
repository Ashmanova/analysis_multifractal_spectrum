# Составление общего датасета по всем образцам
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def calculate_err(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Векторы должны быть одинаковой длины.")
    np_vector1 = np.array(vector1)
    np_vector2 = np.array(vector2)
    return np.sqrt(np.sum((np_vector1 - np_vector2) ** 2))

pd.set_option('display.precision', 12)

base_path = 'C:/Users/22354/PycharmProjects/Diplom_itog'
num_samples = 26
lables = np.loadtxt('C:/Users/22354/PycharmProjects/Diplom_itog/image/1/data/result_labls_1.3')
print(lables[0])

all_data = pd.DataFrame()
path_norms = f'{base_path}/result/norms_matrix.txt'
norms = np.loadtxt(path_norms)
norms_with_original = np.insert(norms, 0, 1)
lablels = np.loadtxt(f'C:/Users/22354/PycharmProjects/Diplom_itog/image/1/data/result_labls_1.3')

for i in range(1, num_samples + 1):
    path_result = f'{base_path}/image/{i}/data/result_{i}.3'
    spectra = np.loadtxt(path_result)
    data = pd.DataFrame(spectra, columns=[f'Параметр {j + 1}' for j in range(spectra.shape[1])])
    # data = pd.DataFrame(spectra, columns=[f'Параметр {j}' for j in lablels[0]])


    deviations = []
    first_row = spectra[0, :]
    for i in range(0, spectra.shape[0]):
        deviation = calculate_err(first_row, spectra[i, :])
        deviations.append(deviation)
    data['Норма искажения'] = norms_with_original
    all_data = pd.concat([all_data, data], ignore_index=True)

print(all_data.head(30))
correlation_matrix = all_data.corr()
# norm_deviation_correlation = correlation_matrix.at['Норма искажения', 'Отклонение']
# print("\nКорреляция 'Норма искажения' и 'Отклонение':", norm_deviation_correlation)
for j in range(1,11):  # Для каждого параметра в данных
    param_name = f'Параметр {j + 1}'
    norm_param_correlation = correlation_matrix.at[param_name, 'Норма искажения']
    print(f"Корреляция '{param_name}' и 'Норма искажения': {norm_param_correlation}")
