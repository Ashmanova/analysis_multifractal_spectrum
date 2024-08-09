# Средняя корреляция
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

base_path = 'C:/Users/22354/PycharmProjects/Diplom_itog'
num_samples = 20  # Количество образцов
num_distortions = 25  # Количество проективных искажений на каждый образец

# Для хранения общих результатов корреляции, с лейблами
overall_correlations = pd.DataFrame()

for i in range(1, num_samples + 1):
    path_norms = f'{base_path}/result/norms_matrix.txt'
    norms = np.loadtxt(path_norms)

    path_result = f'{base_path}/image/{i}/data/result_{i}.2'
    spectra = np.loadtxt(path_result)
    original_spectrum = spectra[0, :]
    distorted_spectra = spectra[1:, :]

    data = pd.DataFrame(distorted_spectra - original_spectrum,
                        columns=[f'Параметр {j + 1}' for j in range(distorted_spectra.shape[1])])
    data['Норма искажения'] = norms

    # Рассчитываем корреляцию
    correlation_matrix = data.corr()['Норма искажения'].drop('Норма искажения')  # Получаем корреляции только для параметров с нормами
    overall_correlations[f'Образец {i}'] = correlation_matrix

# Транспонируем для правильной визуализации
overall_correlations = overall_correlations.T

# Средние значения корреляции по всем образцам
mean_correlations = overall_correlations.mean()
print("Средняя корреляция по всем образцам:")
print(mean_correlations)

# Построение тепловой карты
plt.figure(figsize=(14, 10))
sns.heatmap(overall_correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта корреляции между параметрами и нормами искажений')
plt.xlabel('Параметры')
plt.ylabel('Образцы')
plt.show()
