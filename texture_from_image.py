import cv2
import numpy as np

# Ширина обрабатываемых краёв в процентах от размера изображения (ширины и высоты, соответственно)
BORDER_WIDTH_PERCENT_VERT = 6
BORDER_WIDTH_PERCENT_HORZ = 5

input = cv2.imread('images/texture.png')

def texture_repeat(input):
    duplicated = cv2.repeat(input, 3, 3)
    return duplicated


# Размеры изображения
h, w, c = input.shape

# Определяем центральные зоны для сглаживания швов
border_size_vert = int(BORDER_WIDTH_PERCENT_VERT / 100 * w)
border_size_horz = int(BORDER_WIDTH_PERCENT_HORZ / 100 * h)

# Начнём с вертикальных швов
# Сначала создадим 2 новых тайла, сдвинутых по горизонтали
shifted_horz_left = np.roll(input, shift = w // 2 + border_size_vert, axis = 1)
shifted_horz_right = np.roll(input, shift = w // 2 - border_size_vert, axis = 1)

# Создаём градиентную альфа-маску для плавного перехода
horiz_gradient = np.tile(np.linspace(0.0, 1.0, 2 * border_size_vert).reshape(1, -1, 1), (h, 1, 3))

alpha_mask_horz = np.zeros((h, w, 3))

alpha_mask_horz[:, w // 2 - border_size_vert: w // 2 + border_size_vert, :] = horiz_gradient
alpha_mask_horz[:, w // 2 + border_size_vert:, :] = 1

# Обратная маска
alpha_mask_horz_reverted = 1 - alpha_mask_horz

# Применяем маску для получения правой части
half_right = cv2.multiply(alpha_mask_horz, shifted_horz_right.astype(np.float32), dtype=cv2.CV_32F)

# Применяем обратную маску для получения левой части
half_left = cv2.multiply(alpha_mask_horz_reverted, shifted_horz_left.astype(np.float32), dtype=cv2.CV_32F)

# Суммируем полученные части
outImage_vert_blend = cv2.add(half_right, half_left, dtype=cv2.CV_32F)

# Сделаем то же самое для горизонтальных швов
# Убираем горизонтальные швы
shifted_vert_top = np.roll(outImage_vert_blend, shift = h // 2 + border_size_horz, axis = 0)
shifted_vert_bottom = np.roll(outImage_vert_blend, shift = h // 2 - border_size_horz, axis = 0)

#
vert_gradient = np.tile(np.linspace(0.0, 1.0, 2 * border_size_horz).reshape(-1, 1, 1), (1, w, 3))

alpha_mask_vert = np.zeros((h, w, 3))
#
alpha_mask_vert[h // 2 - border_size_horz : h // 2 + border_size_horz, :, :] = vert_gradient
alpha_mask_vert[h // 2 + border_size_horz :, :, :] = 1

alpha_mask_vert_reverted = 1 - alpha_mask_vert

# Применяем обратную маску для получения верхней части
half_top = cv2.multiply(alpha_mask_vert_reverted, shifted_vert_top.astype(np.float32), dtype=cv2.CV_32F)

# Применяем маску для получения нижней части
half_bottom = cv2.multiply(alpha_mask_vert, shifted_vert_bottom.astype(np.float32), dtype=cv2.CV_32F)

# Суммируем полученные части
outImage_blend = cv2.add(half_top, half_bottom, dtype=cv2.CV_32F)

# Возвращаем тайл в изначальное полжение
original_tile_processed = np.roll(outImage_blend, shift= (h // 2, w // 2), axis = (0, 1))

output_texture_repeat = np.uint8(texture_repeat(original_tile_processed))

# Отобразить изображение
cv2.imshow('Result', output_texture_repeat)
# Дождаться нажатия клавиши и закрыть окно
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранить изображение

cv2.imwrite('output_texture.png', output_texture_repeat)
