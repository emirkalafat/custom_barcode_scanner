import cv2
import numpy as np

# EAN-13 kodlama kuralları
LEFT_ODD = {
    "0001101": 0,
    "0011001": 1,
    "0010011": 2,
    "0111101": 3,
    "0100011": 4,
    "0110001": 5,
    "0101111": 6,
    "0111011": 7,
    "0110111": 8,
    "0001011": 9,
}
LEFT_EVEN = {
    "0100111": 0,
    "0110011": 1,
    "0011011": 2,
    "0100001": 3,
    "0011101": 4,
    "0111001": 5,
    "0000101": 6,
    "0010001": 7,
    "0001001": 8,
    "0010111": 9,
}
RIGHT = {
    "1110010": 0,
    "1100110": 1,
    "1101100": 2,
    "1000010": 3,
    "1011100": 4,
    "1001110": 5,
    "1010000": 6,
    "1000100": 7,
    "1001000": 8,
    "1110100": 9,
}

# Barkodun yapısal işaretçileri
START_GUARD = "101"
CENTER_GUARD = "01010"
END_GUARD = "101"


def resize_image(image, width):
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def binarize_image(image, threshold=128):
    _, binarized_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binarized_image


def find_barcode_bounds(image):
    height, width = image.shape
    start_col, end_col = 0, width - 1

    column_means = np.mean(image, axis=0)

    for i in range(width):
        if column_means[i] < 250:
            start_col = i
            break

    for i in range(width - 1, -1, -1):
        if column_means[i] < 250:
            end_col = i
            break

    return start_col, end_col


def calculate_bar_widths(image, expected_bar_width):
    height, width = image.shape
    center_y = height // 2
    start_y = max(center_y - 32, 0)
    end_y = min(center_y + 32, height)

    middle_strip = image[start_y:end_y, :]
    bar_widths = []
    current_width = 0
    current_color = middle_strip[0, 0]

    for i in range(middle_strip.shape[1]):
        if middle_strip[0, i] == current_color:
            current_width += 1
        else:
            bar_widths.append(current_width)
            current_color = middle_strip[0, i]
            current_width = 1
    bar_widths.append(current_width)

    normalized_widths = [
        width * expected_bar_width / np.mean(bar_widths) for width in bar_widths
    ]

    return normalized_widths


def decode_bars(bar_widths):
    def match_pattern(bars, patterns):
        bar_str = "".join(["1" if bar > np.mean(bars) else "0" for bar in bars])
        return patterns.get(bar_str)

    # Barkod başlangıcı ve sonunu kontrol et
    start_pattern = "".join(
        ["1" if bar_widths[i] > np.mean(bar_widths) else "0" for i in range(3)]
    )
    end_pattern = "".join(
        ["1" if bar_widths[-3 + i] > np.mean(bar_widths) else "0" for i in range(3)]
    )

    if start_pattern != START_GUARD or end_pattern != END_GUARD:
        return "Geçersiz Barkod"

    # Sol ve sağ tarafları ayır
    left_side = bar_widths[3:45]
    right_side = bar_widths[50:-3]
    center_pattern = "".join(
        ["1" if bar_widths[45 + i] > np.mean(bar_widths) else "0" for i in range(5)]
    )

    if center_pattern != CENTER_GUARD:
        return "Geçersiz Orta Barkod"

    # Sol tarafı çöz
    left_digits = []
    for i in range(0, len(left_side), 7):
        odd_digit = match_pattern(left_side[i : i + 7], LEFT_ODD)
        even_digit = match_pattern(left_side[i : i + 7], LEFT_EVEN)
        if odd_digit is not None:
            left_digits.append(odd_digit)
        elif even_digit is not None:
            left_digits.append(even_digit)
        else:
            return "Geçersiz Sol Barkod"

    # Sağ tarafı çöz
    right_digits = []
    for i in range(0, len(right_side), 7):
        digit = match_pattern(right_side[i : i + 7], RIGHT)
        if digit is None:
            return "Geçersiz Sağ Barkod"
        right_digits.append(digit)

    # 13 basamaklı barkod numarasını birleştir
    return left_digits + right_digits


# Barkod görüntüsünü yükle
image_path = "test_barcode.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resmi yeniden ölçeklendir
resized_image = resize_image(original_image, 950)

# Görüntüyü binarize et
binarized_image = binarize_image(resized_image)

# Beyaz boşlukları tespit et ve barkod alanını kırp
start_col, end_col = find_barcode_bounds(binarized_image)
cropped_image = binarized_image[:, start_col : end_col + 1]

# Bar genişliklerini hesapla
expected_bar_width = 10
bar_widths = calculate_bar_widths(cropped_image, expected_bar_width)

# Barkodu çöz
decoded_barkod = decode_bars(bar_widths)
print("Çözülen Barkod:", decoded_barkod)
