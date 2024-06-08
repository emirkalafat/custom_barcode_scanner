import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from RPLCD.i2c import (
    CharLCD,
)  #!Bu import Raspberry Pi üzerinde çalışırken kullanılmalıdır.

lcd = CharLCD(i2c_expander="PCF8574", address=0x27, port=1, cols=16, rows=2, dotsize=8)
lcd.clear()


def process_data(data, n, threshold):
    index = 0
    selected_groups = []
    final_groups = []
    a = int(n / 2)
    while index + n - 1 <= len(data):
        # Take the first a values from the current index

        selected_values = data[index : index + n + a - 1]

        # Split these a values into groups of a
        groups = [selected_values[i : i + n] for i in range(a)]

        # Calculate the average of each group
        averages = [np.mean(group) for group in groups]

        # Determine which group to select based on the condition
        if min(averages) > threshold:
            selected_group = groups[np.argmax(averages)]
            final_groups.append(0)
        else:
            selected_group = groups[np.argmin(averages)]
            final_groups.append(1)

        # Add the selected group to the result list
        selected_groups.extend(selected_group)

        # Move the index to the next position after the selected group's last element
        selected_group_index = next(
            i for i, group in enumerate(groups) if np.array_equal(group, selected_group)
        )
        index += selected_group_index + n

    print("selected_groups: ", len(selected_groups))
    print("data: ", len(data))
    return final_groups


def convert_to_binary(barcode_bits):
    result = []
    for i in range(0, len(barcode_bits), 4):
        segment = barcode_bits[i : i + 4]
        # Çoğunluk esasına göre 1 veya 0 belirleme
        if segment.count("1") >= 2:
            result.append("1")
        else:
            result.append("0")
    return "".join(result)


def extract_barcode_values(data: np.ndarray):
    """Extract barcode values from the cleaned image."""
    # EAN-13 kodlama kuralları
    left_odd = {
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
    left_even = {
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
    right = {
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
    start_guard = "101"
    center_guard = "01010"
    end_guard = "101"

    # Barkod verilerini 0 ve 1'lere dönüştürün
    print("data.min(): ", data.min())
    print("data.max(): ", data.max())
    threshold = int((data.max() - data.min()) / 2)  # Eşik değeri
    print("threshold: ", threshold)
    binary_data = (data < threshold).astype(int)

    print("binary_data: ", binary_data)

    # İlk siyah barı bulana kadar beyaz alanları atlayın

    start_index = np.argmax(data < threshold)
    end_index = len(data) - np.argmax(data[::-1] < threshold) - 1

    print("start_index: ", start_index)
    print("end_index: ", end_index)

    # Barkod verilerini kırpın
    barcode_data = data[start_index : end_index + 1]
    print("barcode_data: \n", barcode_data)

    # Barkod çubuk genişliği n piksel olarak hesapla
    n = len(barcode_data) // 95
    print("n: ", n)

    binary_data = process_data(barcode_data, n, threshold)

    # Barkod verilerini birleştir
    barcode_data = np.array(binary_data)
    # barcode_data = np.array(new_barcode_data)
    barcode_bits = "".join(barcode_data.astype(str))
    print("barcode_bits: ", barcode_bits)
    print("len(barcode_bits): ", len(barcode_bits))

    # Başlangıç işaretini bul
    if not barcode_bits.startswith(start_guard):
        raise ValueError("Geçersiz başlangıç işaretçisi")

    # Bitiş işaretini bul
    if not barcode_bits.endswith(end_guard):
        raise ValueError("Geçersiz bitiş işaretçisi")

    # Orta işaretini bul
    center_index = 45

    # Sol ve sağ verileri ayır
    left_bits = barcode_bits[len(start_guard) : center_index]
    print("left_bits: ", left_bits)
    right_bits = barcode_bits[center_index + len(center_guard) : -len(end_guard)]
    print("right_bits: ", right_bits)

    # Bit gruplarını 7-bitlik gruplara ayır
    left_groups = [left_bits[i : i + 7] for i in range(0, len(left_bits), 7)]
    print("left_groups: ", left_groups)
    right_groups = [right_bits[i : i + 7] for i in range(0, len(right_bits), 7)]
    print("right_groups: ", right_groups)

    # Sol taraftaki verileri analiz et
    ean13_code = []

    for bits in left_groups:
        if bits in left_odd:
            ean13_code.append(left_odd[bits])
        elif bits in left_even:
            ean13_code.append(left_even[bits])
        else:
            raise ValueError(f"Geçersiz sol bit grubu: {bits}")

    # Sağ taraftaki verileri analiz et
    for bits in right_groups:
        if bits in right:
            ean13_code.append(right[bits])
        else:
            raise ValueError(f"Geçersiz sağ bit grubu: {bits}")

    # Kontrol hanesini hesaplayın
    even_sum = sum(ean13_code[1::2])  # Çift pozisyonlardaki sayıların toplamı
    odd_sum = sum(ean13_code[0::2])  # Tek pozisyonlardaki sayıların toplamı
    total_sum = odd_sum + 3 * even_sum
    check_digit = (10 - (total_sum % 10)) % 10

    print("Start Index:", start_index)
    print("End Index:", end_index)
    print("Left Groups:", left_groups)
    print("Right Groups:", right_groups)
    print("EAN-13 Code:", ean13_code)
    print("Check Digit:", check_digit)

    return ean13_code


# first a conservative filter for grayscale images will be defined.
def conservative_smoothing_gray(data, filter_size):
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k, m])
            temp.remove(data[i, j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i, j] > max_value:
                new_image[i, j] = max_value
            elif data[i, j] < min_value:
                new_image[i, j] = min_value
            temp = []
    return new_image.copy()


def detect_barcode(image_path: str):
    # Barkod resmini yükleme ve siyah beyaz formatına çevirme
    image = Image.open(image_path).convert("L")

    # Gürültüyü temizleme (medyan filtre kullanma)
    clean_image = image.filter(ImageFilter.UnsharpMask())

    # Resmin siyah beyaz değerlerini numpy array'e çevirme
    clean_image_array = conservative_smoothing_gray(np.array(clean_image), 3)

    # Temizlenmiş resmi ve ortadaki 64 pikselin ortalama değerlerini gösteren grafikleri oluşturma
    plt.figure(figsize=(12, 6))

    # Temizlenmiş resmi gösterme
    plt.subplot(2, 1, 1)
    plt.imshow(clean_image, cmap="gray")
    plt.title("Cleaned Image")

    # Ortadaki 64 piksel sütununu alarak her satırdaki ortalama değerini hesaplama
    height, width = clean_image_array.shape
    start_row = (height * 2 // 5) - 16
    end_row = (height * 2 // 5) + 16
    middle_section = clean_image_array[start_row:end_row, :]
    clean_average_values = np.mean(middle_section, axis=0)

    hoppala = list(clean_average_values)

    for i in range(len(clean_average_values)):
        clean_average_values[i] = math.trunc(hoppala[i])

    print("clean_average_values:\n", clean_average_values)
    print("len(clean_average_values): ", len(clean_average_values))

    # Ortadaki 64 pikselin ortalama değerlerini bar grafikte gösterme
    plt.subplot(2, 1, 2)
    plt.bar(range(width), clean_average_values, color="gray")
    plt.axhline(y=127, color="r", linestyle="-")
    plt.title("Cleaned Middle 64 Pixels (Vertical Average)")
    plt.xlabel("Pixel Column Index")
    plt.ylabel("Average Pixel Value")

    plt.tight_layout()
    # plt.show() #Bu satırı comment silerek grafikleri görebilirsiniz.

    # Barkod değerlerini okuma ve sayısal değerini yazma

    # ? Alttaki satırları comment-out yaparak barkod değerlerini LCD ekranına yazdırabilirsiniz.
    #! Sadece Raspberry Pi üzerinde çalışırken kullanılabilir.
    barcode_values_list = extract_barcode_values(clean_average_values)
    barcode_values_string = ""
    for s in barcode_values_list:
        barcode_values_string += str(s)
    lcd.write_string("Algilandi")
    lcd._set_cursor_pos((1, 0))
    lcd.write_string(barcode_values_string)
    ######################################################################


# detect_barcode("test_barcode.jpg") #Bu satırı comment silerek test edebilirsiniz.
