# Peningkatan Kualitas Citra dengan imageio, NumPy, dan SciPy ndimage

Program ini meningkatkan kualitas foto (citra awal: `citra_awal.png`) melalui tahapan:

| Tahap | Metode | Library |
|---|---|---|
| 1 | Baca citra & konversi grayscale | imageio, numpy |
| 2 | Reduksi noise (median + Gaussian) | scipy.ndimage |
| 3 | Perbaikan kontras (contrast stretching) | numpy |
| 4 | Penajaman (unsharp masking) | scipy.ndimage |
| 5 | Deteksi tepi (Sobel) | scipy.ndimage |
| 6 | Evaluasi: mean, kontras (std), entropi, ketajaman (variansi Laplacian) | numpy, scipy.ndimage |

## Cara menjalankan
```bash
pip install imageio numpy scipy
python peningkatan_citra.py
```
Hasil tersimpan di folder `hasil/`.

## Hasil evaluasi
| Citra | Mean | Kontras (std) | Entropi | Ketajaman |
|---|---|---|---|---|
| Awal (grayscale) | 119.18 | 60.90 | 7.17 | 598.08* |
| Setelah reduksi noise | 119.30 | 60.05 | 7.06 | 53.23 |
| Setelah contrast stretch | 150.09 | 75.08 | 7.29 | 81.03 |
| **Hasil akhir** | 150.06 | 76.42 | 7.33 | 247.28 |

\*Ketajaman citra awal tinggi karena variansi Laplacian juga menghitung noise/tekstur kasar; setelah denoise, nilainya turun lalu naik kembali secara terkontrol lewat penajaman.
