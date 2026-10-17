"""
Peningkatan Kualitas Citra (Image Enhancement)
Library: imageio, numpy, scipy.ndimage

Tahapan:
1. Baca citra (imageio)
2. Konversi ke grayscale
3. Reduksi noise      -> Gaussian filter / Median filter (scipy.ndimage)
4. Perbaikan kontras  -> Contrast stretching (numpy); histogram equalization tersedia sebagai pembanding
5. Penajaman          -> Unsharp masking (scipy.ndimage)
6. Deteksi tepi       -> Sobel (scipy.ndimage)
7. Evaluasi           -> Statistik, entropi, dan ketajaman (variansi Laplacian)
"""
import numpy as np
import imageio.v3 as iio
from scipy import ndimage as ndi

INPUT = "citra_awal.png"
OUT = "hasil/"


def to_gray(rgb):
    """Konversi RGB -> grayscale (rumus luminansi ITU-R BT.601)."""
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def histogram_equalization(gray):
    """Pemerataan histogram menggunakan numpy."""
    g = np.clip(gray, 0, 255).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255)
    return lut.clip(0, 255).astype(np.uint8)[g].astype(float)


def contrast_stretching(gray, p_low=1, p_high=99, gamma=1.0):
    """Peregangan kontras berbasis persentil (lebih lembut dari equalization)."""
    lo, hi = np.percentile(gray, [p_low, p_high])
    out = np.clip((gray - lo) / (hi - lo), 0, 1)
    return (out ** gamma) * 255


def unsharp_mask(gray, sigma=2.0, amount=1.5):
    """Penajaman: citra + amount * (citra - citra_blur)."""
    blur = ndi.gaussian_filter(gray, sigma=sigma)
    return np.clip(gray + amount * (gray - blur), 0, 255)


def sobel_edges(gray):
    """Deteksi tepi Sobel -> magnitudo gradien (dinormalisasi 0-255)."""
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    return (mag / mag.max() * 255)


def entropy(gray):
    """Entropi Shannon: ukuran kekayaan informasi citra."""
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256)
    p = hist / hist.sum()
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def sharpness(gray):
    """Ketajaman: variansi Laplacian (makin besar makin tajam)."""
    return ndi.laplace(gray).var()


def report(nama, g):
    print(f"{nama:<28} mean={g.mean():7.2f}  std(kontras)={g.std():6.2f}  "
          f"entropi={entropy(g):5.2f}  ketajaman={sharpness(g):9.2f}")


def u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def main():
    rgb = iio.imread(INPUT)[..., :3].astype(float)
    gray = to_gray(rgb)

    # --- Pipeline peningkatan ---
    denoised = ndi.median_filter(gray, size=3)          # buang noise impulsif
    denoised = ndi.gaussian_filter(denoised, sigma=0.7) # haluskan noise ringan
    equalized = contrast_stretching(denoised, 1, 99, gamma=0.9)  # perbaiki kontras
    enhanced = unsharp_mask(equalized, sigma=1.5, amount=1.0)    # pertajam
    edges = sobel_edges(enhanced)

    print("=== EVALUASI KUALITAS CITRA ===")
    report("Citra awal (grayscale)", gray)
    report("Setelah reduksi noise", denoised)
    report("Setelah contrast stretch", equalized)
    report("Citra hasil peningkatan", enhanced)

    # --- Simpan hasil tiap tahap ---
    iio.imwrite(OUT + "1_grayscale.png", u8(gray))
    iio.imwrite(OUT + "2_denoise.png", u8(denoised))
    iio.imwrite(OUT + "3_kontras.png", u8(equalized))
    iio.imwrite(OUT + "4_enhanced.png", u8(enhanced))
    iio.imwrite(OUT + "5_edges_sobel.png", u8(edges))

    # --- Perbandingan sebelum/sesudah (seluruh citra, diperkecil) ---
    h, w = gray.shape
    side = np.hstack([gray, enhanced, edges])
    iio.imwrite(OUT + "perbandingan_penuh.png", u8(side[::2, ::2]))

    # --- Zoom area plat nomor & badan motor (detail) ---
    y0, y1, x0, x1 = 480, 680, 380, 700
    crop = lambda a: a[y0:y1, x0:x1]
    zoom = lambda a: np.kron(crop(a), np.ones((3, 3)))   # perbesar 3x
    pair = np.vstack([zoom(gray), np.full((6, zoom(gray).shape[1]), 255), zoom(enhanced)])
    iio.imwrite(OUT + "perbandingan_detail.png", u8(pair))

    # --- Histogram sebelum/sesudah (digambar dengan numpy) ---
    def hist_img(g, tinggi=200):
        hist = np.bincount(u8(g).ravel(), minlength=256).astype(float)
        hist = hist / hist.max() * (tinggi - 1)
        canvas = np.full((tinggi, 256 * 2), 255.0)
        for i, v in enumerate(hist):
            canvas[tinggi - 1 - int(v):, i * 2:i * 2 + 2] = 40
        return canvas
    hh = np.vstack([hist_img(gray), np.full((6, 512), 255.0), hist_img(enhanced)])
    iio.imwrite(OUT + "histogram_sebelum_sesudah.png", u8(hh))
    print("\nSemua hasil tersimpan di folder:", OUT)


if __name__ == "__main__":
    main()
