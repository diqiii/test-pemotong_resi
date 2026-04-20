import streamlit as st
import cv2
import numpy as np
import os
import shutil
import pytesseract
import re
import base64
from PIL import Image, ExifTags
from datetime import datetime
import io

st.set_page_config(page_title="Mesin Pemotong Resi", page_icon="✂️", layout="wide")

# ==========================================
# PENGATURAN KENDALI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ Kendali Mesin")
    st.markdown("---")
    st.markdown("**✂️ PISAU CUKUR (Potong Layar HP)**")
    trim_atas_ui = st.slider("Potong Atas (%)", 0, 30, 10, help="Membuang menu atas (Baterai, Sinyal, dll)")
    trim_bawah_ui = st.slider("Potong Bawah (%)", 0, 30, 15, help="Membuang menu bawah (Tombol dll)")

    trim_atas_pct = trim_atas_ui / 100.0
    trim_bawah_pct = 1.0 - (trim_bawah_ui / 100.0)

    st.markdown("---")
    st.markdown("**📏 SENSOR GARIS (Khusus TikTok)**")
    tebal_garis_ui = st.slider(
        "Tebal Garis Minimal (px)", 5, 60, 10,
        help="Naikkan angka ini jika resi kepotong di tengah. Turunkan jika 2 resi malah menyambung."
    )

    st.markdown("---")
    st.markdown("**🔬 MODE OCR**")
    ocr_agresif = st.checkbox(
        "Mode Agresif (lebih lambat, lebih akurat)",
        value=True,
        help="Aktifkan jika banyak resi gagal terbaca nomornya."
    )

st.title("✂️ Mesin Pemotong Resi Otomatis")
st.markdown("**Platform:** TikTok Shop & Shopee | **Fitur:** Full Info + EXIF Fix + OCR Robust + Auto-Sort")


# ==========================================
# FUNGSI 1: KOREKSI ORIENTASI EXIF
# ==========================================
def baca_gambar_dengan_exif_fix(file_bytes_array):
    try:
        pil_img = Image.open(io.BytesIO(bytes(file_bytes_array)))
        exif_data = pil_img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
        img_rgb = np.array(pil_img.convert('RGB'))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imdecode(file_bytes_array, 1)


# ==========================================
# FUNGSI 2: OCR MULTI-STRATEGI
# ==========================================
def ocr_dari_gray(crop_gray, agresif=True):
    hasil_teks = []
    img_2x = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    thresh_adaptive = cv2.adaptiveThreshold(
        img_2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    hasil_teks.append(pytesseract.image_to_string(thresh_adaptive, config='--psm 6'))

    kontras = cv2.convertScaleAbs(img_2x, alpha=1.5, beta=0)
    hasil_teks.append(pytesseract.image_to_string(kontras, config='--psm 6'))

    if agresif:
        _, thresh_otsu = cv2.threshold(img_2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hasil_teks.append(pytesseract.image_to_string(thresh_otsu, config='--psm 6'))

        denoised = cv2.fastNlMeansDenoising(img_2x, h=10)
        thresh_denoised = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
        )
        hasil_teks.append(pytesseract.image_to_string(thresh_denoised, config='--psm 6'))

        hasil_teks.append(pytesseract.image_to_string(thresh_adaptive, config='--psm 4'))

    return " ".join(hasil_teks)


def ekstrak_nomor_tiktok(teks):
    match = re.search(r'#\s*([A-Za-z0-9]{10,})', teks)
    if not match:
        match = re.search(r'(\d{15,})', teks)
    return match.group(1) if match else None


# ==========================================
# FUNGSI 3: MESIN TIKTOK
# ==========================================
def proses_tiktok(img_asli, database_nomor):
    hasil_sementara = []

    h_asli, w    = img_asli.shape[:2]
    y_trim_atas  = int(h_asli * trim_atas_pct)
    y_trim_bawah = int(h_asli * trim_bawah_pct)
    if y_trim_bawah <= y_trim_atas:
        y_trim_bawah = h_asli
        y_trim_atas  = 0

    img = img_asli[y_trim_atas:y_trim_bawah, 0:w]
    h   = img.shape[0]

    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sensor_gray = gray[:, int(w * 0.05):int(w * 0.95)]
    row_std     = np.std(sensor_gray, axis=1)
    row_mean    = np.mean(sensor_gray, axis=1)
    is_garis    = (row_std < 8) & (row_mean > 225) & (row_mean < 250)

    garis_ditemukan = []
    in_garis = False
    start_y  = 0
    for y in range(h):
        if is_garis[y]:
            if not in_garis:
                in_garis = True
                start_y  = y
        else:
            if in_garis:
                if (y - start_y) >= tebal_garis_ui:
                    garis_ditemukan.append((start_y, y))
                in_garis = False
    if in_garis and (h - start_y) >= tebal_garis_ui:
        garis_ditemukan.append((start_y, h))

    batas_y = [0]
    for g_start, g_end in garis_ditemukan:
        batas_y.append((g_start + g_end) // 2)
    batas_y.append(h)

    total_segmen = len(batas_y) - 1

    for i in range(total_segmen):
        y_atas      = batas_y[i]
        y_bawah     = batas_y[i + 1]
        tinggi_resi = y_bawah - y_atas

        if tinggi_resi < 270:
            continue

        crop_untuk_disimpan = img[y_atas:y_bawah, 0:w]

        is_resi_pertama  = (i == 0)
        is_resi_terakhir = (i == total_segmen - 1)

        ocr_h        = tinggi_resi if (is_resi_pertama or is_resi_terakhir) else int(tinggi_resi * 0.40)
        area_ocr_bgr = crop_untuk_disimpan[0:ocr_h, 0:w].copy()

        if is_resi_terakhir:
            ix = int(w * 0.72)
            iy = int(ocr_h * 0.65)
            area_ocr_bgr[iy:, ix:] = 255

        area_ocr_gray = cv2.cvtColor(area_ocr_bgr, cv2.COLOR_BGR2GRAY)
        teks          = ocr_dari_gray(area_ocr_gray, agresif=ocr_agresif)

        teks_upper = teks.upper()
        if "BATAL" in teks_upper or "CANCELED" in teks_upper or "CANCELLED" in teks_upper:
            continue

        nomor = ekstrak_nomor_tiktok(teks)

        if not nomor:
            full_bgr = crop_untuk_disimpan.copy()
            if is_resi_terakhir:
                fx = int(w * 0.72)
                fy = int(tinggi_resi * 0.65)
                full_bgr[fy:, fx:] = 255
            full_gray  = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
            teks_full  = ocr_dari_gray(full_gray, agresif=ocr_agresif)
            teks_upper = teks_full.upper()
            if "BATAL" in teks_upper or "CANCELED" in teks_upper or "CANCELLED" in teks_upper:
                continue
            nomor = ekstrak_nomor_tiktok(teks_full)

        if nomor and nomor not in database_nomor:
            database_nomor.add(nomor)
            hasil_sementara.append((nomor, crop_untuk_disimpan))

    return hasil_sementara


# ==========================================
# FUNGSI 4: MESIN SHOPEE
# ==========================================
def ekstrak_tanggal_shopee(teks):
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})[,\s]+(\d{2}):(\d{2})', teks)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)),
                            int(m.group(4)), int(m.group(5)))
        except ValueError:
            pass

    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[,\s]+(\d{2}):(\d{2})', teks)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)))
        except ValueError:
            pass

    m = re.search(r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', teks, re.IGNORECASE)
    if m:
        bulan_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                     'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        try:
            return datetime(int(m.group(3)), bulan_map[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            pass

    return datetime(1970, 1, 1)


def proses_shopee(img, database_nomor):
    hasil_sementara = []

    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    crop_center  = gray[:, int(w * 0.2):int(w * 0.8)]
    row_std      = np.std(crop_center, axis=1)
    row_mean     = np.mean(crop_center, axis=1)
    putih_kertas = np.percentile(row_mean, 95)
    batas_abu    = putih_kertas - 2
    is_gap       = (row_std < 5) & (row_mean < batas_abu)

    gaps    = []
    in_gap  = False
    start_y = 0
    for y in range(h):
        if is_gap[y]:
            if not in_gap:
                in_gap  = True
                start_y = y
        else:
            if in_gap:
                if y - start_y >= 10:
                    gaps.append((start_y, y))
                in_gap = False

    if not gaps:
        return hasil_sementara

    for i in range(len(gaps)):
        y_start_card = gaps[i][1]
        y_end_card   = gaps[i + 1][0] if i + 1 < len(gaps) else h

        if y_end_card - y_start_card < 200:
            continue

        card_gray = gray[y_start_card:y_end_card, 0:w]
        card_ocr  = cv2.resize(card_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        thresh    = cv2.adaptiveThreshold(
            card_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        teks_full = pytesseract.image_to_string(thresh)
        match     = re.search(r'([0-9]{6}[A-Z0-9]{8,10})', teks_full)

        if match:
            nomor_pesanan = match.group(1)
            if nomor_pesanan in database_nomor:
                continue

            tanggal = ekstrak_tanggal_shopee(teks_full)

            d = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            lowest_text_y = 0
            for j in range(len(d['text'])):
                word = str(d['text'][j]).upper()
                if any(k in word for k in ["ID", "ORDER", "TIME", "WAKTU", "PAYMENT", "PEMBAYARAN", nomor_pesanan]):
                    y_bottom = (d['top'][j] + d['height'][j]) / 2
                    if y_bottom > lowest_text_y:
                        lowest_text_y = y_bottom

            cut_y      = int(lowest_text_y) + 30
            start_scan = int(lowest_text_y) + 5
            end_scan   = min(start_scan + 150, y_end_card - y_start_card)

            c_std = np.std(card_gray[:, int(w * 0.2):int(w * 0.8)], axis=1)
            for y_scan in range(start_scan, end_scan):
                if c_std[y_scan] < 8:
                    cut_y = y_scan + 2
                    break

            final_crop = img[y_start_card:y_start_card + cut_y, 0:w]
            database_nomor.add(nomor_pesanan)
            hasil_sementara.append((tanggal, nomor_pesanan, final_crop))

    return hasil_sementara


# ==========================================
# ANTARMUKA STREAMLIT
# ==========================================
platform = st.radio("Pilih Platform Resi:", ("TikTok Shop", "Shopee"), horizontal=True)

uploaded_files = st.file_uploader(
    "📷 Pilih foto resi (bisa banyak sekaligus)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Di HP: pilih semua foto sekaligus dalam satu sesi pemilihan agar tidak timeout."
)

if uploaded_files:
    st.info(f"✅ {len(uploaded_files)} foto siap — tekan tombol di bawah untuk mulai.")

tombol_proses = st.button("🚀 Proses Resi", use_container_width=True, type="primary")

if tombol_proses:
    if not uploaded_files:
        st.warning("⚠️ Pilih foto dulu sebelum memproses!")
    else:
        progress_bar = st.progress(0, text="Memulai proses...")

        with st.spinner("Membedah resi..."):
            temp_dir = "temp_hasil"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            database_nomor = set()

            if platform == "TikTok Shop":
                semua_hasil = []
                for idx, file in enumerate(uploaded_files):
                    pct = int((idx / len(uploaded_files)) * 100)
                    progress_bar.progress(pct, text=f"Memproses file {idx+1} dari {len(uploaded_files)}...")
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img        = baca_gambar_dengan_exif_fix(file_bytes)
                    if img is None:
                        st.warning(f"⚠️ File '{file.name}' gagal dibaca, dilewati.")
                        continue
                    semua_hasil.extend(proses_tiktok(img, database_nomor))

                semua_hasil.sort(key=lambda x: x[0], reverse=True)
                for counter, (nomor, crop) in enumerate(semua_hasil, start=1):
                    cv2.imwrite(os.path.join(temp_dir, f"TikTok_{counter:03d}_{nomor}.jpg"), crop)

            elif platform == "Shopee":
                semua_hasil = []
                for idx, file in enumerate(uploaded_files):
                    pct = int((idx / len(uploaded_files)) * 100)
                    progress_bar.progress(pct, text=f"Memproses file {idx+1} dari {len(uploaded_files)}...")
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img        = baca_gambar_dengan_exif_fix(file_bytes)
                    if img is None:
                        st.warning(f"⚠️ File '{file.name}' gagal dibaca, dilewati.")
                        continue
                    semua_hasil.extend(proses_shopee(img, database_nomor))

                semua_hasil.sort(key=lambda x: x[0], reverse=True)
                for counter, (tanggal, nomor, crop) in enumerate(semua_hasil, start=1):
                    tgl_str = tanggal.strftime("%Y%m%d") if tanggal.year > 1970 else "tglUnknown"
                    cv2.imwrite(os.path.join(temp_dir, f"Shopee_{counter:03d}_{tgl_str}_{nomor}.jpg"), crop)

        progress_bar.progress(100, text="Selesai! ✅")
        hasil_files = sorted(os.listdir(temp_dir))

        if hasil_files:
            st.success(f"🎉 Selesai! Menemukan {len(hasil_files)} potongan resi valid.")

            js_files_array = []
            for filename in hasil_files:
                with open(os.path.join(temp_dir, filename), "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode()
                    js_files_array.append(f'{{name: "{filename}", data: "data:image/jpeg;base64,{b64_str}"}}')

            st.components.v1.html(f"""
            <button id="dl-btn" style="width:100%;padding:15px;background:#FF4B4B;color:white;
              border:none;border-radius:8px;cursor:pointer;font-weight:bold;font-size:16px;margin-bottom:20px;">
                🚀 DOWNLOAD SEMUA {len(hasil_files)} GAMBAR SEKALIGUS
            </button>
            <script>
            document.getElementById('dl-btn').addEventListener('click', async function() {{
                const files = [{",".join(js_files_array)}];
                for(let i=0;i<files.length;i++) {{
                    let a = document.createElement('a');
                    a.href = files[i].data; a.download = files[i].name;
                    document.body.appendChild(a); a.click(); document.body.removeChild(a);
                    await new Promise(r => setTimeout(r, 300));
                }}
            }});
            </script>
            """, height=80)

            st.markdown("---")
            st.markdown("### Preview Hasil Potongan:")
            for filename in hasil_files:
                with open(os.path.join(temp_dir, filename), "rb") as f:
                    img_bytes = f.read()
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(img_bytes, use_container_width=True)
                with col2:
                    st.write(f"**✅ {filename}**")
                    st.download_button("📥 Download", img_bytes, filename, "image/jpeg", key=filename)
                st.divider()
        else:
            st.error(
                "Tidak ada resi valid yang disimpan. "
                "Kemungkinan semua resi duplikat, dibatalkan, atau gagal terbaca nomornya.\n\n"
                "💡 **Tips:** Coba aktifkan 'Mode Agresif' di sidebar, atau sesuaikan slider Potong Atas/Bawah."
            )
