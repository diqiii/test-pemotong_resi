import streamlit as st
import cv2
import numpy as np
import os
import shutil
import pytesseract
import re
import base64
from PIL import Image, ExifTags
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
        help="Aktifkan jika banyak resi gagal terbaca nomornya. Cocok untuk HP dengan foto resolusi rendah."
    )

st.title("✂️ Mesin Pemotong Resi Otomatis")
st.markdown("**Platform:** TikTok Shop & Shopee | **Fitur:** Original Core Logic + EXIF Fix + OCR Robust")


# ==========================================
# [FIX #1] FUNGSI KOREKSI ORIENTASI EXIF
# ==========================================
# MASALAH: Foto dari kamera HP memiliki metadata EXIF yang menyimpan info rotasi.
# OpenCV (cv2.imdecode) MENGABAIKAN metadata ini, sehingga gambar bisa terbaca
# terbalik atau miring di server → sensor garis gagal → resi tidak terpotong.
# SOLUSI: Gunakan PIL/Pillow untuk membaca dan menerapkan rotasi EXIF sebelum
# mengkonversi ke format OpenCV (numpy array BGR).
def baca_gambar_dengan_exif_fix(file_bytes_array):
    """
    Membaca gambar dari bytes, mengoreksi orientasi EXIF,
    lalu mengembalikan numpy array BGR (format OpenCV).
    """
    try:
        pil_img = Image.open(io.BytesIO(bytes(file_bytes_array)))
        
        # Cari tag EXIF Orientation
        exif_data = pil_img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'Orientation':
                    # Terapkan rotasi sesuai nilai EXIF
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
        
        # Konversi PIL (RGB) → numpy array → BGR (format OpenCV)
        img_rgb = np.array(pil_img.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
        
    except Exception:
        # Fallback ke cara lama jika PIL gagal
        return cv2.imdecode(file_bytes_array, 1)


# ==========================================
# [FIX #2] FUNGSI OCR YANG LEBIH TANGGUH
# ==========================================
# MASALAH: Di HP/server dengan resource terbatas, OCR bisa gagal diam-diam atau
# menghasilkan teks kosong karena resolusi preprocessing tidak optimal.
# SOLUSI: Tambahkan multiple preprocessing strategy + PSM mode berbeda,
# sehingga jika satu cara gagal, cara lain bisa menyelamatkan pembacaan nomor.
def ocr_multi_strategi(crop_gray, agresif=True):
    """
    Menjalankan OCR dengan beberapa strategi preprocessing.
    Menggabungkan semua hasil teks untuk memaksimalkan kemungkinan nomor terbaca.
    """
    hasil_teks = []
    
    # Resize 2x (sama seperti asli)
    img_2x = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Strategi 1: Adaptive Threshold (sama seperti asli)
    thresh_adaptive = cv2.adaptiveThreshold(
        img_2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    hasil_teks.append(pytesseract.image_to_string(thresh_adaptive, config='--psm 6'))
    
    # Strategi 2: Kontras tinggi (sama seperti asli)
    kontras = cv2.convertScaleAbs(img_2x, alpha=1.5, beta=0)
    hasil_teks.append(pytesseract.image_to_string(kontras, config='--psm 6'))
    
    if agresif:
        # [BARU] Strategi 3: Otsu Threshold - lebih baik untuk gambar dengan
        # pencahayaan tidak merata (foto HP di kondisi cahaya berbeda)
        _, thresh_otsu = cv2.threshold(img_2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hasil_teks.append(pytesseract.image_to_string(thresh_otsu, config='--psm 6'))
        
        # [BARU] Strategi 4: Denoise dulu baru threshold - menangani foto blur/noise
        denoised = cv2.fastNlMeansDenoising(img_2x, h=10)
        thresh_denoised = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
        )
        hasil_teks.append(pytesseract.image_to_string(thresh_denoised, config='--psm 6'))
        
        # [BARU] Strategi 5: PSM 4 (assume single column) - berguna jika teks
        # tersusun vertikal seperti di resi yang sempit
        hasil_teks.append(pytesseract.image_to_string(thresh_adaptive, config='--psm 4'))
    
    return " ".join(hasil_teks)


# --- FUNGSI MESIN TIKTOK (V15 - EXIF FIX + OCR ROBUST) ---
def proses_tiktok(img_asli, global_counter, database_nomor, temp_dir):
    h_asli, w = img_asli.shape[:2]

    y_trim_atas = int(h_asli * trim_atas_pct)
    y_trim_bawah = int(h_asli * trim_bawah_pct)
    if y_trim_bawah <= y_trim_atas:
        y_trim_bawah = h_asli
        y_trim_atas = 0

    img = img_asli[y_trim_atas:y_trim_bawah, 0:w]
    h = img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[:, int(w * 0.05):int(w * 0.95)]

    row_std = np.std(crop_gray, axis=1)
    row_mean = np.mean(crop_gray, axis=1)

    is_garis_pemisah = (row_std < 8) & (row_mean > 225) & (row_mean < 250)

    garis_ditemukan = []
    in_garis = False
    start_y = 0

    for y in range(h):
        if is_garis_pemisah[y]:
            if not in_garis:
                in_garis = True
                start_y = y
        else:
            if in_garis:
                if (y - start_y) >= tebal_garis_ui:
                    garis_ditemukan.append((start_y, y))
                in_garis = False

    if in_garis:
        if (h - start_y) >= tebal_garis_ui:
            garis_ditemukan.append((start_y, h))

    batas_y = [0]
    for g_start, g_end in garis_ditemukan:
        batas_y.append((g_start + g_end) // 2)
    batas_y.append(h)

    for i in range(len(batas_y) - 1):
        y_atas = batas_y[i]
        y_bawah = batas_y[i + 1]
        tinggi_resi = y_bawah - y_atas

        if tinggi_resi > 270:
            crop = img[y_atas:y_bawah, 0:w]

            header_h = int(tinggi_resi * 0.50)
            crop_ocr = cv2.cvtColor(crop[0:header_h, :], cv2.COLOR_BGR2GRAY)

            # [FIX #2] Ganti dengan fungsi OCR multi-strategi
            teks_gabungan = ocr_multi_strategi(crop_ocr, agresif=ocr_agresif)
            teks_upper = teks_gabungan.upper()

            if "BATAL" in teks_upper or "CANCELED" in teks_upper or "CANCELLED" in teks_upper:
                continue

            match = re.search(r'#\s*([A-Za-z0-9]{10,})', teks_gabungan)
            if not match:
                match = re.search(r'(\d{15,})', teks_gabungan)

            if match:
                nomor_pesanan = match.group(1)
                if nomor_pesanan not in database_nomor:
                    database_nomor.add(nomor_pesanan)
                    nama_file = f"TikTok_{global_counter}_{nomor_pesanan}.jpg"
                    cv2.imwrite(os.path.join(temp_dir, nama_file), crop)
                    global_counter += 1

    return global_counter


# --- FUNGSI MESIN SHOPEE (EXIF FIX + OCR ROBUST) ---
def proses_shopee(img, global_counter, database_nomor, temp_dir):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    crop_center = gray[:, int(w * 0.2):int(w * 0.8)]
    row_std = np.std(crop_center, axis=1)
    row_mean = np.mean(crop_center, axis=1)

    putih_kertas = np.percentile(row_mean, 95)
    batas_abu = putih_kertas - 2

    is_gap = (row_std < 5) & (row_mean < batas_abu)

    gaps = []
    in_gap = False
    start_y = 0
    for y in range(h):
        if is_gap[y]:
            if not in_gap:
                in_gap = True
                start_y = y
        else:
            if in_gap:
                if y - start_y >= 10:
                    gaps.append((start_y, y))
                in_gap = False

    if not gaps:
        return global_counter

    for i in range(len(gaps)):
        y_start_card = gaps[i][1] if i < len(gaps) else 0
        y_end_card = gaps[i + 1][0] if i + 1 < len(gaps) else h

        if y_end_card - y_start_card < 200:
            continue

        card_gray = gray[y_start_card:y_end_card, 0:w]
        
        # [FIX #2] Ganti dengan fungsi OCR multi-strategi
        teks_full = ocr_multi_strategi(card_gray, agresif=ocr_agresif)

        match = re.search(r'([0-9]{6}[A-Z0-9]{8,10})', teks_full)

        if match:
            nomor_pesanan = match.group(1)
            if nomor_pesanan in database_nomor:
                return global_counter

            # Jalankan OCR detail untuk cari posisi teks (tetap pakai cara asli)
            card_ocr = cv2.resize(card_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.adaptiveThreshold(
                card_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
            )
            d = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            
            lowest_text_y = 0
            for j in range(len(d['text'])):
                word = str(d['text'][j]).upper()
                if any(k in word for k in ["ID", "ORDER", "TIME", "WAKTU", "PAYMENT", "PEMBAYARAN", nomor_pesanan]):
                    y_bottom = (d['top'][j] + d['height'][j]) / 2
                    if y_bottom > lowest_text_y:
                        lowest_text_y = y_bottom

            cut_y = int(lowest_text_y) + 30
            start_scan = int(lowest_text_y) + 5
            end_scan = min(start_scan + 150, y_end_card - y_start_card)

            c_std = np.std(card_gray[:, int(w * 0.2):int(w * 0.8)], axis=1)
            for y_scan in range(start_scan, end_scan):
                if c_std[y_scan] < 8:
                    cut_y = y_scan + 2
                    break

            final_crop = img[y_start_card:y_start_card + cut_y, 0:w]
            database_nomor.add(nomor_pesanan)
            nama_file = f"Shopee_{global_counter}_{nomor_pesanan}.jpg"
            cv2.imwrite(os.path.join(temp_dir, nama_file), final_crop)
            global_counter += 1
            return global_counter

    return global_counter


# --- ANTARMUKA STREAMLIT ---
platform = st.radio("Pilih Platform Resi:", ("TikTok Shop", "Shopee"), horizontal=True)

with st.form("form_upload_resi", clear_on_submit=False):
    uploaded_files = st.file_uploader(
        "Upload Foto Resi (Bisa banyak sekaligus)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    tombol_proses = st.form_submit_button("Proses Resi 🚀", use_container_width=True)

if uploaded_files and not tombol_proses:
    st.info(f"✅ Mantap! {len(uploaded_files)} file udah masuk keranjang.")

if tombol_proses:
    if not uploaded_files:
        st.warning("Upload fotonya dulu bro di dalam kotak!")
    else:
        progress_bar = st.progress(0, text="Memulai proses...")
        
        with st.spinner('Membedah resi...'):
            temp_dir = "temp_hasil"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            global_counter = 1
            database_nomor = set()

            for idx, file in enumerate(uploaded_files):
                # Update progress bar
                pct = int((idx / len(uploaded_files)) * 100)
                progress_bar.progress(pct, text=f"Memproses file {idx+1} dari {len(uploaded_files)}...")
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                
                # [FIX #1] Gunakan fungsi baca gambar yang sadar EXIF
                img = baca_gambar_dengan_exif_fix(file_bytes)

                if img is None:
                    st.warning(f"⚠️ File '{file.name}' gagal dibaca, dilewati.")
                    continue

                if platform == "TikTok Shop":
                    global_counter = proses_tiktok(img, global_counter, database_nomor, temp_dir)
                elif platform == "Shopee":
                    global_counter = proses_shopee(img, global_counter, database_nomor, temp_dir)
        
        progress_bar.progress(100, text="Selesai! ✅")
        hasil_files = os.listdir(temp_dir)

        if len(hasil_files) > 0:
            st.success(f"🎉 Selesai! Menemukan {len(hasil_files)} potongan resi valid.")

            js_files_array = []
            for filename in hasil_files:
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode()
                    js_files_array.append(f'{{name: "{filename}", data: "data:image/jpeg;base64,{b64_str}"}}')

            js_array_str = ",\n".join(js_files_array)

            custom_html = f"""
            <button id="dl-btn" style="width:100%; padding: 15px; background-color:#FF4B4B; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold; font-size:16px; margin-bottom: 20px;">
                🚀 DOWNLOAD SEMUA {len(hasil_files)} GAMBAR SEKALIGUS
            </button>
            <script>
            document.getElementById('dl-btn').addEventListener('click', async function() {{
                const files = [
                    {js_array_str}
                ];
                for(let i=0; i<files.length; i++) {{
                    let link = document.createElement('a');
                    link.href = files[i].data;
                    link.download = files[i].name;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    await new Promise(r => setTimeout(r, 300));
                }}
            }});
            </script>
            """
            st.components.v1.html(custom_html, height=80)

            st.markdown("---")
            st.markdown("### Preview Hasil Potongan:")

            for filename in hasil_files:
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "rb") as file:
                    img_bytes = file.read()

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(img_bytes, use_container_width=True)
                with col2:
                    st.write(f"**✅ {filename}**")
                    st.download_button(
                        label="📥 Download",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/jpeg",
                        key=filename
                    )
                st.divider()
        else:
            st.error(
                "Tidak ada resi valid yang disimpan. "
                "Kemungkinan semua resi duplikat, dibatalkan, atau gagal terbaca nomornya.\n\n"
                "💡 **Tips:** Coba aktifkan 'Mode Agresif' di sidebar, atau sesuaikan slider Potong Atas/Bawah."
            )
