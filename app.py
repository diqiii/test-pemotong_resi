import streamlit as st
import cv2
import numpy as np
import os
import shutil
import pytesseract
import re
import zipfile
import io

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Mesin Pemotong Resi", page_icon="✂️", layout="centered")

st.title("✂️ Mesin Pemotong Resi Otomatis")
st.markdown("**Platform:** TikTok Shop & Shopee | **Fitur:** Auto-Crop, OCR, Anti-Duplikat")

# --- FUNGSI MESIN TIKTOK ---
def proses_tiktok(img, global_counter, database_nomor, temp_dir):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    crop_gray = gray[:, int(w*0.05) : int(w*0.95)]
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
                if (y - start_y) >= 10: garis_ditemukan.append((start_y, y))
                in_garis = False
    if in_garis:
        if (h - start_y) >= 10: garis_ditemukan.append((start_y, h))

    if len(garis_ditemukan) >= 2:
        for i in range(len(garis_ditemukan) - 1):
            y_atas = garis_ditemukan[i][1] 
            y_bawah = garis_ditemukan[i+1][0]
            tinggi_resi = y_bawah - y_atas
            
            if tinggi_resi > 250:
                crop = img[y_atas:y_bawah, 0:w]
                
                header_h = int(tinggi_resi * 0.50)
                crop_ocr = cv2.cvtColor(crop[0:header_h, :], cv2.COLOR_BGR2GRAY)
                crop_ocr = cv2.resize(crop_ocr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                thresh_adaptive = cv2.adaptiveThreshold(crop_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
                teks_1 = pytesseract.image_to_string(thresh_adaptive)
                kontras = cv2.convertScaleAbs(crop_ocr, alpha=1.5, beta=0)
                teks_2 = pytesseract.image_to_string(kontras)
                
                teks_gabungan = teks_1 + " " + teks_2
                
                match = re.search(r'#\s*([A-Za-z0-9]{10,})', teks_gabungan)
                if not match:
                    match = re.search(r'(\d{15,})', teks_gabungan)
                
                if match:
                    nomor_pesanan = match.group(1)
                    if nomor_pesanan in database_nomor:
                        continue # Skip duplikat
                    
                    database_nomor.add(nomor_pesanan)
                    nama_file = f"TikTok_{global_counter}_{nomor_pesanan}.jpg"
                else:
                    nama_file = f"TikTok_{global_counter}_CEK_MANUAL.jpg"

                # Simpan ke folder temp
                cv2.imwrite(os.path.join(temp_dir, nama_file), crop)
                global_counter += 1
    return global_counter

# --- FUNGSI MESIN SHOPEE ---
def proses_shopee(img, global_counter, database_nomor, temp_dir):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    crop_center = gray[:, int(w*0.2) : int(w*0.8)]
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
                if y - start_y >= 10: gaps.append((start_y, y))
                in_gap = False
                
    if not gaps: return global_counter
        
    for i in range(len(gaps)):
        y_start_card = gaps[i][1] if i < len(gaps) else 0
        y_end_card = gaps[i+1][0] if i+1 < len(gaps) else h
        
        if y_end_card - y_start_card < 200: continue
        
        card_gray = gray[y_start_card:y_end_card, 0:w]
        
        card_ocr = cv2.resize(card_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.adaptiveThreshold(card_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        
        teks_full = pytesseract.image_to_string(thresh)
        match = re.search(r'([0-9]{6}[A-Z0-9]{8,10})', teks_full)
        
        if match:
            nomor_pesanan = match.group(1)
            
            if nomor_pesanan in database_nomor:
                return global_counter # Skip duplikat
            
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
            
            c_std = np.std(card_gray[:, int(w*0.2):int(w*0.8)], axis=1)
            for y_scan in range(start_scan, end_scan):
                if c_std[y_scan] < 8:
                    cut_y = y_scan + 2
                    break
            
            final_crop = img[y_start_card : y_start_card + cut_y, 0:w]
            
            database_nomor.add(nomor_pesanan)
            nama_file = f"Shopee_{global_counter}_{nomor_pesanan}.jpg"
            cv2.imwrite(os.path.join(temp_dir, nama_file), final_crop)
            global_counter += 1
            return global_counter
            
    return global_counter

# --- ANTARMUKA STREAMLIT ---
platform = st.radio("Pilih Platform Resi:", ("TikTok Shop", "Shopee"), horizontal=True)
uploaded_files = st.file_uploader("Upload Foto Resi (Bisa banyak sekaligus)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if st.button("Proses Resi 🚀"):
    if not uploaded_files:
        st.warning("Upload fotonya dulu bro!")
    else:
        with st.spinner('Mesin sedang memotong dan membaca nomor...'):
            temp_dir = "temp_hasil"
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            global_counter = 1
            database_nomor = set()
            
            # Loop semua foto yang diupload
            for file in uploaded_files:
                # Convert file uploader to cv2 image
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                if platform == "TikTok Shop":
                    global_counter = proses_tiktok(img, global_counter, database_nomor, temp_dir)
                elif platform == "Shopee":
                    global_counter = proses_shopee(img, global_counter, database_nomor, temp_dir)
            
            # Buat file ZIP
            hasil_files = os.listdir(temp_dir)
            if len(hasil_files) > 0:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for filename in hasil_files:
                        file_path = os.path.join(temp_dir, filename)
                        zip_file.write(file_path, arcname=filename)
                
                st.success(f"🎉 Selesai! Berhasil memproses {len(hasil_files)} resi unik (Anti-Duplikat aktif).")
                
                # Tombol Download
                st.download_button(
                    label="📥 Download Hasil Potongan (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"Hasil_Resi_{platform}.zip",
                    mime="application/zip"
                )
            else:
                st.error("Gagal memproses gambar. Pastikan gambar resi valid dan sesuai platform.")