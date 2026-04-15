import streamlit as st
import cv2
import numpy as np
import os
import shutil
import pytesseract
import re
import zipfile
import io
import base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Mesin Pemotong Resi", page_icon="✂️", layout="centered")

st.title("✂️ Mesin Pemotong Resi Otomatis")
st.markdown("**Platform:** TikTok Shop & Shopee | **Fitur:** Auto-Crop, OCR, Anti-Duplikat, Filter Batal")

# --- FUNGSI MESIN TIKTOK (V8 - STABILITAS MATANG) ---
def proses_tiktok(img_asli, global_counter, database_nomor, temp_dir):
    h_asli, w = img_asli.shape[:2]
    
    # Pisau Cukur (Sesuai HP)
    y_trim_atas = int(h_asli * 0.15)
    y_trim_bawah = int(h_asli * 0.85)
    
    img = img_asli[y_trim_atas:y_trim_bawah, 0:w]
    h = img.shape[0] 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # EFEK BLUR DIHAPUS - Kembali murni ke logika aslimu!
    crop_gray = gray[:, int(w*0.05) : int(w*0.95)]
    row_std = np.std(crop_gray, axis=1)
    row_mean = np.mean(crop_gray, axis=1)
    
    # Kembali ke rumus skriptiktokmatang.txt (Toleransi std sedikit dinaikkan untuk laptop)
    is_garis_pemisah = (row_std < 12) & (row_mean > 220) & (row_mean < 252)
    
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
                # Ketebalan 5 pixel (Biar resolusi kecil di laptop tetap ke-detect)
                if (y - start_y) >= 5: 
                    garis_ditemukan.append((start_y, y))
                in_garis = False

    if in_garis:
        if (h - start_y) >= 5: 
            garis_ditemukan.append((start_y, h))

    # Smart Slicer: Garis potong dari paling atas (0) sampai paling bawah (h)
    batas_y = [0]
    for g_start, g_end in garis_ditemukan:
        batas_y.append((g_start + g_end) // 2)
    batas_y.append(h)

    for i in range(len(batas_y) - 1):
        y_atas = batas_y[i]
        y_bawah = batas_y[i+1]
        tinggi_resi = y_bawah - y_atas
        
        if tinggi_resi > 150: 
            crop = img[y_atas:y_bawah, 0:w]
            
            header_h = int(tinggi_resi * 0.50)
            crop_ocr = cv2.cvtColor(crop[0:header_h, :], cv2.COLOR_BGR2GRAY)
            crop_ocr = cv2.resize(crop_ocr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            thresh_adaptive = cv2.adaptiveThreshold(crop_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            teks_1 = pytesseract.image_to_string(thresh_adaptive)
            
            kontras = cv2.convertScaleAbs(crop_ocr, alpha=1.5, beta=0)
            teks_2 = pytesseract.image_to_string(kontras)
            
            teks_gabungan = teks_1 + " " + teks_2
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

# --- FUNGSI MESIN SHOPEE (Dikembalikan Utuh ke V5 yang Stabil) ---
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
        teks_upper = teks_full.upper()
        
        if "BATAL" in teks_upper or "CANCELED" in teks_upper or "CANCELLED" in teks_upper:
            return global_counter 
            
        match = re.search(r'([0-9]{6}[A-Z0-9]{8,10})', teks_full)
        
        if match:
            nomor_pesanan = match.group(1)
            
            if nomor_pesanan in database_nomor:
                return global_counter 
            
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

with st.form("form_upload_resi", clear_on_submit=False):
    uploaded_files = st.file_uploader("Upload Foto Resi (Bisa banyak sekaligus)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    tombol_proses = st.form_submit_button("Proses Resi 🚀", use_container_width=True)

if uploaded_files and not tombol_proses:
    st.info(f"✅ Mantap! {len(uploaded_files)} file udah masuk keranjang. Silakan klik tombol Proses 🚀")

if tombol_proses:
    if not uploaded_files:
        st.warning("Upload fotonya dulu bro di dalam kotak!")
    else:
        with st.spinner('Mesin sedang memotong, membaca nomor, dan menyaring resi batal...'):
            temp_dir = "temp_hasil"
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            global_counter = 1
            database_nomor = set()
            
            for file in uploaded_files:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                if platform == "TikTok Shop":
                    global_counter = proses_tiktok(img, global_counter, database_nomor, temp_dir)
                elif platform == "Shopee":
                    global_counter = proses_shopee(img, global_counter, database_nomor, temp_dir)
            
            hasil_files = os.listdir(temp_dir)
            
            if len(hasil_files) > 0:
                st.success(f"🎉 Selesai! Berhasil memproses {len(hasil_files)} resi valid.")
                
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
                        st.write(f"**{filename}**")
                        st.download_button(
                            label="📥 Download",
                            data=img_bytes,
                            file_name=filename,
                            mime="image/jpeg",
                            key=filename
                        )
                    st.divider()
            else:
                st.error("Tidak ada resi yang disimpan. Kemungkinan semua resi duplikat, dibatalkan, atau gagal terbaca nomornya.")
