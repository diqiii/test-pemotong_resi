import streamlit as st
import cv2
import numpy as np
import pytesseract

st.set_page_config(page_title="Mode Diagnosa", layout="wide")
st.title("🛠️ Bedah Forensik Mesin TikTok")
st.markdown("Upload 1 gambar yang berisi 3 resi tapi gagal terdeteksi itu.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    # 1. BACA GAMBAR
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_asli = cv2.imdecode(file_bytes, 1)
    
    h_asli, w = img_asli.shape[:2]
    
    # Sunat UI Atas Bawah
    y_trim_atas = int(h_asli * 0.17)
    y_trim_bawah = int(h_asli * 0.85)
    img = img_asli[y_trim_atas:y_trim_bawah, 0:w]
    h = img.shape[0]
    
    # 2. CARI GARIS ABU-ABU (Rumus paling dasar & longgar)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[:, int(w*0.05) : int(w*0.95)]
    row_std = np.std(crop_gray, axis=1)
    row_mean = np.mean(crop_gray, axis=1)
    
    is_garis_pemisah = (row_std < 15) & (row_mean > 200) & (row_mean < 255)
    
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
                if (y - start_y) >= 3: # Turunin ke 3 pixel aja biar sensitif
                    garis_ditemukan.append((start_y, y))
                in_garis = False
    if in_garis and (h - start_y) >= 3:
        garis_ditemukan.append((start_y, h))
        
    batas_y = [0]
    for g_start, g_end in garis_ditemukan:
        batas_y.append((g_start + g_end) // 2)
    batas_y.append(h)
    
    # 3. TAMPILKAN PREVIEW GARIS POTONG
    img_preview = img.copy()
    for by in batas_y:
        cv2.line(img_preview, (0, by), (w, by), (0, 0, 255), 3) # Gambar garis merah
    
    st.image(img_preview, channels="BGR", use_container_width=True, caption="🔴 Garis Merah = Tempat Mesin Memotong")
    st.markdown("---")
    
    # 4. TAMPILKAN HASIL BACAAN OCR PER POTONGAN
    for i in range(len(batas_y) - 1):
        y_atas = batas_y[i]
        y_bawah = batas_y[i+1]
        tinggi = y_bawah - y_atas
        
        if tinggi > 100: # Cuma proses kalau tingginya wajar
            crop = img[y_atas:y_bawah, 0:w]
            
            # Fokus OCR ke 50% atas (Tempat nomor pesanan berada)
            header_h = int(tinggi * 0.50)
            crop_ocr = cv2.cvtColor(crop[0:header_h, :], cv2.COLOR_BGR2GRAY)
            crop_ocr = cv2.resize(crop_ocr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 2 Jenis filter OCR yang kita pakai di mesin
            thresh = cv2.adaptiveThreshold(crop_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            kontras = cv2.convertScaleAbs(crop_ocr, alpha=1.5, beta=0)
            
            teks_1 = pytesseract.image_to_string(thresh)
            teks_2 = pytesseract.image_to_string(kontras)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(crop, channels="BGR", use_container_width=True, caption=f"Potongan ke-{i+1}")
            with col2:
                st.write("**Teks yang dilihat Mesin:**")
                st.code(teks_1 + "\n\n--- FILTER 2 ---\n\n" + teks_2, language="text")
