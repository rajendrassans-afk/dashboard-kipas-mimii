# NRP 2 — Klasifikasi Kondisi Kipas Pendingin Panel Elektronika
Matematika 3 · D4 Teknik Elektronika · Dataset: MIMII (fan) · Metode: **STFT + MFCC + Random Forest**

Dashboard Streamlit dengan penekanan pada **konsep matematika** (halaman pertama): sampling/Nyquist, DFT/FFT (naif vs radix-2),
window & STFT, skala mel + log, DCT-II → MFCC, impuritas Gini & voting Random Forest, metrik evaluasi. Setiap konsep punya rumus,
demo interaktif pada sinyal aktif, dan kaitannya dengan kipas.

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```
1. Unduh zip `fan` dari https://zenodo.org/records/3384388 (baseline: https://github.com/MIMII-hitachi/mimii_baseline), ekstrak ke `./dataset`.
2. Halaman **Data & Validasi** → pindai folder. 3. **Eksperimen** → latih & bandingkan. 4. **Prediksi** → unggah WAV.

Uji cepat tanpa dataset: `python tools/make_synthetic.py --out demo_synthetic` (data SINTETIS, hanya untuk tes aplikasi, bukan hasil penelitian).

## Hubungan dengan baseline MIMII
Parameter fitur mengikuti baseline (`n_fft=1024, hop=512, n_mels=64`, kanal 0, 16 kHz). Perbedaan: baseline = autoencoder
unsupervised (latih normal saja, metrik AUC per id/SNR); proyek ini = Random Forest supervised dengan label normal/abnormal.
Karena itu angkanya tidak setara langsung; tabel AUC per ID × SNR disediakan agar bisa dibandingkan secara kualitatif.

## Verifikasi
`python tests/verify_math.py` — membandingkan DFT, FFT, Hann, STFT, filterbank mel, DCT-II, log-mel, delta buatan sendiri dengan librosa/scipy (galat relatif < 1e-6).
`python tests/flow.py` — uji alur Streamlit end-to-end; jalankan dari folder utama setelah `python tools/make_synthetic.py --out demo_synthetic`. `tests/smoke.py` = uji cepat render halaman.

## Batasan (sesuai ketentuan dosen)
Prototipe indikatif, bukan sertifikasi alat. MIMII hanya berlabel normal/abnormal; status NORMAL/WASPADA/ABNORMAL adalah
ambang probabilitas, **bukan** diagnosis kerusakan fisik. Gunakan skema uji *lintas-id* untuk evaluasi yang lebih ketat.
