"""Halaman utama: konsep Matematika 3 di balik pipeline (dengan demo interaktif pada sinyal aktif)."""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from core import mathcore as M

BLUE, ORANGE, GREY = "#1F5FBF", "#E4572E", "#8A94A6"


def _fig(h=320, **kw):
    f = go.Figure(); f.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
                                     legend=dict(orientation="h", y=-0.35), **kw); return f


def _seg(y, sr, n, start=None):
    """ambil n sampel dari bagian tengah sinyal"""
    s = (len(y) - n) // 2 if start is None else start
    return y[s:s + n]


def render(y, sr, src_label):
    st.title("Konsep Matematika di Balik Klasifikasi Kondisi Kipas")
    st.caption("Mata kuliah Matematika 3 · D4 Teknik Elektronika · NRP 2 — rumus → intuisi → demo pada sinyal aktif → hubungannya dengan kipas pendingin")
    st.info(f"Sinyal untuk demo: **{src_label}** ({len(y)/sr:.1f} s, {sr} Hz). Ganti di sidebar → *Sinyal aktif*.", icon="🎧")

    tabs = st.tabs(["0 · Peta konsep", "1 · Sampling & Nyquist", "2 · DFT & FFT", "3 · Window & STFT",
                    "4 · Mel, log & MFCC (DCT)", "5 · Random Forest", "6 · Evaluasi"])

    # -------------------------------------------------------------- 0
    with tabs[0]:
        st.subheader("Dari gelombang suara ke keputusan Normal / Abnormal")
        st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF1F6", color="#1F5FBF", fontname="Helvetica"];
        a[label="Suara kipas\\nx(t)"]; b[label="Sampling/ADC\\nx[n]=x(n/fs)"]; c[label="Framing + Window\\nx[n]·w[n]"]; d[label="STFT (FFT)\\nX[m,k]"];
        e[label="|X|² → filter mel\\n→ log"]; f[label="DCT-II\\nMFCC"]; g[label="Statistik klip\\nμ, σ"]; h[label="Random Forest\\nvoting pohon"]; i[label="Evaluasi\\nF1, AUC"];
        a->b->c->d->e->f->g->h->i; }""")
        df = pd.DataFrame([
            ("Sampling & ADC", "x[n] = x(n/fs), fs ≥ 2·f_max", "Materi: Sampling/ADC", "dibahas di tab 1"),
            ("DFT / FFT", "X[k] = Σ x[n]e^(−j2πkn/N)", "Materi: DFT/FFT", "mathcore.dft_naive, fft_radix2"),
            ("Window + STFT", "X[m,k] = Σ x[n+mH]w[n]e^(−j2πkn/N)", "Materi: DFT/FFT (analisis waktu-frekuensi)", "mathcore.stft"),
            ("Mel + log", "m = 2595·log10(1+f/700)", "Fitur audio", "mathcore.mel_filterbank"),
            ("DCT-II → MFCC", "c_k = Σ L[n]cos(πk(2n+½)/N)", "Fitur audio, reduksi dimensi", "mathcore.dct_matrix, mfcc"),
            ("Statistik klip", "μ = (1/T)Σx_t,  σ² = (1/T)Σ(x_t−μ)²", "Fitur statistik", "features._stats"),
            ("Random Forest", "G = 1 − Σ p_c², voting mayoritas", "Klasifikasi", "sklearn (+ demo Gini manual)"),
            ("Evaluasi", "P, R, F1, ROC-AUC", "Evaluasi", "models.evaluate"),
        ], columns=["Langkah", "Rumus inti", "Materi Mat 3", "Kode"])
        st.dataframe(df, hide_index=True, width="stretch")
        st.success("Semua langkah transformasi (DFT, FFT, window, STFT, filter mel, DCT, delta, fitur spektral) ditulis dari rumus di NumPy "
                   "dan **diverifikasi** terhadap librosa/scipy (`tests/verify_math.py`, galat relatif < 1e-6). "
                   "Hanya Random Forest yang memakai scikit-learn.")

    # -------------------------------------------------------------- 1
    with tabs[1]:
        st.subheader("Sampling dan teorema Nyquist–Shannon")
        st.latex(r"x[n]=x(nT_s),\quad T_s=\frac1{f_s},\qquad f_s\ge 2f_{\max}\;\Rightarrow\;\text{tanpa aliasing}")
        st.latex(r"f_{\text{alias}}=\left|\,f-f_s\cdot\operatorname{round}\!\left(f/f_s\right)\right|")
        st.write(f"MIMII direkam pada **fs = 16 kHz** → frekuensi Nyquist = **8 kHz**. Kipas pendingin menghasilkan komponen tonal "
                 "(putaran & *blade-pass frequency*) pada puluhan–ratusan Hz plus derau pita lebar; semuanya di bawah Nyquist.")
        c1, c2 = st.columns(2)
        f_sig = c1.slider("Frekuensi tone asli f (Hz)", 100, 20000, 9000, 100)
        fs_d = c2.select_slider("Sampling rate fs (Hz)", [4000, 8000, 16000, 22050, 44100], 16000)
        fa = abs(f_sig - fs_d * round(f_sig / fs_d))
        tt = np.linspace(0, 0.004, 4000); ts = np.arange(0, 0.004, 1 / fs_d)
        fg = _fig(300)
        fg.add_scatter(x=tt * 1e3, y=np.sin(2 * np.pi * f_sig * tt), name=f"asli {f_sig} Hz", line=dict(color=GREY, width=1))
        fg.add_scatter(x=ts * 1e3, y=np.sin(2 * np.pi * f_sig * ts), mode="markers", name="sampel", marker=dict(color=ORANGE, size=7))
        tr = np.sin(2 * np.pi * fa * tt) if f_sig <= fs_d / 2 else np.sin(2 * np.pi * f_sig * tt)
        fg.update_xaxes(title="t (ms)"); st.plotly_chart(fg, width="stretch")
        (st.success if f_sig <= fs_d / 2 else st.error)(
            f"Nyquist = {fs_d/2:.0f} Hz. " + ("f di bawah Nyquist → terekam benar." if f_sig <= fs_d / 2
            else f"f melebihi Nyquist → **aliasing**: terbaca sebagai {fa:.0f} Hz."))
        st.caption("Konsekuensi praktis: audio selain 16 kHz di-resample (polyphase) sebelum ekstraksi fitur, dan dashboard memberi peringatan.")

    # -------------------------------------------------------------- 2
    with tabs[2]:
        st.subheader("DFT dan FFT")
        st.latex(r"X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-j2\pi kn/N},\qquad f_k=\frac{k\,f_s}{N},\quad k=0,\dots,N-1")
        st.latex(r"\text{FFT radix-2:}\quad X[k]=E[k]+W_N^{k}O[k],\;\; X[k+\tfrac N2]=E[k]-W_N^{k}O[k],\;\; W_N^k=e^{-j2\pi k/N}")
        st.latex(r"\text{Kompleksitas: DFT } O(N^2)\ \longrightarrow\ \text{FFT } O(N\log_2 N)")
        N = st.select_slider("Panjang N (sampel)", [64, 128, 256, 512, 1024, 2048, 4096], 1024, key="Nfft")
        x = _seg(y, sr, N).astype(float)
        t0 = time.perf_counter(); Xd = M.dft_naive(x); td = time.perf_counter() - t0
        t0 = time.perf_counter(); Xf = M.fft_radix2(x); tf = time.perf_counter() - t0
        Xr = np.fft.fft(x)
        c = st.columns(4)
        c[0].metric("Waktu DFT O(N²)", f"{td*1e3:.2f} ms"); c[1].metric("Waktu FFT radix-2", f"{tf*1e3:.2f} ms")
        c[2].metric("Operasi N² vs (N/2)log₂N", f"{N*N:,} vs {int(N/2*np.log2(N)):,}")
        c[3].metric("Galat maks vs np.fft", f"{max(np.max(np.abs(Xd-Xr)), np.max(np.abs(Xf-Xr))):.1e}")
        fr = np.arange(N // 2 + 1) * sr / N
        fg = _fig(320); fg.add_scatter(x=fr, y=20 * np.log10(np.abs(Xf[:N // 2 + 1]) / N + 1e-12), line=dict(color=BLUE))
        fg.update_layout(xaxis_title="Frekuensi (Hz)", yaxis_title="|X[k]|/N (dB)"); st.plotly_chart(fg, width="stretch")
        st.write(f"Resolusi frekuensi **Δf = fs/N = {sr/N:.2f} Hz**. Puncak harmonik kipas tampak sebagai garis; rattle/gesekan menambah energi pita lebar.")

    # -------------------------------------------------------------- 3
    with tabs[3]:
        st.subheader("Window dan STFT: kompromi resolusi waktu–frekuensi")
        st.latex(r"X[m,k]=\sum_{n=0}^{N-1}x[n+mH]\,w[n]\,e^{-j2\pi kn/N},\qquad w[n]=\tfrac12\Big(1-\cos\tfrac{2\pi n}{N}\Big)\ \text{(Hann)}")
        st.latex(r"\Delta f=\frac{f_s}{N},\qquad \Delta t=\frac{N}{f_s},\qquad \Delta f\cdot\Delta t=1")
        a, b = st.columns(2)
        with a:
            st.markdown("**(a) Kebocoran spektral (leakage)**")
            f_t = st.slider("Frekuensi tone uji (Hz)", 1000.0, 1100.0, 1031.25, 0.25, key="ft")
            Nn = 1024; n = np.arange(Nn); xs = np.sin(2 * np.pi * f_t * n / sr)
            fg = _fig(300)
            for nm, w, col in (("Persegi (tanpa window)", np.ones(Nn), ORANGE), ("Hann", M.hann(Nn), BLUE)):
                sp = np.abs(np.fft.rfft(xs * w)); sp = 20 * np.log10(sp / sp.max() + 1e-6)
                fg.add_scatter(x=np.arange(len(sp)) * sr / Nn, y=sp, name=nm, line=dict(color=col))
            fg.update_xaxes(range=[800, 1300], title="Hz"); fg.update_yaxes(range=[-100, 3], title="dB rel.")
            st.plotly_chart(fg, width="stretch")
            st.caption("Bila f tidak tepat di bin DFT, window persegi bocor ke bin lain; Hann menekan sidelobe sehingga harmonik kipas tetap terpisah.")
        with b:
            st.markdown("**(b) Spektrogram sinyal aktif**")
            nf = st.select_slider("n_fft (N)", [256, 512, 1024, 2048, 4096], 1024, key="stft_n")
            hop = nf // 2
            P = 10 * np.log10(M.power_spec(M.stft(y, nf, hop)) + 1e-12)
            tx = np.arange(P.shape[1]) * hop / sr; fx = np.arange(P.shape[0]) * sr / nf
            fg = _fig(300); fg.add_heatmap(x=tx, y=fx, z=P, colorscale="Viridis", colorbar=dict(title="dB"))
            fg.update_yaxes(range=[0, min(4000, sr / 2)], title="Hz"); fg.update_xaxes(title="detik"); st.plotly_chart(fg, width="stretch")
            st.write(f"N={nf}: Δf = **{sr/nf:.1f} Hz**, Δt = **{1e3*nf/sr:.0f} ms**. N besar → harmonik tajam, impuls (rattle) kabur; N kecil → sebaliknya. "
                     "Baseline MIMII memakai N=1024, H=512.")

    # -------------------------------------------------------------- 4
    with tabs[4]:
        st.subheader("Skala mel, logaritma, dan DCT → MFCC")
        st.latex(r"m=2595\log_{10}\!\Big(1+\frac f{700}\Big),\qquad L_i[t]=10\log_{10}\!\Big(\sum_k H_i[k]\,|X[t,k]|^2+\varepsilon\Big)")
        st.latex(r"c_q[t]=\sum_{i=0}^{I-1}L_i[t]\cos\!\Big[\frac{\pi q}{I}\big(i+\tfrac12\big)\Big]\ \ (\text{DCT-II, ortonormal}),\qquad \Delta c[t]=\frac{\sum_{j=1}^{W}j\,(c[t+j]-c[t-j])}{2\sum_{j=1}^{W}j^2}")
        c1, c2, c3 = st.columns(3)
        nm = c1.slider("Jumlah filter mel I", 16, 128, 64, 8); nq = c2.slider("Jumlah koefisien MFCC Q", 4, 40, 13)
        nfm = c3.select_slider("n_fft", [512, 1024, 2048], 1024, key="mfn")
        fb = M.mel_filterbank(sr, nfm, nm); fr = np.linspace(0, sr / 2, fb.shape[1])
        a, b = st.columns(2)
        with a:
            fg = _fig(280, title="Filterbank mel segitiga H_i[k]")
            for i in range(nm): fg.add_scatter(x=fr, y=fb[i], line=dict(width=1), showlegend=False)
            fg.update_xaxes(title="Hz"); st.plotly_chart(fg, width="stretch")
        with b:
            D = M.dct_matrix(nm, 6)
            fg = _fig(280, title="Basis DCT-II (6 pertama)")
            for q in range(6): fg.add_scatter(y=D[q], name=f"q={q}")
            fg.update_xaxes(title="indeks mel i"); st.plotly_chart(fg, width="stretch")
        L = M.log_mel(y, sr, nfm, nfm // 2, nm); C = M.mfcc(y, sr, nfm, nfm // 2, nm, nq)
        fg = make_subplots(rows=1, cols=2, subplot_titles=("log-mel (dB)", f"MFCC (Q={nq})"))
        fg.add_heatmap(z=L, colorscale="Viridis", showscale=False, row=1, col=1); fg.add_heatmap(z=C, colorscale="RdBu", zmid=0, showscale=False, row=1, col=2)
        fg.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fg, width="stretch")
        st.markdown(f"- **Mel**: telinga/analisis lebih peka di frekuensi rendah → filter rapat di bawah, renggang di atas.\n"
                    f"- **Log**: mengubah perkalian (sumber × saluran) menjadi penjumlahan dan memadatkan rentang dinamis.\n"
                    f"- **DCT**: mendekorelasi {nm} energi mel yang berkorelasi → hanya {nq} koefisien pertama dipakai (reduksi dimensi {nm}→{nq}); koefisien rendah = bentuk garis besar spektrum.\n"
                    f"- **Delta**: kecepatan perubahan MFCC (menangkap kondisi tidak stasioner seperti impuls).")

    # -------------------------------------------------------------- 5
    with tabs[5]:
        st.subheader("Random Forest: impuritas Gini, bagging, voting")
        st.latex(r"G(S)=1-\sum_{c}p_c^{\,2},\qquad \Delta G=G(S)-\frac{|S_L|}{|S|}G(S_L)-\frac{|S_R|}{|S|}G(S_R)")
        st.latex(r"\hat y(\mathbf x)=\operatorname{mode}\{h_1(\mathbf x),\dots,h_B(\mathbf x)\},\qquad \hat P(\text{abnormal})=\frac1B\sum_{b=1}^{B}\mathbb 1[h_b(\mathbf x)=1]")
        st.markdown("Tiap pohon dilatih pada **bootstrap** sampel dan hanya melihat **√d fitur acak** di tiap simpul → pohon saling dekorelasi, "
                    "variansi ensambel turun. Simpul memilih fitur & ambang yang memaksimalkan ΔG.")
        st.markdown("**Hitung manual satu pemisahan** (mis. fitur `mfcc1_mean < θ`):")
        c = st.columns(4)
        nl0 = c[0].number_input("Kiri: normal", 0, 10000, 80); nl1 = c[1].number_input("Kiri: abnormal", 0, 10000, 5)
        nr0 = c[2].number_input("Kanan: normal", 0, 10000, 10); nr1 = c[3].number_input("Kanan: abnormal", 0, 10000, 45)
        gini = lambda a, b: 0.0 if a + b == 0 else 1 - (a / (a + b)) ** 2 - (b / (a + b)) ** 2
        n_l, n_r = nl0 + nl1, nr0 + nr1; n_all = max(n_l + n_r, 1)
        gp = gini(nl0 + nr0, nl1 + nr1); gl, gr = gini(nl0, nl1), gini(nr0, nr1)
        dg = gp - n_l / n_all * gl - n_r / n_all * gr
        st.latex(rf"G_{{induk}}={gp:.3f},\ G_L={gl:.3f},\ G_R={gr:.3f}\ \Rightarrow\ \Delta G={dg:.3f}")
        st.caption("Pohon hanya menerima pemisahan dengan ΔG besar. Nilai ΔG yang dijumlah di semua pohon menjadi *feature importance* di halaman Eksperimen.")
        mdl = st.session_state.get("model")
        if mdl is not None and hasattr(mdl, "estimators_"):
            from sklearn.tree import export_text
            st.markdown("**Pohon pertama dari model terlatih (kedalaman 2):**")
            st.code(export_text(mdl.estimators_[0], feature_names=list(st.session_state["feat_names"]), max_depth=2))

    # -------------------------------------------------------------- 6
    with tabs[6]:
        st.subheader("Metrik evaluasi")
        st.latex(r"\text{Acc}=\frac{TP+TN}{TP+TN+FP+FN},\quad P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}")
        st.latex(r"\text{TPR}=\frac{TP}{TP+FN},\ \text{FPR}=\frac{FP}{FP+TN},\qquad \text{AUC}=\int_0^1\text{TPR}\,d(\text{FPR})=P\big(\hat p_{\text{abn}}>\hat p_{\text{nor}}\big)")
        st.markdown("Konteks kipas: **recall** tinggi = sedikit kipas rusak terlewat (biaya kegagalan panel); **presisi** tinggi = sedikit alarm palsu. "
                    "MIMII tidak seimbang (abnormal lebih sedikit) → F1 dan AUC lebih informatif daripada akurasi.")
        r = st.session_state.get("result")
        if r:
            c = st.columns(5)
            for col, (k, lab) in zip(c, [("akurasi", "Akurasi"), ("presisi", "Presisi"), ("recall", "Recall"), ("f1", "F1"), ("auc", "AUC")]):
                col.metric(lab, f"{r[k]:.3f}")
        else:
            st.caption("Latih model di halaman *Eksperimen* untuk melihat angka aktual di sini.")
