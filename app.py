        f.add_scatter(x=fpr, y=tpr, line=dict(color=BLUE)); f.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash="dash"), showlegend=False); b.plotly_chart(f, width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"TP={tp}, TN={tn}, FP={fp} (alarm palsu), FN={fn} (kipas rusak terlewat). "
                    f"Presisi = {tp}/({tp}+{fp}), Recall = {tp}/({tp}+{fn}).")
    with t2:
        imp = pd.Series(S["model"].feature_importances_, index=S["feat_names"]).sort_values().tail(15)
        f = fig(420, title="15 fitur teratas (rata-rata penurunan impuritas Gini)"); f.add_bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE); st.plotly_chart(f, width="stretch")
    with t3:
        dt_ = d.iloc[te].assign(p=proba, y=y[te]); rows = []
        for (i, s), g in dt_.groupby(["id", "snr"], dropna=False):
            if g.y.nunique() > 1: rows.append(dict(id=i, snr=s, n=len(g), AUC=skm.roc_auc_score(g.y, g.p)))
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption("Tabel gaya baseline MIMII (AUC per id & SNR). Catatan: baseline = autoencoder *tanpa label abnormal saat latih*; proyek ini supervised (Random Forest), sehingga angkanya tidak setara langsung.")
        st.markdown("Contoh AUC baseline resmi (readme): fan id_00 → 0 dB 0.634 · 6 dB 0.745 · −6 dB 0.576; id_02 → 0.856 · 0.993 · 0.640; id_04 → 0.730 · 0.869 · 0.572.")
    with t4:
        X = S["X"]; tr_ = S["tr"]
        if st.button("Bandingkan 4 model (fitur sama)"):
            S["cmp_models"] = MD.compare_models(X, y, tr_, te)
        if "cmp_models" in S: st.dataframe(S["cmp_models"].round(3), hide_index=True, width="stretch")
        if st.button("Sweep n_estimators × max_depth (Random Forest)"):
            S["cmp_rf"] = MD.sweep_rf(X, y, tr_, te)
        if "cmp_rf" in S:
            sw = S["cmp_rf"]; pv = sw.pivot(index="max_depth", columns="n_estimators", values="f1")
            f = fig(300, title="F1 uji"); f.add_heatmap(z=pv.values, x=pv.columns.astype(str), y=pv.index.astype(str), text=pv.values.round(3), texttemplate="%{text}", colorscale="Blues"); st.plotly_chart(f, width="stretch")
        st.markdown("**Parameter STFT/MFCC** (ekstraksi ulang tiap kombinasi; hasil di-cache)")
        if st.button("Sweep n_fft × n_mfcc × set fitur"):
            rows = []; combos = [(nf, nm, fs) for nf in (512, 1024, 2048) for nm in (8, 13, 20) for fs in ("mfcc", "mfcc_spec")] + [(nf, 13, "logmel") for nf in (512, 1024, 2048)]
            bar = st.progress(0.0)
            for i, (nf, nm, fs) in enumerate(combos):
                c_ = FeatureConfig(n_fft=nf, hop=nf // 2, n_mfcc=nm, feature_set=fs)
                Xc, _ = build_matrix(d.path.tolist(), c_, cache_dir=str(ROOT / "cache"))
                _, _, rr = MD.fit_eval("Random Forest", Xc, y, tr_, te); rows.append(dict(n_fft=nf, n_mfcc=str(nm) if fs != "logmel" else "-", fitur=fs, f1=rr["f1"], auc=rr["auc"], dim=Xc.shape[1]))
                bar.progress((i + 1) / len(combos))
            S["cmp_feat"] = pd.DataFrame(rows)
        if "cmp_feat" in S: st.dataframe(S["cmp_feat"].round(3), hide_index=True, width="stretch")
    st.warning("Hasil adalah prototipe indikatif, bukan sertifikasi alat. Label anomali tidak mengidentifikasi jenis kerusakan fisik.")

def page_predict():
    st.title("Prediksi Kondisi Kipas")
    if "model" not in S and MODEL_F.exists():
        o = joblib.load(MODEL_F); S.update(model=o["model"], cfg=o["cfg"], feat_names=o["names"])
    if "model" not in S:
        st.info("Latih model dulu di halaman *Eksperimen* (atau letakkan `model/rf_fan.joblib`)."); return
    c1, c2 = st.columns(2); lo = c1.slider("Batas WASPADA (P abnormal ≥)", .05, .6, MD.WARN_LOW, .05); hi = c2.slider("Batas ABNORMAL (P abnormal ≥)", .2, .95, MD.WARN_HIGH, .05)
    ups = st.file_uploader("Unggah satu/lebih WAV", type=["wav"], accept_multiple_files=True)
    for up in ups:
        y, sr = load_wav(up); v = validate_audio(y, sr)
        if any(l == "err" for l, _ in v):
            for _, m in v: st.error(f"{up.name}: {m}")
            continue
        y16 = resample_to(y, sr); x, _ = extract(y16, EXPECTED_SR, S["cfg"])
        p = float(S["model"].predict_proba(x[None])[0, 1]); label, lvl = MD.status(p, lo, hi)
        with st.container(border=True):
            a, b, c = st.columns([1.2, 1, 2])
            a.markdown(f"**{up.name}**"); {"ok": a.success, "warn": a.warning, "err": a.error}[lvl](f"{label}")
            b.metric("P(abnormal)", f"{p:.1%}")
            for l, m in v:
                if l != "ok": c.warning(m)
            P = 10 * np.log10(M.power_spec(M.stft(y16, 1024, 512)) + 1e-12)
            f = fig(180, margin=dict(l=0, r=0, t=0, b=0)); f.add_heatmap(z=P, colorscale="Viridis", showscale=False); f.update_layout(margin=dict(l=0, r=0, t=0, b=0)); c.plotly_chart(f, width="stretch")
            if lvl != "ok": st.caption("Saran: jadwalkan inspeksi (cek debu/filter, bearing, pemasangan). Model tidak menentukan penyebab fisik.")
        log_history(dict(waktu=dt.datetime.now().isoformat(timespec="seconds"), file=up.name, p_abnormal=round(p, 4), status=label,
                         set_fitur=S["cfg"].feature_set, n_fft=S["cfg"].n_fft, n_mfcc=S["cfg"].n_mfcc))
    st.caption("Status: NORMAL / WASPADA / ABNORMAL adalah pemetaan ambang probabilitas — bukan diagnosis fisik.")

def page_report():
    st.title("Histori Pengujian & Laporan")
    if HIST.exists():
        h = pd.read_csv(HIST); st.dataframe(h.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Unduh histori (CSV)", h.to_csv(index=False), "histori_prediksi.csv", "text/csv")
        st.plotly_chart(px.histogram(h, x="status", color="status", color_discrete_map={"NORMAL": BLUE, "WASPADA": "#E0A800", "ABNORMAL": ORANGE}, height=260), width="stretch")
    else: st.info("Belum ada histori prediksi.")
    st.subheader("Laporan eksperimen")
    if "result" not in S: st.caption("Latih model untuk membuat laporan."); return
    r, ce, cfg = S["result"], S["cfg_exp"], S["cfg"]; pred = (S["proba"] >= .5).astype(int); cm = skm.confusion_matrix(S["y"][S["te"]], pred, labels=[0, 1])
    md = f"""# Laporan Eksperimen — Klasifikasi Kondisi Kipas Pendingin Panel Elektronika
Tanggal: {dt.date.today()} · Dataset: MIMII (fan) · {len(S['d'])} klip ({int(S['d'].label.sum())} abnormal)

## Pipeline
WAV → validasi → resample 16 kHz → STFT (n_fft={cfg.n_fft}, hop={cfg.hop}, Hann) → mel ({cfg.n_mels}) → log → DCT-II (MFCC {cfg.n_mfcc}) + delta → mean/std klip → Random Forest

## Konsep Matematika 3
Sampling/Nyquist, DFT/FFT, window & STFT, skala mel + logaritma, DCT-II, impuritas Gini & voting ensambel, metrik evaluasi (P, R, F1, ROC-AUC).

## Model
Random Forest n_estimators={ce['n_est']}, max_depth={ce['depth']}, min_samples_leaf={ce['leaf']}; skema uji: {ce['mode']}; set fitur: {cfg.feature_set}

## Hasil (ambang 0.5)
| Metrik | Nilai |
|---|---|
| Akurasi | {r['akurasi']:.3f} |
| Presisi | {r['presisi']:.3f} |
| Recall | {r['recall']:.3f} |
| F1 | {r['f1']:.3f} |
| AUC | {r['auc']:.3f} |

Confusion matrix [[TN, FP],[FN, TP]] = {cm.tolist()}

## Batasan
Prototipe indikatif, bukan sertifikasi alat. Label anomali tidak mengidentifikasi jenis kerusakan fisik.
"""
    st.markdown(md); st.download_button("Unduh laporan (Markdown)", md, "laporan_nrp2.md", "text/markdown")

# ------------------------------------------------------------ navigasi
st.sidebar.markdown("## 🌀 Kondisi Kipas\n<span style='color:#6b7686;font-size:.85rem'>NRP 2 · Matematika 3 · D4 Elektronika</span>", unsafe_allow_html=True)
page = st.sidebar.radio("Halaman", ["Konsep Matematika", "Data & Validasi", "Analisis Sinyal", "Eksperimen", "Prediksi", "Histori & Laporan"], label_visibility="collapsed")
y, sr, src, name = active_signal()
if page == "Konsep Matematika": concepts.render(resample_to(y, sr), EXPECTED_SR, src)
elif page == "Data & Validasi": page_data()
elif page == "Analisis Sinyal": page_signal(y, sr, src, name)
elif page == "Eksperimen": page_experiment()
elif page == "Prediksi": page_predict()
else: page_report()
