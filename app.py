"""Dashboard NRP 2 — Klasifikasi Kondisi Kipas Pendingin Panel Elektronika (MIMII fan)."""
import io, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, joblib
import plotly.graph_objects as go, plotly.express as px
import streamlit as st
from sklearn import metrics as skm

from core import mathcore as M
from core.data import scan_dataset, load_wav, validate_audio, resample_to, EXPECTED_SR
from core.features import FeatureConfig, FEATURE_SETS, build_matrix, extract
from core import models as MD
from core.synth import fan
from views import concepts

st.set_page_config(page_title="Kondisi Kipas · Mat 3", page_icon="🌀", layout="wide")
BLUE, ORANGE, GREY = "#1F5FBF", "#E4572E", "#8A94A6"
ROOT = Path(__file__).parent
HIST = ROOT / "history.csv"; MODEL_F = ROOT / "model" / "rf_fan.joblib"
st.markdown("<style>.block-container{padding-top:1.6rem;max-width:1250px} h1{letter-spacing:-.5px}</style>", unsafe_allow_html=True)
S = st.session_state

def fig(h=320, **kw):
    f = go.Figure(); f.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", **kw); return f

# ------------------------------------------------------------ sinyal aktif
def active_signal():
    side = st.sidebar
    side.markdown("### Sinyal aktif")
    mode = side.radio("Sumber", ["Contoh sintetis", "Unggah WAV", "Dari dataset"], label_visibility="collapsed")
    if mode == "Unggah WAV":
        up = side.file_uploader("WAV", type=["wav"], key="sig_up")
        if up:
            y, sr = load_wav(up); return y, sr, f"unggahan: {up.name}", up.name
    elif mode == "Dari dataset" and "df" in S and len(S["df"]):
        df = S["df"]; p = side.selectbox("File", df.path, format_func=lambda s: "/".join(s.split("/")[-4:]))
        y, sr = load_wav(p); return y, sr, f"dataset: {'/'.join(p.split('/')[-3:])}", p
    kind = side.selectbox("Jenis contoh", ["normal", "imbalance", "rattle", "friction"]) if mode == "Contoh sintetis" else "normal"
    y = fan(np.random.default_rng(7), kind, f0=50.0)
    return y, 16000, f"sintetis ({kind}) — bukan data MIMII", f"synthetic_{kind}.wav"

def cfg_widgets(prefix="", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    fs_name = c1.selectbox("Set fitur", list(FEATURE_SETS), key=prefix + "fs")
    n_fft = c2.select_slider("n_fft", [512, 1024, 2048], 1024, key=prefix + "nfft")
    n_mfcc = c3.slider("n_mfcc", 8, 30, 13, key=prefix + "nm")
    n_mels = c4.select_slider("n_mels", [32, 64, 96], 64, key=prefix + "nmel")
    return FeatureConfig(n_fft=n_fft, hop=n_fft // 2, n_mels=n_mels, n_mfcc=n_mfcc, feature_set=FEATURE_SETS[fs_name])

def log_history(rec):
    df = pd.DataFrame([rec])
    df.to_csv(HIST, mode="a", header=not HIST.exists(), index=False)

# ------------------------------------------------------------ halaman
def page_data():
    st.title("Data & Validasi")
    st.markdown("Struktur yang dibaca: `<folder>/.../fan/id_XX/{normal,abnormal}/*.wav` (hasil ekstrak zip MIMII, termasuk varian `-6_dB`, `0_dB`, `6_dB`). "
                "Label hanya dipakai untuk **evaluasi/supervisi**, masukan utama tetap sinyal suara.")
    root = st.text_input("Folder dataset", S.get("root", "./dataset"), help="Folder hasil ekstrak MIMII (zip dari Zenodo).")
    if st.button("Pindai dataset", type="primary"):
        S["root"] = root; S["df"] = scan_dataset(root, "fan")
    df = S.get("df")
    if df is None:
        st.info("Belum ada data. Unduh `fan` dari https://zenodo.org/records/3384388, ekstrak, lalu isi path di atas. "
                "Untuk uji coba cepat: `python tools/make_synthetic.py --out demo_synthetic` (data SINTETIS, hanya untuk tes aplikasi)."); return
    if df.empty:
        st.error("Tidak ditemukan file normal/abnormal untuk 'fan'. Periksa path & struktur folder."); return
    c = st.columns(4); c[0].metric("Total klip", len(df)); c[1].metric("Normal", int((df.label == 0).sum()))
    c[2].metric("Abnormal", int((df.label == 1).sum())); c[3].metric("Rasio abnormal", f"{df.label.mean():.1%}")
    g = df.groupby(["id", "snr", "kondisi"], dropna=False).size().reset_index(name="n")
    g["snr"] = g["snr"].astype(str)
    st.plotly_chart(px.bar(g, x="id", y="n", color="kondisi", facet_col="snr", barmode="group",
                           color_discrete_map={"normal": BLUE, "abnormal": ORANGE}, height=320), width="stretch")
    if df.label.nunique() < 2: st.error("Hanya satu kelas ditemukan — klasifikasi butuh normal dan abnormal.")
    if df.label.mean() < 0.15 or df.label.mean() > 0.85: st.warning("Kelas sangat tidak seimbang; gunakan F1/AUC & `class_weight=balanced`.")
    if st.checkbox("Validasi sampel file (sampling rate, durasi, clipping)"):
        rows = []
        for p in df.sample(min(25, len(df)), random_state=0).path:
            y, sr = load_wav(p); v = validate_audio(y, sr); rows.append(dict(file="/".join(p.split("/")[-3:]), sr=sr, durasi=round(len(y)/sr, 1), status=v[0][0], catatan=v[0][1]))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.dataframe(df.assign(path=df.path.str.split("/").str[-4:].str.join("/")).head(200), hide_index=True, width="stretch")

def page_signal(y, sr, src, name):
    st.title("Analisis Sinyal")
    st.caption(f"Sumber: {src}")
    for lv, msg in validate_audio(y, sr): {"ok": st.success, "warn": st.warning, "err": st.error}[lv](msg)
    y16 = resample_to(y, sr)
    cfg = cfg_widgets("sig_")
    st.audio(y16, sample_rate=EXPECTED_SR)
    t = np.arange(len(y16)) / EXPECTED_SR
    c1, c2 = st.columns(2)
    with c1:
        f = fig(260, title="Waveform", xaxis_title="detik"); step = max(1, len(y16) // 6000)
        f.add_scatter(x=t[::step], y=y16[::step], line=dict(color=BLUE, width=1)); st.plotly_chart(f, width="stretch")
    with c2:
        sp = np.abs(np.fft.rfft(y16 * M.hann(len(y16)))) / len(y16); fr = np.fft.rfftfreq(len(y16), 1 / EXPECTED_SR)
        f = fig(260, title="Spektrum (FFT seluruh klip, Hann)", xaxis_title="Hz", yaxis_title="dB")
        f.add_scatter(x=fr, y=20 * np.log10(sp + 1e-12), line=dict(color=BLUE, width=1)); st.plotly_chart(f, width="stretch")
    P = 10 * np.log10(M.power_spec(M.stft(y16, cfg.n_fft, cfg.hop)) + 1e-12)
    L = M.log_mel(y16, EXPECTED_SR, cfg.n_fft, cfg.hop, cfg.n_mels); C = M.mfcc(y16, EXPECTED_SR, cfg.n_fft, cfg.hop, cfg.n_mels, cfg.n_mfcc)
    tx = np.arange(P.shape[1]) * cfg.hop / EXPECTED_SR
    c1, c2, c3 = st.columns(3)
    for col, Z, ttl, cs, ex in ((c1, P, "Spektrogram STFT (dB)", "Viridis", dict(y=np.arange(P.shape[0]) * EXPECTED_SR / cfg.n_fft)),
                                (c2, L, "log-mel (dB)", "Viridis", {}), (c3, C, "MFCC", "RdBu", {})):
        f = fig(300, title=ttl); f.add_heatmap(z=Z, x=tx, colorscale=cs, showscale=False, **ex)
        if Z is C: f.data[0].update(zmid=0)
        col.plotly_chart(f, width="stretch")
    v, names = extract(y16, EXPECTED_SR, cfg)
    sp_ = M.spectral_features(y16, EXPECTED_SR, cfg.n_fft, cfg.hop)
    st.markdown("**Ringkasan fitur spektral**")
    st.dataframe(pd.DataFrame({k: [a.mean(), a.std()] for k, a in sp_.items()}, index=["mean", "std"]).round(4), width="stretch")
    st.caption(f"Vektor fitur klip: {len(v)} dimensi (set: {cfg.feature_set}).")

def page_experiment():
    st.title("Eksperimen & Evaluasi")
    df = S.get("df")
    if df is None or df.empty or df.label.nunique() < 2:
        st.info("Pindai dataset yang memiliki kelas normal & abnormal di halaman *Data & Validasi*."); return
    with st.expander("1 · Seleksi data & konfigurasi fitur", expanded=True):
        c1, c2 = st.columns(2)
        ids = c1.multiselect("ID mesin", sorted(df.id.unique()), default=sorted(df.id.unique()))
        snrs = sorted(df.snr.dropna().unique().tolist()); snr_sel = c2.multiselect("SNR (dB)", snrs, default=snrs) if snrs else []
        d = df[df.id.isin(ids)]; d = d[d.snr.isin(snr_sel)] if snrs else d
        max_n = st.slider("Batasi jumlah klip (cepat)", 100, max(len(d), 101), min(len(d), 1500), 50)
        if len(d) > max_n: d = d.groupby("label", group_keys=False).apply(lambda g: g.sample(int(max_n * len(g) / len(d)), random_state=0))
        cfg = cfg_widgets("exp_")
    with st.expander("2 · Model & pembagian data", expanded=True):
        c = st.columns(5)
        n_est = c[0].select_slider("n_estimators", [10, 50, 100, 200, 400], 200); depth = c[1].select_slider("max_depth", [None, 5, 10, 20], None)
        leaf = c[2].select_slider("min_samples_leaf", [1, 2, 5], 1)
        mode = c[3].radio("Skema uji", ["acak", "lintas-id"], help="lintas-id: ID mesin uji tidak ada di data latih (lebih ketat)")
        tsz = c[4].slider("Porsi uji", .1, .5, .25, .05)
    if st.button("Ekstrak fitur & latih Random Forest", type="primary"):
        bar = st.progress(0.0, "Ekstraksi STFT→MFCC…")
        X, names = build_matrix(d.path.tolist(), cfg, cache_dir=str(ROOT / "cache"), progress=lambda p: bar.progress(p, "Ekstraksi STFT→MFCC…"))
        bar.empty()
        y = d.label.values; groups = d.id.values
        tr, te = MD.split(X, y, groups, mode, tsz)
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            st.error("Pembagian menghasilkan satu kelas saja pada latih/uji. Ubah skema/porsi uji."); return
        mdl, proba, r = MD.fit_eval("Random Forest", X, y, tr, te, n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf)
        MODEL_F.parent.mkdir(exist_ok=True); joblib.dump(dict(model=mdl, cfg=cfg, names=names), MODEL_F)
        S.update(model=mdl, cfg=cfg, feat_names=names, X=X, y=y, tr=tr, te=te, proba=proba, result=r, d=d.reset_index(drop=True),
                 cfg_exp=dict(n_est=n_est, depth=depth, leaf=leaf, mode=mode))
    if "result" not in S: return
    r, y, te, proba, d = S["result"], S["y"], S["te"], S["proba"], S["d"]
    thr = st.slider("Ambang keputusan abnormal", .05, .95, .5, .05)
    r = MD.evaluate(y[te], proba, thr) | {"waktu_latih_s": S["result"]["waktu_latih_s"]}
    st.subheader("Hasil pada data uji")
    cols = st.columns(6)
    for col, (k, lab) in zip(cols, [("akurasi", "Akurasi"), ("presisi", "Presisi"), ("recall", "Recall"), ("f1", "F1"), ("bal_acc", "Bal. Acc"), ("auc", "AUC")]):
        col.metric(lab, f"{r[k]:.3f}")
    t1, t2, t3, t4 = st.tabs(["Confusion matrix & ROC", "Fitur penting", "Per ID × SNR", "Perbandingan model & parameter"])
    with t1:
        a, b = st.columns(2)
        pred = (proba >= thr).astype(int); cm = skm.confusion_matrix(y[te], pred, labels=[0, 1])
        f = fig(330, title="Confusion matrix (baris: sebenarnya)")
        f.add_heatmap(z=cm, x=["Normal", "Abnormal"], y=["Normal", "Abnormal"], text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False)
        f.update_yaxes(autorange="reversed"); a.plotly_chart(f, width="stretch")
        fpr, tpr, _ = skm.roc_curve(y[te], proba); f = fig(330, title=f"Kurva ROC (AUC={r['auc']:.3f})", xaxis_title="FPR", yaxis_title="TPR")
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
