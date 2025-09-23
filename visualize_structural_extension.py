# visualize_structural_extension.py
"""
Structural Extension görselleri:
- Motif payları ısı haritası (model x motif)
- Dendrogram (struct_similarity_matrix.csv varsa onu, yoksa motif-kosinüs benzerlikten üretir)

Sağlam CSV okuyucu:
- Ayırıcıyı otomatik dener (',',';','\\t','|')
- struct_motif_shares.csv hem LONG (model,motif,share) hem de WIDE (model,M2_*,...) biçimini destekler
- struct_similarity_matrix.csv hem WIDE (kare matris) hem de LONG (A,B,similarity) desteklenir
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


# ---------- Yardımcılar ----------

def read_csv_smart(path, must_exist=True):
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(f"Yok: {path}")
    last_err = None
    for sep in [',', ';', '\t', '|']:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            return df
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"CSV okunamadı: {path} (son hata: {last_err})")


def normalize_header(df: pd.DataFrame):
    # Sütun adlarını küçük harfe indirip strip edelim
    mapping = {c: c.strip() for c in df.columns}
    df = df.rename(columns=mapping)
    low = {c: c.lower() for c in df.columns}
    # Orijinal isimleri koruyarak küçük harf eşlemelerini döndürelim
    inv = {}
    for orig, lowc in low.items():
        inv[lowc] = orig
    return df, inv


# ---------- Motif payları (LONG/WIDE algılama) ----------

def read_motif_shares(in_dir: str) -> pd.DataFrame:
    """
    struct_motif_shares.csv dosyasını okur ve MUTLAKA pivot (model x motif) döndürür.
    - LONG biçim: model, motif, share  --> pivot
    - WIDE biçim: model, M2_*, M3_*, ... doğrudan pivot sayılır
    """
    path = os.path.join(in_dir, "struct_motif_shares.csv")
    df = read_csv_smart(path)

    # Başlıkları normalize edip LONG mu WIDE mı algılayalım
    df2, inv = normalize_header(df)
    cols_lower = [c.lower() for c in df2.columns]

    has_model = ("model" in cols_lower)
    has_motif = ("motif" in cols_lower)
    has_share = ("share" in cols_lower)

    if has_model and has_motif and has_share:
        # LONG -> pivot
        model_col = inv["model"]
        motif_col = inv["motif"]
        share_col = inv["share"]
        df2[share_col] = pd.to_numeric(df2[share_col], errors="coerce")
        df2 = df2.dropna(subset=[share_col])
        piv = df2.pivot_table(index=model_col, columns=motif_col,
                              values=share_col, aggfunc="mean", fill_value=0.0)
        # motif kolon sırasını doğal sıraya yaklaştır
        piv = piv.reindex(sorted(piv.columns, key=str), axis=1)
        return piv

    # WIDE: 'model' + motif kolonları
    if has_model:
        model_col = inv["model"]
        # motif kolonları = model dışındaki tüm sayısal kolonlar
        motif_cols = [c for c in df2.columns if c != model_col]
        if len(motif_cols) == 0:
            raise ValueError("struct_motif_shares.csv WIDE görünüyor ama motif kolonu bulunamadı.")
        # sayıya çevir, NaN->0
        for c in motif_cols:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)
        piv = df2.set_index(model_col)[motif_cols].copy()
        # motif kolon sırası
        piv = piv.reindex(sorted(piv.columns, key=str), axis=1)
        return piv

    # Buraya düşüyorsa beklenen başlıklar yok
    raise ValueError(
        "struct_motif_shares.csv beklenen biçimlerde değil. "
        "LONG: (model,motif,share) veya WIDE: (model,M2_*,M3_*,...)")


# ---------- Yapısal benzerlik matrisi (WIDE/LONG algılama) ----------

def read_struct_similarity(in_dir: str) -> tuple[pd.DataFrame | None, list[str] | None]:
    """
    struct_similarity_matrix.csv:
    - WIDE: 1. sütun model adı, diğerleri kare matris
    - LONG: kolonlar (A,B,similarity) veya benzeri → pivot
    Dönen: kare pandas DataFrame ve etiket listesi (sıra)
    """
    path = os.path.join(in_dir, "struct_similarity_matrix.csv")
    if not os.path.exists(path):
        return None, None

    df = read_csv_smart(path)
    df2, inv = normalize_header(df)
    cols = [c.lower() for c in df2.columns]

    # LONG form olasılıkları
    long_keys = [("a", "b", "similarity"), ("model_a", "model_b", "similarity"),
                 ("left", "right", "similarity")]

    is_long = False
    for a, b, s in long_keys:
        if a in cols and b in cols and s in cols:
            A = inv[a]; B = inv[b]; S = inv[s]
            tmp = df2[[A, B, S]].copy()
            tmp[S] = pd.to_numeric(tmp[S], errors="coerce").fillna(0.0)
            mat = tmp.pivot_table(index=A, columns=B, values=S, aggfunc="mean", fill_value=0.0)
            # kare yap, simetrikle
            labels = sorted(set(mat.index).union(set(mat.columns)))
            mat = mat.reindex(index=labels, columns=labels, fill_value=0.0)
            # simetri (maksimumu al)
            mat = np.maximum(mat.values, mat.values.T)
            mat = pd.DataFrame(mat, index=labels, columns=labels)
            return mat, labels
    # Eğer LONG değilse WIDE varsayalım: ilk kolonda başlık gibi model adları olabilir
    # index kolonunu seç
    # Heuristik: 'model' adlı kolon varsa onu index yap
    if "model" in cols:
        idx_col = inv["model"]
    else:
        # ilk sütun model etiketi olarak kullanılır
        idx_col = df2.columns[0]

    mat = df2.set_index(idx_col)
    # sayıya çevir
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0.0)

    # kare mi?
    labels = list(mat.index)
    # Eğer sütunlar da aynı etiketleri taşımıyorsa, kolonları da aynı sıraya çek
    if set(mat.columns) != set(labels):
        cols_inter = sorted(set(mat.columns).intersection(set(labels)))
        mat = mat.reindex(index=labels, columns=cols_inter, fill_value=0.0)
        # eksik kolon varsa genişlet
        for lab in labels:
            if lab not in mat.columns:
                mat[lab] = 0.0
        mat = mat[labels]
    else:
        mat = mat.reindex(columns=labels)

    # Simetri güvence
    mat = pd.DataFrame(np.maximum(mat.values, mat.values.T), index=labels, columns=labels)
    return mat, labels


# ---------- Görseller ----------

def plot_heatmap(piv: pd.DataFrame, out_path: str, title: str = "Motif Shares (row-normalized)"):
    plt.figure(figsize=(10, 4 + 0.3 * len(piv)))
    im = plt.imshow(piv.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right")
    plt.yticks(range(piv.shape[0]), piv.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def cosine_similarity_from_rows(X: np.ndarray) -> np.ndarray:
    # satır vektörleri üzerinden cosine
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Y = X / norms
    return Y @ Y.T


def plot_dendrogram_from_matrix(sim: pd.DataFrame, out_path: str, title: str = "Structural Similarity – Dendrogram"):
    labels = list(sim.index)
    S = np.asarray(sim.values, dtype=float)
    # [0,1] bandında varsayıyoruz; mesafe = 1 - S
    D = 1.0 - np.clip(S, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    # condensed
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    plt.figure(figsize=(10, 4 + 0.25 * len(labels)))
    dendrogram(Z, labels=labels, orientation="right", color_threshold=None)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Motif payları pivotunu oku (LONG veya WIDE)
    piv = read_motif_shares(args.in_dir)
    # Her ihtimale karşı NaN -> 0
    piv = piv.fillna(0.0)

    # Isı haritası
    out_heat = os.path.join(args.out_dir, "struct_motif_heatmap.png")
    plot_heatmap(piv, out_heat, title="Structural Motif Shares")
    print(f"[OK] Heatmap saved: {out_heat}")

    # 2) Dendrogram:
    # Öncelik: struct_similarity_matrix.csv varsa onu kullan
    S_struct, labels = read_struct_similarity(args.in_dir)
    if S_struct is not None:
        out_den = os.path.join(args.out_dir, "struct_dendrogram.png")
        plot_dendrogram_from_matrix(S_struct, out_den, title="Structural Similarity (matrix) – Dendrogram")
        print(f"[OK] Dendrogram saved (struct matrix): {out_den}")
    else:
        # Yoksa motif payları üzerinden cosine → dendrogram
        S_cos = cosine_similarity_from_rows(piv.values)
        S_cos = pd.DataFrame(S_cos, index=piv.index, columns=piv.index)
        out_den = os.path.join(args.out_dir, "struct_dendrogram_from_motif.png")
        plot_dendrogram_from_matrix(S_cos, out_den, title="Structural Similarity (motif cosine) – Dendrogram")
        print(f"[OK] Dendrogram saved (motif cosine): {out_den}")


if __name__ == "__main__":
    main()
