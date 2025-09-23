# export_m5_to_long.py
# Amaç: struct_functions_shares_wide.csv içindeki LB/Shear/Moment/Bracing paylarını
#       struct_motif_shares_long.csv dosyasına M5_* motifleri olarak enjekte etmek.
# Kullanım:
#   python export_m5_to_long.py --dir ".\repro_pack\output\07 - Structural_Extension_v25p2"
#
# Çıktılar:
#  - struct_motif_shares_long_m5.csv  (yeni dosya; M5_* satırları eklenmiş)
#  - struct_motif_shares_long.csv     (varsayılan: ÜZERİNE YAZMAK için --overwrite kullanın)

import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="v25p2 çıktılarının olduğu klasör")
    ap.add_argument("--in-long", default="struct_motif_shares_long.csv")
    ap.add_argument("--in-func-wide", default="struct_functions_shares_wide.csv")
    ap.add_argument("--out-long", default="struct_motif_shares_long_m5.csv",
                    help="Varsayılan olarak yeni bir dosyaya yazar; --overwrite ile orijinali ezebilirsiniz.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Açılırsa struct_motif_shares_long.csv üzerine yazar.")
    args = ap.parse_args()

    d = args.dir
    f_long = os.path.join(d, args.in_long)
    f_wide = os.path.join(d, args.in_func_wide)

    if not os.path.isfile(f_long):
        raise FileNotFoundError(f"Bulunamadı: {f_long}")
    if not os.path.isfile(f_wide):
        raise FileNotFoundError(f"Bulunamadı: {f_wide}")

    df_long = pd.read_csv(f_long)
    # Beklenen kolonlar: model, motif, share
    expect_cols = {"model","motif","share"}
    if not expect_cols.issubset(set(df_long.columns)):
        raise ValueError(f"{args.in_long} beklenen kolonlara sahip değil (model/motif/share). "
                         f"Algılanan: {list(df_long.columns)}")

    df_wide = pd.read_csv(f_wide)
    # Beklenen kolonlar: model, LB, Shear, Moment, Bracing (Moment/Bracing eksik olabilir → 0'a doldur)
    if "model" not in df_wide.columns:
        raise ValueError(f"{args.in_func_wide} dosyasında 'model' kolonu yok.")
    for c in ["LB","Shear","Moment","Bracing"]:
        if c not in df_wide.columns:
            df_wide[c] = 0.0
    # Uzun forma dök ve M5_* motif adlarına map et
    func_long = df_wide.melt(id_vars="model", value_vars=["LB","Shear","Moment","Bracing"],
                             var_name="func", value_name="share")
    func_long["motif"] = func_long["func"].map({
        "LB": "M5_LB",
        "Shear": "M5_Shear",
        "Moment": "M5_Moment",
        "Bracing": "M5_Bracing",
    })
    func_long = func_long[["model","motif","share"]]

    # Var olan uzun forma M5_* satırlarını enjekte et.
    # Önce M5_* olanları df_long'dan düş, sonra func_long ile birleştir.
    df_non_m5 = df_long[~df_long["motif"].str.startswith("M5_", na=False)].copy()
    df_out = pd.concat([df_non_m5, func_long], ignore_index=True)

    # Stabil sıralama
    motif_order = ["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode",
                   "M5_LB","M5_Shear","M5_Moment","M5_Bracing"]
    df_out["motif"] = pd.Categorical(df_out["motif"], categories=motif_order, ordered=True)
    df_out = df_out.sort_values(["model","motif"]).reset_index(drop=True)

    # Yaz
    out_path = os.path.join(d, args.out_long)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] M5 enjekte edildi → {out_path}")

    # İstenirse üzerine yaz
    if args.overwrite:
        orig_path = os.path.join(d, "struct_motif_shares_long.csv")
        df_out.to_csv(orig_path, index=False)
        print(f"[OK] Üzerine yazıldı → {orig_path}")

if __name__ == "__main__":
    main()
