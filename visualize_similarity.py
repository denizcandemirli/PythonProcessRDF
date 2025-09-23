# visualize_similarity.py
# Final similarity'den ısı haritası + dendrogram üretir ve matrisi CSV olarak kaydeder.
import os, argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def main():
    parser = argparse.ArgumentParser(description="Heatmap + dendrogram from final similarity CSV")
    parser.add_argument(
        "--sim-dir",
        default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
        help="similarity_combined_weighted.csv dosyasının bulunduğu klasör (varsayılan: ./repro_pack/output/02 - Similarity)"
    )
    args = parser.parse_args()

    sim_dir = os.path.abspath(args.sim_dir)
    os.makedirs(sim_dir, exist_ok=True)

    csv_path = os.path.join(sim_dir, "similarity_combined_weighted.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)

    # Model listesini ve simetriği oluştur
    models = sorted(set(df["model_a"]).union(set(df["model_b"])))
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    M = np.zeros((n, n), dtype=float)
    for _, r in df.iterrows():
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        M[i, j] = float(r["final_similarity"])
        M[j, i] = float(r["final_similarity"])
    np.fill_diagonal(M, 1.0)

    # Isı haritası
    plt.figure(figsize=(7, 6))
    im = plt.imshow(M, interpolation="nearest")  # varsayılan colormap
    plt.title("Final Similarity – Heatmap")
    plt.xticks(ticks=np.arange(n), labels=models, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(n), labels=models)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_path = os.path.join(sim_dir, "heatmap_final_similarity.png")
    plt.savefig(heatmap_path, dpi=160)
    plt.close()

    # Dendrogram (mesafe = 1 - benzerlik)
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)

    def plot_dendrogram_with_fallback(D, labels, out_path):
        n = D.shape[0]
        if SCIPY_OK and n >= 2:
            Z = linkage(squareform(D, checks=False), method="average")
            plt.figure(figsize=(8, 5))
            dendrogram(Z, labels=labels, leaf_rotation=45)
            plt.title("Hierarchical Clustering – Dendrogram (1 - final similarity)")
            plt.tight_layout()
            plt.savefig(out_path, dpi=160)
            plt.close()
            return

        # SciPy yoksa basit average-linkage fallback
        clusters = {i: [i] for i in range(n)}
        active = list(range(n))
        merges = []
        def clust_dist(a, b):
            pairs = [(i, j) for i in clusters[a] for j in clusters[b]]
            return float(np.mean([D[i, j] for (i, j) in pairs]))
        next_id = n
        while len(active) > 1:
            best = None; pair = None
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    a, b = active[i], active[j]
                    dist = clust_dist(a, b)
                    if best is None or dist < best:
                        best = dist; pair = (a, b)
            a, b = pair
            merges.append((a, b, best, len(clusters[a]) + len(clusters[b])))
            clusters[next_id] = clusters[a] + clusters[b]
            active = [x for x in active if x not in (a, b)] + [next_id]
            next_id += 1

        parent = {}; height = {}; children = {}; root = next_id - 1
        for i in range(n): height[i] = 0.0
        cur = n
        for a, b, dist, size in merges:
            parent[a] = cur; parent[b] = cur; height[cur] = dist; children[cur] = (a, b); cur += 1

        order = []
        def inorder(node):
            if node < n: order.append(node)
            else:
                l, r = children[node]; inorder(l); inorder(r)
        inorder(root)
        posx = {leaf: i for i, leaf in enumerate(order)}
        def xpos(node):
            if node < n: return posx[node]
            l, r = children[node]; return 0.5 * (xpos(l) + xpos(r))
        def draw(node):
            if node < n: return
            l, r = children[node]; xl, xr = xpos(l), xpos(r)
            hl, hr = height[l], height[r]; hn = height[node]
            plt.plot([xl, xl], [hl, hn]); plt.plot([xr, xr], [hr, hn]); plt.plot([xl, xr], [hn, hn])
            draw(l); draw(r)
        plt.figure(figsize=(8, 5)); draw(root)
        plt.xticks(ticks=[posx[i] for i in order], labels=[labels[i] for i in order], rotation=45, ha="right")
        plt.ylabel("Distance (1 - similarity)"); plt.title("Hierarchical Clustering – Dendrogram (fallback)")
        plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

    dendro_path = os.path.join(sim_dir, "dendrogram_final_similarity.png")
    plot_dendrogram_with_fallback(D, models, dendro_path)

    # MATRİSİ CSV OLARAK KAYDET
    matrix_csv = os.path.join(sim_dir, "matrix_final_similarity.csv")
    pd.DataFrame(M, index=models, columns=models).to_csv(matrix_csv)

    print("✅ Kaydedildi:")
    print(" -", heatmap_path)
    print(" -", dendro_path)
    print(" -", matrix_csv)

if __name__ == "__main__":
    main()
