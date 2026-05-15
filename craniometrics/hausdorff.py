import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


def per_vertex_distances(ref_mesh_pv, tgt_mesh_pv):
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ref_mesh_pv.points)
    distances, _ = nn.kneighbors(tgt_mesh_pv.points)
    return distances.squeeze()


def hausdorff_stats(distances, ref_name, tgt_name):
    return {
        'Reference': ref_name,
        'Target':    tgt_name,
        'min_mm':    round(float(distances.min()),  3),
        'max_mm':    round(float(distances.max()),  3),
        'mean_mm':   round(float(distances.mean()), 3),
        'rms_mm':    round(float(np.sqrt((distances ** 2).mean())), 3),
    }


def run_pairwise_hausdorff(mesh_paths, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    mesh_paths = [Path(p) for p in mesh_paths]
    meshes = {p: pv.read(str(p)) for p in mesh_paths}

    all_metrics = []
    per_vertex_all = {p: [] for p in mesh_paths}

    n = len(mesh_paths)
    total = n * (n - 1)
    count = 0

    for ref_path in mesh_paths:
        for tgt_path in mesh_paths:
            if ref_path == tgt_path:
                continue
            count += 1
            print(f'  [{count}/{total}] {ref_path.stem} → {tgt_path.stem}')

            distances = per_vertex_distances(meshes[ref_path], meshes[tgt_path])
            per_vertex_all[tgt_path].append(distances)

            stats = hausdorff_stats(distances, ref_path.name, tgt_path.name)
            all_metrics.append(stats)

            tgt_copy = meshes[tgt_path].copy()
            tgt_copy['hausdorff_mm'] = distances
            out = output_folder / f'heatmap_{ref_path.stem}_vs_{tgt_path.stem}.ply'
            tgt_copy.save(str(out))

    for tgt_path, dist_list in per_vertex_all.items():
        if not dist_list:
            continue
        stack = np.stack(dist_list, axis=0)

        avg_mesh = meshes[tgt_path].copy()
        avg_mesh['hausdorff_mean_mm'] = np.mean(stack, axis=0)
        avg_mesh.save(str(output_folder / f'avg_heatmap_{tgt_path.stem}.ply'))

        mae_mesh = meshes[tgt_path].copy()
        mae_mesh['hausdorff_mae_mm'] = np.mean(np.abs(stack), axis=0)
        mae_mesh.save(str(output_folder / f'mae_heatmap_{tgt_path.stem}.ply'))

        print(f'  Saved avg/MAE heatmaps: {tgt_path.stem}')

    csv_path = output_folder / 'hausdorff_metrics.csv'
    pd.DataFrame(all_metrics).to_csv(str(csv_path), index=False)
    print(f'  CSV saved: {csv_path}')

    return all_metrics
