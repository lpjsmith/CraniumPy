import numpy as np
import pandas as pd
import pyvista as pv
import pymeshlab as ml
import trimesh
from pathlib import Path


BINS_MM = np.array([1.0, 2.0, 3.0], dtype=float)
COLORS_RGB = np.array([
    [0,   200,   0],   # green  — <1 mm
    [255, 255,   0],   # yellow — 1–2 mm
    [255, 165,   0],   # orange — 2–3 mm
    [255,   0,   0],   # red    — ≥3 mm
], dtype=np.uint8)


def compute_per_vertex_distances(ref_path, tgt_path, signed=False):
    """
    Uses pymeshlab to compute per-vertex distances on tgt from ref surface.
    Returns (per_vertex_distances ndarray, stats dict).
    Stats are always unsigned. Per-vertex array is signed when signed=True.
    """
    ms = ml.MeshSet()
    ms.load_new_mesh(str(ref_path))   # mesh 0 — reference
    ms.load_new_mesh(str(tgt_path))   # mesh 1 — target

    vcount = ms.mesh(1).vertex_number()
    kwargs = dict(
        sampledmesh=1,
        targetmesh=0,
        savesample=True,
        samplevert=True,
        sampleedge=False,
        sampleface=False,
        samplenum=max(vcount, 1),
    )
    try:
        ms.apply_filter('get_hausdorff_distance', **kwargs)
    except TypeError:
        kwargs.pop('maxdist', None)
        ms.apply_filter('get_hausdorff_distance', **kwargs)

    sample_dists = np.asarray(
        ms.mesh(ms.number_meshes() - 1).vertex_scalar_array(), dtype=float
    )
    if sample_dists.size:
        stats = {
            'min_mm':  round(float(sample_dists.min()),  3),
            'max_mm':  round(float(sample_dists.max()),  3),
            'mean_mm': round(float(sample_dists.mean()), 3),
            'rms_mm':  round(float(np.sqrt((sample_dists ** 2).mean())), 3),
        }
    else:
        stats = {'min_mm': 0.0, 'max_mm': 0.0, 'mean_mm': 0.0, 'rms_mm': 0.0}

    ms.apply_filter(
        'compute_scalar_by_distance_from_another_mesh_per_vertex',
        measuremesh=1,
        refmesh=0,
        signeddist=signed,
    )
    per_vertex = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)
    return per_vertex, stats


def save_colored_ply(tgt_path, distances, out_path, cmap=None, clim=None):
    """
    Save a face-coloured PLY via trimesh.
    cmap=None        → 4-bin green/yellow/orange/red (ERN/batch mode)
    cmap='coolwarm'  → continuous matplotlib colormap; clim=(min, max)
    """
    mesh = trimesh.load_mesh(str(tgt_path), process=False)
    face_distances = np.mean(distances[mesh.faces], axis=1)

    if cmap is not None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        vmin, vmax = clim if clim is not None else (face_distances.min(), face_distances.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cm.get_cmap(cmap)(norm(face_distances))
        face_colors = (rgba[:, :3] * 255).astype(np.uint8)
    else:
        bin_idx = np.clip(np.digitize(face_distances, BINS_MM), 0, len(COLORS_RGB) - 1)
        face_colors = COLORS_RGB[bin_idx]

    new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)
    expanded_colors = np.repeat(face_colors, 3, axis=0)

    colored = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        vertex_colors=expanded_colors,
        process=False,
    )
    colored.export(str(out_path))


def save_screenshot_sheet(mesh_pv, distances, output_path, title='', cmap=None, clim=None):
    """
    Render 4 orthographic views arranged as a projection layout:
        [     ] [ Top  ] [      ]
        [ Left ] [Front ] [Right ]

    cmap=None        → discrete 4-bin green/yellow/orange/red (ERN/batch mode)
    cmap='coolwarm'  → continuous colormap with colorbar (single-comparison mode)
    clim=(min, max)  → fix the colormap range (e.g. symmetric about zero)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mesh_col = mesh_pv.copy()
        if cmap is not None:
            mesh_col['distance_mm'] = distances
            mesh_kw = dict(scalars='distance_mm', cmap=cmap, show_scalar_bar=False,
                           clim=list(clim) if clim is not None else None)
        else:
            bin_idx = np.clip(np.digitize(distances, BINS_MM), 0, len(COLORS_RGB) - 1)
            mesh_col['dist_rgb'] = COLORS_RGB[bin_idx]
            mesh_kw = dict(scalars='dist_rgb', rgb=True, show_scalar_bar=False)

        views = {
            'Top':   lambda pl: pl.view_xz(True),
            'Left':  lambda pl: pl.view_zy(True),
            'Front': lambda pl: pl.view_xy(),
            'Right': lambda pl: pl.view_zy(),
        }

        imgs = {}
        for view_name, set_view in views.items():
            try:
                pl = pv.Plotter(off_screen=True, window_size=(600, 600))
                pl.set_background('white')
                pl.add_mesh(mesh_col.copy(), **mesh_kw)
                set_view(pl)
                pl.reset_camera()
                img = pl.screenshot(return_img=True)
                if view_name == 'Top':
                    img = np.rot90(img, 2)
                imgs[view_name] = img
                pl.close()
            except Exception as e:
                print(f'  Screenshot {view_name} failed: {e}')

        if not imgs:
            print('  Screenshot sheet skipped (all views failed)')
            return

        # 2-row, 3-col grid: Top centred above Front, Left and Right flanking
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        if title:
            fig.suptitle(title, fontsize=12)

        for ax in axes.flat:
            ax.axis('off')

        layout = {
            'Top':   (0, 1),
            'Left':  (1, 0),
            'Front': (1, 1),
            'Right': (1, 2),
        }
        for view_name, (row, col) in layout.items():
            if view_name in imgs:
                axes[row, col].imshow(imgs[view_name])
                axes[row, col].set_title(view_name, fontsize=10)

        if cmap is not None:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            vmin, vmax = clim if clim is not None else (distances.min(), distances.max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=list(axes.flat), orientation='horizontal',
                                fraction=0.02, pad=0.02, shrink=0.5)
            cbar.set_label('Distance (mm)', fontsize=9)
            plt.tight_layout(rect=[0, 0.06, 1, 1])
        else:
            legend_labels = ['<1 mm', '1–2 mm', '2–3 mm', '≥3 mm']
            handles = [plt.Rectangle((0, 0), 1, 1, color=c / 255.0) for c in COLORS_RGB]
            fig.legend(handles, legend_labels, loc='lower center', ncol=4,
                       fontsize=9, frameon=False)
            plt.tight_layout(rect=[0, 0.04, 1, 1])

        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Sheet saved: {Path(output_path).name}')
    except Exception as e:
        print(f'  Screenshot sheet error: {e}')


def run_pairwise_hausdorff(mesh_paths, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    mesh_paths = [Path(p) for p in mesh_paths]
    meshes_pv = {p: pv.read(str(p)) for p in mesh_paths}

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

            distances, stats = compute_per_vertex_distances(ref_path, tgt_path)
            per_vertex_all[tgt_path].append(distances)

            all_metrics.append({
                'Reference': ref_path.name,
                'Target':    tgt_path.name,
                **stats,
            })

    mae_per_target = {}
    for tgt_path, dist_list in per_vertex_all.items():
        if not dist_list:
            continue
        stack = np.stack(dist_list, axis=0)
        mae_dists = np.mean(np.abs(stack), axis=0)
        mae_per_target[tgt_path] = mae_dists

        mae_ply = output_folder / f'mae_heatmap_{tgt_path.stem}.ply'
        save_colored_ply(tgt_path, mae_dists, mae_ply)

        save_screenshot_sheet(
            meshes_pv[tgt_path], mae_dists,
            output_folder / f'mae_heatmap_{tgt_path.stem}_sheet.png',
            title=f'MAE Hausdorff (vs all): {tgt_path.stem}',
        )
        print(f'  Saved MAE heatmap + sheet: {tgt_path.stem}')

    csv_path = output_folder / 'hausdorff_metrics.csv'
    pd.DataFrame(all_metrics).to_csv(str(csv_path), index=False)
    print(f'  CSV saved: {csv_path}')

    return all_metrics, mae_per_target
