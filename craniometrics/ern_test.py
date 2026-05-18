import numpy as np
import pandas as pd
import pyvista as pv
import json
from pathlib import Path
from collections import defaultdict

from craniometrics.hausdorff import run_pairwise_hausdorff, BINS_MM, COLORS_RGB
from craniometrics.craniometrics import CranioMetrics
from craniometrics.elawadly import compute_elawadly, POINT_LABELS


CRANIO_KEYS = ['OFD_depth_mm', 'BPD_breadth_mm', 'Cephalic_Index', 'Circumference_cm', 'MeshVolume_cc']

ELAW_SUMMARY_KEYS = [
    'MAP_circumference_cm', 'FA30_degrees',
    'APWR1', 'APWR2', 'rAPDR30', 'lAPDR30', 'rAPDR60', 'lAPDR60',
    'APAR', 'APVR',
]


def _get_front_view_img(mesh_path, distances):
    bin_idx = np.clip(np.digitize(distances, BINS_MM), 0, len(COLORS_RGB) - 1)
    mesh_pv = pv.read(str(mesh_path))
    mesh_pv['dist_rgb'] = COLORS_RGB[bin_idx]
    pl = pv.Plotter(off_screen=True, window_size=(300, 300))
    pl.set_background('white')
    pl.add_mesh(mesh_pv.copy(), scalars='dist_rgb', rgb=True, show_scalar_bar=False)
    pl.view_xy()
    pl.reset_camera()
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _run_craniometrics(mesh_paths):
    results = {}
    for p in mesh_paths:
        p = Path(p)
        try:
            m = CranioMetrics(p)
            m.extract_dimensions(m.slice_height)
            results[p.stem] = {
                'OFD_depth_mm':    round(float(m.depth), 2),
                'BPD_breadth_mm':  round(float(m.breadth), 2),
                'Cephalic_Index':  m.CI,
                'Circumference_cm': m.HC,
                'MeshVolume_cc':   round((m.pvmesh.volume / 1000), 2),
            }
            print(f'  Craniometrics computed: {p.name}')
        except Exception as e:
            print(f'  Craniometrics failed for {p.name}: {e}')
    return results


def _find_lmk_path(p):
    """Find a landmarks JSON for p.

    Strips _clipC or _clipF suffix first, then checks:
      1. {base}_landmarks_raw.json  — raw landmarks saved before registration
      2. {base}_landmarks.json      — landmarks saved by register()
    Falls back to checking the un-stripped stem too.
    """
    stem = p.stem
    if stem.endswith('_clipC') or stem.endswith('_clipF'):
        base = stem[:-6]
    else:
        base = stem
    for suffix in ('_landmarks_raw.json', '_landmarks.json'):
        for s in ([base, stem] if base != stem else [stem]):
            candidate = p.parent / (s + suffix)
            if candidate.exists():
                return candidate
    return None


def _read_lmk_coords(lmk):
    """Extract (nasion, tragus_left, tragus_right) from either landmark format."""
    if 'nasion' in lmk:
        return (np.array(lmk['nasion']),
                np.array(lmk['tragus_left']),
                np.array(lmk['tragus_right']))
    # registration format: new_nasion, new_lh_coord, new_rh_coord
    return (np.array(lmk['new_nasion']),
            np.array(lmk['new_lh_coord']),
            np.array(lmk['new_rh_coord']))


def _run_elawadly(mesh_paths):
    results = {}
    for p in mesh_paths:
        p = Path(p)
        lmk_path = _find_lmk_path(p)
        if lmk_path is None:
            print(f'  Elawadly: no landmarks for {p.name} — skipped')
            continue
        try:
            with open(lmk_path) as f:
                lmk = json.load(f)
            nasion, tragus_left, tragus_right = _read_lmk_coords(lmk)
            mesh_pv = pv.read(str(p))
            _, _, _, _, _, _, _, _, measurements = compute_elawadly(
                mesh_pv, nasion, tragus_right, tragus_left, p
            )
            results[p.stem] = measurements
            print(f'  Elawadly computed: {p.name}')
        except Exception as e:
            print(f'  Elawadly failed for {p.name}: {e}')
    return results


def _flatten_elawadly(m):
    flat = {}
    flat['MAP_circumference_cm'] = m.get('MAP_circumference_cm')
    flat['FA30_degrees'] = m.get('FA30_degrees')
    for k, v in m.get('ratios', {}).items():
        flat[k] = v
    for k, v in m.get('area_volume', {}).items():
        flat[k] = v
    return flat


def _hausdorff_weighted_means(all_metrics, mesh_paths):
    vertex_counts = {}
    for p in mesh_paths:
        p = Path(p)
        try:
            vertex_counts[p.name] = pv.read(str(p)).n_points
        except Exception:
            vertex_counts[p.name] = 1

    by_target = defaultdict(list)
    for row in all_metrics:
        by_target[row['Target']].append(row)

    results = {}
    for tgt, rows in by_target.items():
        total_w = sum(vertex_counts.get(r['Reference'], 1) for r in rows)
        wm = (
            sum(vertex_counts.get(r['Reference'], 1) * r['mean_mm'] for r in rows) / total_w
            if total_w > 0 else None
        )
        results[tgt] = {
            'weighted_mean_mm': round(wm, 3) if wm is not None else None,
            'pass': wm is not None and wm <= 2.0,
        }
    return results


def _equivalence_rows(data_dict, keys, stems, tol_pct=5.0):
    rows = []
    for key in keys:
        vals = {}
        for s in stems:
            d = data_dict.get(s) or {}
            v = d.get(key)
            if v is not None:
                try:
                    vals[s] = float(v)
                except (TypeError, ValueError):
                    pass
        if not vals:
            # No data for any mesh — include row as all-N/A so it's visible
            row = {'Metric': key, 'GrandMean': None, 'Lower': None, 'Upper': None}
            for s in stems:
                row[s] = (None, None)
            rows.append(row)
            continue
        gm = np.mean(list(vals.values()))
        tol = abs(gm) * tol_pct / 100.0
        lo, hi = gm - tol, gm + tol
        row = {
            'Metric': key,
            'GrandMean': round(gm, 3),
            'Lower': round(lo, 3),
            'Upper': round(hi, 3),
        }
        for s in stems:
            v = vals.get(s)
            row[s] = (
                round(v, 3) if v is not None else None,
                lo <= v <= hi if v is not None else None,
            )
        rows.append(row)
    return rows


def _draw_hd_table(ax, hd_results, mesh_paths):
    ax.set_title(
        'Hausdorff Weighted Mean Distance — ±2 mm Equivalence Test',
        fontsize=8, fontweight='bold', pad=3,
    )
    ax.axis('off')
    if not hd_results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    col_labels = ['Mesh', 'Weighted Mean (mm)', 'Result']
    table_data, cell_colors = [], []
    for p in mesh_paths:
        p = Path(p)
        r = hd_results.get(p.name)
        if r is None:
            table_data.append([p.stem, 'N/A', 'N/A'])
            cell_colors.append(['#DDDDDD', '#EEEEEE', '#EEEEEE'])
        else:
            wm = r['weighted_mean_mm']
            passed = r['pass']
            bg = '#90EE90' if passed else '#FF6B6B'
            table_data.append([p.stem, f'{wm:.3f}', 'PASS' if passed else 'FAIL'])
            cell_colors.append(['#DDDDDD', bg, bg])

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width([0, 1, 2])
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(cell_colors[ri - 1][ci])
        cell.set_edgecolor('#AAAAAA')
        cell.set_linewidth(0.5)


def _draw_equiv_table(ax, rows, stems, title, value_fmt='{:.3f}'):
    ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
    ax.axis('off')
    if not rows:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    col_labels = ['Metric', 'Grand Mean', 'Lower (−5%)', 'Upper (+5%)'] + stems
    table_data, cell_colors = [], []
    for row in rows:
        gm, lo, hi = row['GrandMean'], row['Lower'], row['Upper']
        r_data = [
            row['Metric'],
            f'{gm:.3f}' if gm is not None else 'N/A',
            f'{lo:.3f}' if lo is not None else 'N/A',
            f'{hi:.3f}' if hi is not None else 'N/A',
        ]
        r_col = ['#DDDDDD'] * 4
        for s in stems:
            cell = row.get(s)
            if cell is None or cell[0] is None:
                r_data.append('N/A')
                r_col.append('#EEEEEE')
            else:
                val, passed = cell
                r_data.append(value_fmt.format(val))
                r_col.append('#90EE90' if passed else '#FF6B6B')
        table_data.append(r_data)
        cell_colors.append(r_col)

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='bold', fontsize=6.5)
        else:
            cell.set_facecolor(cell_colors[ri - 1][ci])
        cell.set_edgecolor('#AAAAAA')
        cell.set_linewidth(0.5)


def run_ern_test(mesh_paths, output_folder, metric_paths=None):
    """
    mesh_paths   — registered (unclipped) meshes used for Hausdorff.
    metric_paths — clipped/resampled meshes used for craniometrics and Elawadly.
                   Defaults to mesh_paths when not supplied.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    mesh_paths = [Path(p) for p in mesh_paths]
    metric_paths = [Path(p) for p in metric_paths] if metric_paths else mesh_paths
    n = len(mesh_paths)
    stems = [p.stem for p in mesh_paths]  # canonical display names

    # Hausdorff all-vs-all (on unclipped registered meshes)
    print('ERN: Hausdorff all-vs-all...')
    all_metrics, mae_per_target = run_pairwise_hausdorff(mesh_paths, output_folder)

    # Craniometrics on clipped/resampled meshes
    print('ERN: Craniometrics...')
    _cranio_raw = _run_craniometrics(metric_paths)
    # Remap from metric stem → display stem
    cranio_data = {
        hd.stem: _cranio_raw.get(mt.stem, {})
        for hd, mt in zip(mesh_paths, metric_paths)
    }

    # Elawadly on clipped/resampled meshes
    print('ERN: Elawadly...')
    _elaw_raw = _run_elawadly(metric_paths)
    # Remap from metric stem → display stem
    elaw_data = {
        hd.stem: _elaw_raw.get(mt.stem, {})
        for hd, mt in zip(mesh_paths, metric_paths)
    }

    # Hausdorff weighted mean equivalence
    hd_results = _hausdorff_weighted_means(all_metrics, mesh_paths)

    # Craniometrics equivalence (±5% grand mean)
    cranio_rows = _equivalence_rows(cranio_data, CRANIO_KEYS, stems)

    # Elawadly equivalence (±5% grand mean)
    flat_elaw = {s: _flatten_elawadly(m) for s, m in elaw_data.items()}
    elaw_rows = _equivalence_rows(flat_elaw, ELAW_SUMMARY_KEYS, stems)

    # Front-view images from MAE distances (on unclipped meshes for visual)
    print('ERN: Rendering front views...')
    front_imgs = {}
    for p in mesh_paths:
        if p in mae_per_target:
            try:
                front_imgs[p.stem] = _get_front_view_img(p, mae_per_target[p])
            except Exception as e:
                print(f'  Front view failed for {p.stem}: {e}')

    # CSV outputs
    hd_out = [{'Target': t, **r} for t, r in hd_results.items()]
    pd.DataFrame(hd_out).to_csv(str(output_folder / 'ern_hausdorff_weighted.csv'), index=False)

    cranio_out = [{'Mesh': s, **d} for s, d in cranio_data.items()]
    pd.DataFrame(cranio_out).to_csv(str(output_folder / 'ern_craniometrics.csv'), index=False)

    elaw_out = [{'Mesh': s, **d} for s, d in flat_elaw.items()]
    pd.DataFrame(elaw_out).to_csv(str(output_folder / 'ern_elawadly.csv'), index=False)

    # Summary PNG
    print('ERN: Building summary PNG...')
    n_cr = len(cranio_rows)
    n_el = len(elaw_rows)

    fig_w = max(16, n * 2.4 + 4)
    hm_h = 2.2
    hd_h = max(1.2, n * 0.28 + 0.6)
    cr_h = max(1.0, n_cr * 0.22 + 0.6)
    el_h = max(1.0, n_el * 0.22 + 0.6)
    fig_h = hm_h + hd_h + cr_h + el_h + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    folder_name = mesh_paths[0].parent.name if mesh_paths else ''
    fig.suptitle(f'ERN Data Pooling Test — {folder_name}', fontsize=12, fontweight='bold')

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[hm_h, hd_h, cr_h, el_h],
        hspace=0.3, top=0.94, bottom=0.02, left=0.02, right=0.98,
    )

    # Row 0: front-view heatmaps
    if n > 0:
        gs_hm = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=gs[0], wspace=0.08)
        for i, p in enumerate(mesh_paths):
            ax = fig.add_subplot(gs_hm[i])
            img = front_imgs.get(p.stem)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=7)
            ax.axis('off')
            ax.set_title(p.stem, fontsize=6, pad=1)

    # Row 1: Hausdorff equivalence
    ax_hd = fig.add_subplot(gs[1])
    _draw_hd_table(ax_hd, hd_results, mesh_paths)

    # Row 2: Craniometrics equivalence
    ax_cr = fig.add_subplot(gs[2])
    _draw_equiv_table(
        ax_cr, cranio_rows, stems,
        'Craniometrics Equivalence (±5% grand mean)',
        value_fmt='{:.2f}',
    )

    # Row 3: Elawadly equivalence
    ax_el = fig.add_subplot(gs[3])
    _draw_equiv_table(
        ax_el, elaw_rows, stems,
        'Elawadly Cephalometrics Equivalence (±5% grand mean)',
        value_fmt='{:.4f}',
    )

    out_path = output_folder / 'ern_test_summary.png'
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'ERN summary saved: {out_path}')

    return all_metrics
