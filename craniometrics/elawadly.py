import numpy as np
import pyvista as pv
import json
import datetime
from pathlib import Path

RAY_LENGTH = 250.0

# Clock-face labels: 0°=P12 (anterior), 90°=P3 (right), 180°=P6 (posterior), 270°=P9 (left)
POINT_LABELS = ['P12', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
ANGLES_DEG = np.arange(0, 360, 30)


def compute_elawadly(mesh_pv, nasion, tragus_right, tragus_left, file_path):
    # TNT plane normal from nasion and two tragi
    v1 = tragus_right - nasion
    v2 = tragus_left - nasion
    normal_tnt = np.cross(v1, v2)
    normal_tnt /= np.linalg.norm(normal_tnt)

    # MAP vertex: mesh vertex furthest in the nasion (anterior) direction
    tragal_midpoint = (tragus_right + tragus_left) / 2
    anterior_dir = nasion - tragal_midpoint
    anterior_dir /= np.linalg.norm(anterior_dir)
    map_vertex = mesh_pv.points[np.argmax(mesh_pv.points @ anterior_dir)]

    # MAP plane: parallel to TNT plane, passing through the MAP vertex
    d_map = np.dot(normal_tnt, map_vertex)

    def _project_onto_map_plane(point):
        t = (d_map - np.dot(normal_tnt, point)) / np.dot(normal_tnt, normal_tnt)
        return point + t * normal_tnt

    midpoint = (_project_onto_map_plane(tragus_right) + _project_onto_map_plane(tragus_left)) / 2

    # Radial frame: midline points RIGHT, first_dir points ANTERIOR
    # Gives: 0°=P12 (front), 90°=P3 (right), 180°=P6 (back), 270°=P9 (left)
    midline = tragus_right - tragus_left
    midline /= np.linalg.norm(midline)
    first_dir = np.cross(midline, normal_tnt)
    first_dir /= np.linalg.norm(first_dir)
    if np.dot(first_dir, anterior_dir) < 0:
        first_dir = -first_dir

    # --- Radial distances via reverse ray casting ---
    radial_distances = {}
    radial_pts = {}

    for label, angle_deg in zip(POINT_LABELS, ANGLES_DEG):
        angle = np.radians(angle_deg)
        direction = first_dir * np.cos(angle) + midline * np.sin(angle)
        direction /= np.linalg.norm(direction)
        start = midpoint + direction * RAY_LENGTH
        pts, _ = mesh_pv.ray_trace(start, midpoint)
        if len(pts) > 0:
            hit = pts[np.argmin(np.linalg.norm(pts - start, axis=1))]
            radial_distances[label] = round(float(np.linalg.norm(hit - midpoint)), 2)
            radial_pts[label] = hit
        else:
            radial_distances[label] = None
            radial_pts[label] = None

    # --- Width measurements ---
    def _dist(a, b):
        pa, pb = radial_pts.get(a), radial_pts.get(b)
        return round(float(np.linalg.norm(pa - pb)), 2) if pa is not None and pb is not None else None

    w1_ant  = _dist('P11', 'P1')
    w1_post = _dist('P7',  'P5')
    w2_ant  = _dist('P10', 'P2')
    w2_post = _dist('P8',  'P4')

    # --- Ratios ---
    def _ratio(a, b):
        return round(a / b, 4) if a is not None and b is not None and b != 0 else None

    APWR1  = _ratio(w1_ant, w1_post)
    APWR2  = _ratio(w2_ant, w2_post)
    rAPDR30 = _ratio(radial_distances.get('P1'),  radial_distances.get('P7'))
    lAPDR30 = _ratio(radial_distances.get('P11'), radial_distances.get('P5'))
    rAPDR60 = _ratio(radial_distances.get('P2'),  radial_distances.get('P8'))
    lAPDR60 = _ratio(radial_distances.get('P10'), radial_distances.get('P4'))

    # --- Frontal angulation FA30: angle at P12 between vectors P11→P12 and P12→P1 ---
    FA30 = None
    if all(radial_pts.get(p) is not None for p in ['P11', 'P12', 'P1']):
        v_in  = radial_pts['P12'] - radial_pts['P11']
        v_out = radial_pts['P1']  - radial_pts['P12']
        cos_a = np.dot(v_in, v_out) / (np.linalg.norm(v_in) * np.linalg.norm(v_out))
        FA30  = round(float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))), 2)

    # --- MAP circumference via convex hull slice ---
    circumference, circ_slice = _compute_circumference(mesh_pv, midpoint, normal_tnt)

    # --- Anterior/posterior area and volume ratios ---
    APAR, APVR = _compute_area_volume_ratios(mesh_pv, midpoint, first_dir)

    # --- Save JSON ---
    file_path = Path(file_path)
    measurements = {
        "Datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filepath": str(file_path),
        "MAP_midpoint_mm": midpoint.tolist(),
        "MAP_circumference_cm": circumference,
        "radial_distances_mm": radial_distances,
        "widths_mm": {
            "W1_anterior_P11_P1": w1_ant,
            "W1_posterior_P7_P5": w1_post,
            "W2_anterior_P10_P2": w2_ant,
            "W2_posterior_P8_P4": w2_post,
        },
        "ratios": {
            "APWR1":   APWR1,
            "APWR2":   APWR2,
            "rAPDR30": rAPDR30,
            "lAPDR30": lAPDR30,
            "rAPDR60": rAPDR60,
            "lAPDR60": lAPDR60,
        },
        "FA30_degrees": FA30,
        "area_volume": {
            "APAR": APAR,
            "APVR": APVR,
        },
    }

    jsonpath = str(file_path.parent / (file_path.stem + '_elawadly.json'))
    with open(jsonpath, 'w') as f:
        json.dump(measurements, f, indent=4)
    print(f'Elawadly results saved: {jsonpath}')

    return midpoint, map_vertex, normal_tnt, first_dir, radial_distances, radial_pts, \
           circumference, circ_slice, measurements


def _compute_circumference(mesh_pv, midpoint, normal_tnt):
    try:
        circ_slice = mesh_pv.slice(normal=normal_tnt.tolist(), origin=midpoint.tolist())
        if circ_slice.n_points < 3:
            return None, None

        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(normal_tnt, u)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(normal_tnt, u); e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal_tnt, e1)
        local = circ_slice.points - midpoint
        sorted_pts = circ_slice.points[np.argsort(np.arctan2(local @ e2, local @ e1))]
        perimeter = sum(
            np.linalg.norm(sorted_pts[i] - sorted_pts[(i + 1) % len(sorted_pts)])
            for i in range(len(sorted_pts))
        )
        return round(perimeter / 10, 1), circ_slice
    except Exception:
        return None, None


def _compute_area_volume_ratios(mesh_pv, midpoint, first_dir):
    APAR, APVR = None, None
    try:
        ant  = mesh_pv.clip(normal=first_dir.tolist(),         origin=midpoint.tolist(), invert=False)
        post = mesh_pv.clip(normal=first_dir.tolist(),         origin=midpoint.tolist(), invert=True)
        if post.area > 0:
            APAR = round(ant.area / post.area, 4)
        ant_vol  = abs(ant.volume)
        post_vol = abs(post.volume)
        if post_vol > 0:
            APVR = round(ant_vol / post_vol, 4)
    except Exception:
        pass
    return APAR, APVR
