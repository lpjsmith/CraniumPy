"""
CraniumPy regression tests.

Run from the project root:
    python -m unittest discover -s tests -v

Known-good values come from reference JSON files in resources/test_mesh/.
To regenerate after a deliberate change: run manually, verify outputs, update
constants here.
"""

import sys
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import pyvista as pv

# ---------------------------------------------------------------------------
# Paths to reference data distributed with the repo
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent.parent

# Clipped registered test mesh — used for CranioMetrics and Elawadly
TEST_MESH_C = REPO_ROOT / 'resources' / 'test_mesh' / 'test_mesh_rg_C.ply'
# Registration landmarks for the test mesh (nasion/tragi in registered space)
TEST_LMK    = REPO_ROOT / 'resources' / 'test_mesh' / 'test_mesh_rg_landmarks.json'

# Hausdorff: two structurally different meshes (human vs. mannequin)
HD_REF      = REPO_ROOT / 'resources' / 'test_mesh' / 'test_mesh_rg.ply'
HD_TARGET   = REPO_ROOT / 'resources' / 'test_mesh' / 'mannequin' / 'Styro_Head_Reduced_rg.ply'

# ERN pipeline: 7 face-registered meshes
ERN_FOLDER  = REPO_ROOT / 'resources' / 'test_mesh' / 'ERN_7_MHT_meshes'

# ---------------------------------------------------------------------------
# Known-good values — derived from reference JSON files in resources/test_mesh
# ---------------------------------------------------------------------------

# CranioMetrics on TEST_MESH_C  (source: test_mesh_rg_metrics.json)
KNOWN_OFD_MM         = 184.27
KNOWN_BPD_MM         = 138.28
KNOWN_CI             = 75.0
KNOWN_HC_CM          = 50.6
KNOWN_VOLUME_CC      = 2034.88

# Elawadly on TEST_MESH_C with TEST_LMK  (source: test_mesh_rg_C_elawadly.json)
KNOWN_MAP_CIRC_CM    = 50.7
KNOWN_FA30_DEG       = 54.6
KNOWN_APWR1          = 1.2078
KNOWN_APWR2          = 1.0501
KNOWN_rAPDR30        = 1.2335
KNOWN_lAPDR30        = 1.1814
KNOWN_rAPDR60        = 1.0453
KNOWN_lAPDR60        = 1.0549
KNOWN_APAR           = 1.175
KNOWN_APVR           = 1.0213
KNOWN_RADIALS        = {
    'P12': 102.86, 'P1': 93.02, 'P2': 72.89, 'P3': 67.76,
    'P4':   68.83, 'P5': 76.17, 'P6': 80.83, 'P7': 75.41,
    'P8':   69.73, 'P9': 67.93, 'P10': 72.61, 'P11': 89.99,
}

# Hausdorff: HD_REF (test_mesh_rg) as reference, HD_TARGET (mannequin) as target
# (source: hausdorff_Styro_Head_Reduced_rg_vs_test_mesh_rg.csv, row ref=test_mesh_rg)
KNOWN_HD_MEAN_MM     = 10.519
KNOWN_HD_MAX_MM      = 77.845
KNOWN_HD_MIN_MM      = 0.063
KNOWN_HD_RMS_MM      = 17.187


# ===========================================================================
# 1. Imports / installation
# ===========================================================================
class TestImports(unittest.TestCase):
    """Verify every required package is installed and importable."""

    def test_numpy(self):
        import numpy

    def test_pyvista(self):
        import pyvista

    def test_pymeshlab(self):
        import pymeshlab

    def test_pyacvd(self):
        import pyacvd

    def test_pymeshfix(self):
        import pymeshfix

    def test_trimesh(self):
        import trimesh

    def test_pandas(self):
        import pandas

    def test_matplotlib(self):
        import matplotlib

    def test_craniometrics_module(self):
        from craniometrics.craniometrics import CranioMetrics

    def test_elawadly_module(self):
        from craniometrics.elawadly import compute_elawadly

    def test_hausdorff_module(self):
        from craniometrics.hausdorff import compute_per_vertex_distances, run_pairwise_hausdorff

    def test_ern_test_module(self):
        from craniometrics.ern_test import run_ern_test


# ===========================================================================
# 2. CranioMetrics
# ===========================================================================
class TestCranioMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from craniometrics.craniometrics import CranioMetrics
        cls.m = CranioMetrics(TEST_MESH_C)
        cls.m.extract_dimensions(cls.m.slice_height)

    # -- structural: outputs exist and are the right type --------------------

    def test_ofd_is_float(self):
        self.assertIsInstance(float(self.m.depth), float)

    def test_bpd_is_float(self):
        self.assertIsInstance(float(self.m.breadth), float)

    def test_ci_is_float(self):
        self.assertIsInstance(float(self.m.CI), float)

    def test_hc_is_float(self):
        self.assertIsInstance(float(self.m.HC), float)

    def test_volume_positive(self):
        self.assertGreater(self.m.pvmesh.volume, 0)

    def test_ofd_positive(self):
        self.assertGreater(float(self.m.depth), 0)

    def test_bpd_positive(self):
        self.assertGreater(float(self.m.breadth), 0)

    def test_ci_positive(self):
        self.assertGreater(float(self.m.CI), 0)

    def test_hc_positive(self):
        self.assertGreater(float(self.m.HC), 0)

    # -- known-good regression values ----------------------------------------

    def test_ofd_known_value(self):
        self.assertAlmostEqual(float(self.m.depth), KNOWN_OFD_MM, places=1)

    def test_bpd_known_value(self):
        self.assertAlmostEqual(float(self.m.breadth), KNOWN_BPD_MM, places=1)

    def test_ci_known_value(self):
        self.assertAlmostEqual(float(self.m.CI), KNOWN_CI, places=1)

    def test_hc_known_value(self):
        self.assertAlmostEqual(float(self.m.HC), KNOWN_HC_CM, places=1)

    def test_volume_known_value(self):
        vol_cc = round(self.m.pvmesh.volume / 1000, 2)
        self.assertAlmostEqual(vol_cc, KNOWN_VOLUME_CC, places=1)


# ===========================================================================
# 3. Elawadly
# ===========================================================================
class TestElawadly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from craniometrics.elawadly import compute_elawadly
        with open(TEST_LMK) as f:
            lmk = json.load(f)
        # Landmarks are stored in registered space with these key names
        nasion       = np.array(lmk['new_nasion'])
        tragus_right = np.array(lmk['new_rh_coord'])
        tragus_left  = np.array(lmk['new_lh_coord'])
        mesh_pv = pv.read(str(TEST_MESH_C))
        result = compute_elawadly(mesh_pv, nasion, tragus_right, tragus_left, TEST_MESH_C)
        # result = (midpoint, map_vertex, normal_tnt, first_dir,
        #           radial_distances, radial_pts, circumference, circ_slice, measurements)
        cls.measurements = result[8]
        cls.radials      = cls.measurements['radial_distances_mm']
        cls.ratios       = cls.measurements['ratios']
        cls.area_vol     = cls.measurements['area_volume']
        cls.widths       = cls.measurements['widths_mm']

    # -- structural: all 12 radials hit the mesh -----------------------------

    def test_all_radials_non_none(self):
        from craniometrics.elawadly import POINT_LABELS
        for label in POINT_LABELS:
            with self.subTest(label=label):
                self.assertIsNotNone(
                    self.radials[label],
                    msg=f'Radial distance for {label} is None — ray did not hit mesh',
                )

    def test_map_circumference_not_none(self):
        self.assertIsNotNone(self.measurements['MAP_circumference_cm'])

    def test_fa30_not_none(self):
        self.assertIsNotNone(self.measurements['FA30_degrees'])

    def test_radials_keys_complete(self):
        from craniometrics.elawadly import POINT_LABELS
        for label in POINT_LABELS:
            with self.subTest(label=label):
                self.assertIn(label, self.radials)

    def test_ratios_keys_present(self):
        for key in ('APWR1', 'APWR2', 'rAPDR30', 'lAPDR30', 'rAPDR60', 'lAPDR60'):
            with self.subTest(key=key):
                self.assertIn(key, self.ratios)

    def test_area_volume_keys_present(self):
        for key in ('APAR', 'APVR'):
            with self.subTest(key=key):
                self.assertIn(key, self.area_vol)

    def test_widths_keys_present(self):
        for key in ('W1_anterior_P11_P1', 'W1_posterior_P7_P5',
                    'W2_anterior_P10_P2', 'W2_posterior_P8_P4'):
            with self.subTest(key=key):
                self.assertIn(key, self.widths)

    def test_elawadly_json_written(self):
        jsonpath = TEST_MESH_C.parent / (TEST_MESH_C.stem + '_elawadly.json')
        self.assertTrue(jsonpath.exists())

    # -- known-good regression values ----------------------------------------

    def test_map_circumference_known_value(self):
        self.assertAlmostEqual(self.measurements['MAP_circumference_cm'], KNOWN_MAP_CIRC_CM, places=1)

    def test_fa30_known_value(self):
        self.assertAlmostEqual(self.measurements['FA30_degrees'], KNOWN_FA30_DEG, places=1)

    def test_apwr1_known_value(self):
        self.assertAlmostEqual(self.ratios['APWR1'], KNOWN_APWR1, places=3)

    def test_apwr2_known_value(self):
        self.assertAlmostEqual(self.ratios['APWR2'], KNOWN_APWR2, places=3)

    def test_rapdr30_known_value(self):
        self.assertAlmostEqual(self.ratios['rAPDR30'], KNOWN_rAPDR30, places=3)

    def test_lapdr30_known_value(self):
        self.assertAlmostEqual(self.ratios['lAPDR30'], KNOWN_lAPDR30, places=3)

    def test_rapdr60_known_value(self):
        self.assertAlmostEqual(self.ratios['rAPDR60'], KNOWN_rAPDR60, places=3)

    def test_lapdr60_known_value(self):
        self.assertAlmostEqual(self.ratios['lAPDR60'], KNOWN_lAPDR60, places=3)

    def test_apar_known_value(self):
        self.assertAlmostEqual(self.area_vol['APAR'], KNOWN_APAR, places=3)

    def test_apvr_known_value(self):
        self.assertAlmostEqual(self.area_vol['APVR'], KNOWN_APVR, places=3)

    def test_all_radials_known_values(self):
        for label, expected in KNOWN_RADIALS.items():
            with self.subTest(label=label):
                self.assertAlmostEqual(self.radials[label], expected, places=1)


# ===========================================================================
# 4. Hausdorff
# ===========================================================================
class TestHausdorff(unittest.TestCase):
    """
    Reference: test_mesh_rg.ply (human head, registered)
    Target:    Styro_Head_Reduced_rg.ply (mannequin, structurally different)
    Known-good: hausdorff_Styro_Head_Reduced_rg_vs_test_mesh_rg.csv
    """

    @classmethod
    def setUpClass(cls):
        from craniometrics.hausdorff import compute_per_vertex_distances
        cls.distances, cls.stats = compute_per_vertex_distances(HD_REF, HD_TARGET)

    # -- structural ----------------------------------------------------------

    def test_distances_array_length(self):
        mesh_target = pv.read(str(HD_TARGET))
        self.assertEqual(len(self.distances), mesh_target.n_points)

    def test_distances_non_negative(self):
        self.assertTrue(np.all(self.distances >= 0))

    def test_distances_dtype_float(self):
        self.assertEqual(self.distances.dtype.kind, 'f')

    def test_stats_keys_present(self):
        for key in ('min_mm', 'max_mm', 'mean_mm', 'rms_mm'):
            with self.subTest(key=key):
                self.assertIn(key, self.stats)

    def test_stats_non_negative(self):
        for key, val in self.stats.items():
            with self.subTest(key=key):
                self.assertGreaterEqual(val, 0)

    def test_self_distance_near_zero(self):
        from craniometrics.hausdorff import compute_per_vertex_distances
        dists, stats = compute_per_vertex_distances(HD_REF, HD_REF)
        self.assertAlmostEqual(stats['mean_mm'], 0.0, places=2)

    def test_different_meshes_have_nonzero_distance(self):
        self.assertGreater(self.stats['mean_mm'], 0.0)

    # -- known-good regression values ----------------------------------------

    def test_hausdorff_mean_known_value(self):
        self.assertAlmostEqual(self.stats['mean_mm'], KNOWN_HD_MEAN_MM, places=2)

    def test_hausdorff_max_known_value(self):
        self.assertAlmostEqual(self.stats['max_mm'], KNOWN_HD_MAX_MM, places=2)

    def test_hausdorff_min_known_value(self):
        self.assertAlmostEqual(self.stats['min_mm'], KNOWN_HD_MIN_MM, places=2)

    def test_hausdorff_rms_known_value(self):
        self.assertAlmostEqual(self.stats['rms_mm'], KNOWN_HD_RMS_MM, places=2)


# ===========================================================================
# 5. ERN test — full pipeline outputs exist
# ===========================================================================
class TestERNPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from craniometrics.ern_test import run_ern_test
        cls.tmp = tempfile.mkdtemp()
        ern_meshes = sorted([
            p for p in ERN_FOLDER.glob('*.ply')
            if not p.stem.endswith('_C')
        ])
        cls.mesh_paths = ern_meshes
        run_ern_test(ern_meshes, cls.tmp)

    def test_hausdorff_csv_exists(self):
        self.assertTrue((Path(self.tmp) / 'ern_hausdorff_weighted.csv').exists())

    def test_craniometrics_csv_exists(self):
        self.assertTrue((Path(self.tmp) / 'ern_craniometrics.csv').exists())

    def test_elawadly_csv_exists(self):
        self.assertTrue((Path(self.tmp) / 'ern_elawadly.csv').exists())

    def test_summary_png_exists(self):
        self.assertTrue((Path(self.tmp) / 'ern_test_summary.png').exists())

    def test_summary_png_non_empty(self):
        png = Path(self.tmp) / 'ern_test_summary.png'
        self.assertGreater(png.stat().st_size, 0)

    def test_hausdorff_csv_has_correct_row_count(self):
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / 'ern_hausdorff_weighted.csv')
        n = len(self.mesh_paths)
        self.assertEqual(len(df), n)

    def test_craniometrics_csv_has_correct_row_count(self):
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / 'ern_craniometrics.csv')
        self.assertEqual(len(df), len(self.mesh_paths))

    def test_elawadly_csv_has_correct_row_count(self):
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / 'ern_elawadly.csv')
        self.assertEqual(len(df), len(self.mesh_paths))

    def test_hausdorff_csv_has_required_columns(self):
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / 'ern_hausdorff_weighted.csv')
        for col in ('Target', 'weighted_mean_mm', 'pass'):
            with self.subTest(col=col):
                self.assertIn(col, df.columns)

    def test_craniometrics_csv_has_required_columns(self):
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / 'ern_craniometrics.csv')
        for col in ('Mesh', 'OFD_depth_mm', 'BPD_breadth_mm',
                    'Cephalic_Index', 'Circumference_cm', 'MeshVolume_cc'):
            with self.subTest(col=col):
                self.assertIn(col, df.columns)

    def test_elawadly_csv_has_required_columns(self):
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / 'ern_elawadly.csv')
        for col in ('Mesh', 'MAP_circumference_cm', 'FA30_degrees',
                    'APWR1', 'APWR2', 'APAR', 'APVR'):
            with self.subTest(col=col):
                self.assertIn(col, df.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
