# -*- coding: utf-8 -*-
"""
Plugin QGIS: Análisis Geológico Estructural – Best-Fit Plane (PCA)
Versión: absorbe tramos cortos hacia el SIGUIENTE, mantiene historial de fusiones,
parámetros editables en UI (step_m y min_segment_length_m), línea real 3D y visor.
"""

import math
import traceback
import numpy as np

from qgis.PyQt.QtCore import QCoreApplication, QVariant, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAction, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QGroupBox, QGridLayout, QFileDialog, QComboBox, QCheckBox,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QSpinBox
)
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer, QgsFeature,
    QgsGeometry, QgsPointXY, QgsField, QgsFields, QgsVectorFileWriter,
    QgsWkbTypes, QgsMessageLog, Qgis, QgsDistanceArea, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform
)

# dependencia opcional: ruptures
try:
    import ruptures as rpt  # type: ignore
    HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False

# Matplotlib embebido en Qt para visor 3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ==========================
# DIALOG PRINCIPAL (UI editable)
# ==========================
class GeologicalAnalysisDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_all = []
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle(self.tr("Análisis Geológico – Best-Fit Plane (PCA)"))
        self.setMinimumSize(760, 620)
        layout = QVBoxLayout(self)

        # Selección de capas
        layers = QGroupBox(self.tr("Selección de Capas"))
        lgrid = QGridLayout()
        lgrid.addWidget(QLabel(self.tr("Capa de contactos (líneas):")), 0, 0)
        self.contact_layer = QComboBox()
        lgrid.addWidget(self.contact_layer, 0, 1)
        lgrid.addWidget(QLabel(self.tr("Capa DEM (ráster):")), 1, 0)
        self.dem_layer = QComboBox()
        lgrid.addWidget(self.dem_layer, 1, 1)
        layers.setLayout(lgrid)
        layout.addWidget(layers)

        # Archivo de salida
        output = QGroupBox(self.tr("Archivo de Salida"))
        oh = QHBoxLayout()
        self.output_path = QLabel(self.tr("Seleccionar…"))
        oh.addWidget(self.output_path)
        btn = QPushButton(self.tr("Examinar"))
        btn.clicked.connect(self.browse)
        oh.addWidget(btn)
        output.setLayout(oh)
        layout.addWidget(output)

        # Opciones fácilmente editables por el usuario
        opts = QGroupBox(self.tr("Parámetros (editable)"))
        og = QGridLayout()
        og.addWidget(QLabel("Paso muestreo (m) [0 = auto]:"), 0, 0)
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.0, 1000.0)
        self.spin_step.setDecimals(2)
        self.spin_step.setSingleStep(0.5)
        self.spin_step.setValue(0.0)  # 0 -> auto
        self.spin_step.setToolTip("Dejar 0 usa heurística automática basada en la resolución del DEM.")
        og.addWidget(self.spin_step, 0, 1)

        og.addWidget(QLabel("Longitud mínima segment (m):"), 1, 0)
        self.spin_minseg = QDoubleSpinBox()
        self.spin_minseg.setRange(1.0, 10000.0)
        self.spin_minseg.setDecimals(1)
        self.spin_minseg.setValue(150.0)
        self.spin_minseg.setSingleStep(10.0)
        og.addWidget(self.spin_minseg, 1, 1)

        og.addWidget(QLabel("Escala objetivo (m) [afecta microventana]:"), 2, 0)
        self.spin_target_scale = QDoubleSpinBox()
        self.spin_target_scale.setRange(5.0, 1000.0)
        self.spin_target_scale.setDecimals(1)
        self.spin_target_scale.setValue(50.0)
        og.addWidget(self.spin_target_scale, 2, 1)

        self.chk_debug = QCheckBox("Mostrar logs detallados (debug)")
        og.addWidget(self.chk_debug, 3, 0, 1, 2)

        opts.setLayout(og)
        layout.addWidget(opts)

        # Progreso y log
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(220)
        layout.addWidget(self.log)

        # Botonera
        hb = QHBoxLayout()
        run_btn = QPushButton(self.tr("Ejecutar Análisis"))
        run_btn.clicked.connect(self.accept)
        hb.addWidget(run_btn)

        self.view_results_btn = QPushButton(self.tr("Ver Segmentos / 3D…"))
        self.view_results_btn.setEnabled(False)
        hb.addWidget(self.view_results_btn)

        cancel_btn = QPushButton(self.tr("Cerrar"))
        cancel_btn.clicked.connect(self.reject)
        hb.addWidget(cancel_btn)
        layout.addLayout(hb)

        self.populate_layers()

    def populate_layers(self):
        self.contact_layer.clear()
        self.dem_layer.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.LineGeometry:
                self.contact_layer.addItem(layer.name(), layer.id())
            elif isinstance(layer, QgsRasterLayer):
                self.dem_layer.addItem(layer.name(), layer.id())

    def browse(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, self.tr("Guardar resultado"), "", "ESRI Shapefile (*.shp)"
        )
        if fname and not fname.lower().endswith(".shp"):
            fname += ".shp"
        if fname:
            self.output_path.setText(fname)

    def get_selected_layer(self, combo):
        return QgsProject.instance().mapLayer(combo.currentData())

    def log_message(self, message, debug=False):
        if debug and not self.chk_debug.isChecked():
            return
        self.log.append(message)
        QCoreApplication.processEvents()


# ==========================
# LÓGICA DEL ANALIZADOR
# ==========================
class StructuralAnalyzer:
    """
    Analizador con:
      - muestreo por interpolación controlada,
      - segmentación por changepoints,
      - fusión de tramos cortos ABSORBIÉNDOLOS hacia el SIGUIENTE,
      - preservación de historial (merged_from).
    """
    def __init__(self, target_scale_m: float = 50.0, min_segment_length_m: float = 150.0):
        self.target_scale_m = float(target_scale_m)
        self.min_segment_length_m = float(min_segment_length_m)
        self.dist_calc = QgsDistanceArea()
        self.dist_calc.setEllipsoid(QgsProject.instance().ellipsoid())

    # --- CRS métrico auxiliar
    def _get_metric_crs(self, crs, geom):
        if not crs.isGeographic():
            return crs
        p = geom.centroid().asPoint()
        zone = int((p.x() + 180) / 6) + 1
        if zone < 1 or zone > 60:
            zone = 31
        epsg = 32600 + zone if p.y() >= 0 else 32700 + zone
        return QgsCoordinateReferenceSystem(epsg)

    # --- Heurística de paso (si user_step == 0 -> auto)
    def auto_sampling_step(self, dem_layer, target_scale_m=50.0):
        try:
            gsd_x = abs(dem_layer.rasterUnitsPerPixelX())
            gsd_y = abs(dem_layer.rasterUnitsPerPixelY())
            gsd = max(gsd_x, gsd_y) if (gsd_x > 0 and gsd_y > 0) else 1.0
        except Exception:
            gsd = 1.0

        max_step = max(1.0, float(target_scale_m) / 5.0)
        pref = max(1.0, gsd)
        candidate = min(pref, max_step)
        step = max(gsd, candidate)
        step = float(max(0.5, step))
        return step

    def auto_micro_window(self, step, target_scale_m=50.0):
        return max(8.0, min(30.0, 3.0 * step))

    # --- Muestreo por interpolación (controlado) ---
    def _sample_points(self, geom_metric: QgsGeometry, dem_layer: QgsRasterLayer,
                       to_dem: QgsCoordinateTransform, step_m: float,
                       log_func, start_d: float = 0.0, end_d: float = None):
        if end_d is None:
            try:
                length = geom_metric.length()
            except Exception:
                length = 0.0
            end_d = length

        try:
            length = geom_metric.length()
        except Exception:
            length = 0.0

        if length <= 0 or end_d <= start_d:
            return np.empty((0, 3))

        est_n = int(math.floor((end_d - start_d) / float(step_m))) + 1
        max_points = 15000
        if est_n > max_points:
            old_step = step_m
            step_m = float((end_d - start_d) / float(max_points))
            log_func(f"  ⚠️ Ajustado step_m {old_step:.3f}→{step_m:.3f} m para limitar {max_points} pts", debug=True)
            est_n = max_points

        prov = dem_layer.dataProvider()
        pts = []
        d = float(start_d)
        cnt = 0
        while d <= end_d + 1e-8 and cnt < max_points:
            try:
                pgeom = geom_metric.interpolate(d)
                p = pgeom.asPoint()
            except Exception:
                d += step_m
                cnt += 1
                continue
            ptm = QgsPointXY(p.x(), p.y())
            try:
                dem_pt = to_dem.transform(ptm)
            except Exception:
                d += step_m
                cnt += 1
                continue
            z, ok = prov.sample(dem_pt, 1)
            if ok and z is not None and not math.isnan(z):
                pts.append([p.x(), p.y(), float(z)])
            d += step_m
            cnt += 1

        # forzar incluir el extremo end_d
        try:
            pgeom = geom_metric.interpolate(end_d)
            p = pgeom.asPoint()
            ptm = QgsPointXY(p.x(), p.y())
            dem_pt = to_dem.transform(ptm)
            z, ok = prov.sample(dem_pt, 1)
            if ok and z is not None and not math.isnan(z):
                if len(pts) == 0 or (abs(pts[-1][0] - p.x()) > 1e-9 or abs(pts[-1][1] - p.y()) > 1e-9):
                    pts.append([p.x(), p.y(), float(z)])
        except Exception:
            pass

        log_func(f"  -> {len(pts)} pts muestreados (start={start_d:.2f} end={end_d:.2f} step≈{step_m:.2f})", debug=True)
        return np.array(pts)

    # --- PCA / orientación ---
    @staticmethod
    def pca_normal(pts3d: np.ndarray):
        mean_pt = np.mean(pts3d, axis=0)
        data_c = pts3d - mean_pt
        covm = np.cov(data_c, rowvar=False)
        vals, vecs = np.linalg.eigh(covm)
        nrm = vecs[:, np.argmin(vals)]
        if nrm[2] < 0:
            nrm = -nrm
        lam_min = float(np.min(vals))
        lam_sum = float(np.sum(vals)) if float(np.sum(vals)) > 0 else 1.0
        planarity = 1.0 - (lam_min / lam_sum)
        return nrm, planarity

    @staticmethod
    def normal_to_angles(nrm):
        nx, ny, nz = nrm
        if nz < 0:
            nx, ny, nz = -nx, -ny, -nz
        dip = math.degrees(math.acos(max(-1.0, min(1.0, nz))))
        dx, dy = -nx, -ny
        dip_dir = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        strike = (dip_dir - 90.0) % 360.0
        return strike, dip, dip_dir

    # --- Normales locales (microventanas) ---
    def local_normals_along(self, pts3d, w_micro_m, step_m):
        k = max(3, int(round(w_micro_m / step_m)))
        s = max(1, k // 2)
        normals, spans = [], []
        for i in range(0, len(pts3d) - k + 1, s):
            sub = pts3d[i:i + k]
            if sub.shape[0] >= 3:
                n, _ = self.pca_normal(sub)
                normals.append(n)
                spans.append((i, i + k))
        return np.array(normals), spans

    # --- Suavizado circular ---
    @staticmethod
    def smooth_circular_deg(angles_deg, win=5):
        if len(angles_deg) == 0:
            return np.array([])
        a = np.radians(angles_deg)
        out = []
        r = max(1, win // 2)
        for i in range(len(a)):
            L = max(0, i - r)
            R = min(len(a), i + r + 1)
            s = a[L:R]
            C = math.degrees(math.atan2(np.mean(np.sin(s)), np.mean(np.cos(s))))
            out.append((C + 360.0) % 360.0)
        return np.array(out)

    # --- Segmentación (sin fusión) ---
    def segment_by_changepoints(self, pts3d, step_m, target_scale_m, log_func):
        w_micro = self.auto_micro_window(step_m, target_scale_m)
        normals, spans = self.local_normals_along(pts3d, w_micro, step_m)
        if len(normals) == 0:
            return []
        azis = []
        for n in normals:
            nx, ny, nz = n
            dx, dy = -nx, -ny
            az = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            azis.append(az)
        azis = np.array(azis)
        azis_s = self.smooth_circular_deg(azis, win=5)
        w_minseg = max(self.min_segment_length_m, 0.6 * target_scale_m)

        if HAS_RUPTURES and len(azis_s) >= 6:
            try:
                model = rpt.Pelt(model="rbf").fit(azis_s.reshape(-1, 1))
                pen = 3.0 * math.log(max(2, len(azis_s)))
                idxs = model.predict(pen=pen)
                cps = [0] + [i - 1 for i in idxs if 0 < i <= len(azis_s)]
                cps = sorted(list(set([max(0, min(len(azis_s) - 1, c)) for c in cps])))
                min_pts = max(3, int(round(w_minseg / step_m)))
                pruned = [cps[0]]
                for j in range(1, len(cps)):
                    if (cps[j] - pruned[-1]) >= min_pts:
                        pruned.append(cps[j])
                if pruned[-1] != len(azis_s) - 1:
                    pruned[-1] = len(azis_s) - 1
                cuts = pruned
            except Exception:
                cuts = self.simple_changepoints(azis_s, step_m, w_minseg, 12.0)
        else:
            cuts = self.simple_changepoints(azis_s, step_m, w_minseg, 12.0)

        segments = []
        for ii in range(len(cuts) - 1):
            a = cuts[ii]
            b = cuts[ii + 1]
            start_i = spans[a][0]
            end_i = spans[b][1]
            start_i = max(0, min(start_i, len(pts3d) - 1))
            end_i = max(start_i + 1, min(end_i, len(pts3d)))
            segments.append((start_i, end_i))

        if len(segments) == 0 and len(spans) > 0:
            segments = [(spans[0][0], spans[-1][1])]

        log_func(f"  → Segmentos detectados (pre-fusión): {len(segments)}", debug=True)
        return segments

    def simple_changepoints(self, azi_deg_smooth, step_m, w_minseg_m=30.0, thresh_deg=12.0):
        if len(azi_deg_smooth) < 3:
            return [0, max(0, len(azi_deg_smooth) - 1)]
        a = np.radians(azi_deg_smooth)
        d = np.degrees(np.arctan2(np.sin(np.diff(a)), np.cos(np.diff(a))))
        win = max(3, int(round((w_minseg_m / 4.0) / step_m)))
        if win > len(d):
            return [0, len(azi_deg_smooth) - 1]
        roll = np.convolve(np.abs(d), np.ones(win) / win, mode='same')
        candidates = np.where(roll > thresh_deg)[0]
        min_pts = max(3, int(round(w_minseg_m / step_m)))
        cps, last = [], -10**9
        for c in candidates:
            if c - last >= min_pts:
                cps.append(c)
                last = c
        N = len(azi_deg_smooth)
        bps = [0] + cps + [N - 1]
        pruned = [bps[0]]
        for j in range(1, len(bps)):
            if (bps[j] - pruned[-1]) >= min_pts:
                pruned.append(bps[j])
            else:
                continue
        if pruned[-1] != N - 1:
            pruned[-1] = N - 1
        return pruned

    # --- Fusión: ABSORBER tramos cortos hacia el SIGUIENTE ---
    def _merge_segments_absorb_next(self, segments, step_m, log_func):
        """
        segments: list[(start_i,end_i)] (end_i exclusive)
        step_m: sampling step (m)
        Regla: todos los segmentos con longitud < min_segment_length_m
              se agregan (absorben) al siguiente segmento. Si hay short(s)
              al principio, se guardan en pending hasta encontrarse con el siguiente grande.
              Si hay short(s) al final sin siguiente -> se absorben en el anterior (fallback).
        Devuelve lista de dicts: {'start_i','end_i','orig_idxs':[...]}
        """
        if not segments:
            return []

        n = len(segments)
        pending = []
        merged = []

        for idx in range(n):
            a, b = segments[idx]
            length_m = (b - a) * step_m
            if length_m < self.min_segment_length_m:
                # acumular para absorber por el siguiente
                pending.append(idx)
                continue
            else:
                # segmento suficientemente largo => crear merged que incluye pending + this
                if pending:
                    first_pending = pending[0]
                    start_i = segments[first_pending][0]
                else:
                    start_i = a
                end_i = segments[idx][1]
                orig_idxs = pending + [idx] if pending else [idx]
                merged.append({'start_i': start_i, 'end_i': end_i, 'orig_idxs': orig_idxs})
                pending = []

        # si quedaron pending al final (sin siguiente grande):
        if pending:
            if merged:
                # absorbemos pending en el último merged (fallback)
                last = merged[-1]
                last_end = segments[pending[-1]][1]
                last['end_i'] = last_end
                last['orig_idxs'].extend(pending)
                log_msg = f"    · Pending final absorbido por segmento anterior (fallback): {pending}"
                log_func(log_msg, debug=True)
            else:
                # no hay merged previos: todos son cortos -> juntarlos en un solo merged
                start_i = segments[pending[0]][0]
                end_i = segments[pending[-1]][1]
                merged.append({'start_i': start_i, 'end_i': end_i, 'orig_idxs': list(pending)})
                log_func("    · Todos los segmentos son menores; se agrupan en uno solo.", debug=True)

        # asegurar rangos válidos
        processed = []
        for m in merged:
            a = int(max(0, m['start_i']))
            b = int(max(a + 1, min(10**9, m['end_i'])))
            processed.append({'start_i': a, 'end_i': b, 'orig_idxs': list(m['orig_idxs'])})

        return processed

    # --- Extraer geom real del tramo y muestrear Z sobre ella ---
    def _extract_segment_line_and_geom(self, geom_metric: QgsGeometry, dem_layer: QgsRasterLayer,
                                       to_dem: QgsCoordinateTransform, start_m: float, end_m: float,
                                       step_m: float, log_func):
        seg_pts3d = np.empty((0, 3))
        seg_geom_metric = None

        # intentar curveSubstring
        try:
            if hasattr(geom_metric, "curveSubstring"):
                try:
                    subgeom = geom_metric.curveSubstring(start_m, end_m)
                    if subgeom and not subgeom.isEmpty():
                        seg_pts3d = self._sample_points(subgeom, dem_layer, to_dem, step_m, log_func, 0.0, subgeom.length())
                        seg_geom_metric = subgeom
                        log_func("    · Usando curveSubstring para tramo real.", debug=True)
                        return seg_pts3d, seg_geom_metric
                except Exception:
                    pass
        except Exception:
            pass

        # fallback: interpolar sobre geom_metric entre start_m y end_m
        log_func("    · No curveSubstring; interpolando tramo real.", debug=True)
        seg_pts3d = self._sample_points(geom_metric, dem_layer, to_dem, step_m, log_func, start_d=start_m, end_d=end_m)
        if seg_pts3d.shape[0] >= 2:
            try:
                pts2d = [QgsPointXY(float(x), float(y)) for x, y, z in seg_pts3d]
                seg_geom_metric = QgsGeometry.fromPolylineXY(pts2d)
            except Exception:
                seg_geom_metric = None
        return seg_pts3d, seg_geom_metric

    # --- PROCESO PRINCIPAL POR LÍNEA (usa merge absorb_next) ---
    def analyze_line(self, geom: QgsGeometry, dem_layer: QgsRasterLayer, line_id, contact_crs, log_func,
                     user_step: float = 0.0, target_scale_m: float = None):
        try:
            log_func(f"\n--- Analizando Línea ID: {line_id} ---", debug=True)

            if target_scale_m is None:
                target_scale_m = self.target_scale_m

            dem_crs = dem_layer.crs()
            metric_crs = self._get_metric_crs(contact_crs, geom)

            ctx = QgsProject.instance().transformContext()
            to_metric = QgsCoordinateTransform(contact_crs, metric_crs, ctx)
            to_dem = QgsCoordinateTransform(metric_crs, dem_crs, ctx)
            to_contact = QgsCoordinateTransform(metric_crs, contact_crs, ctx)

            geom_m = QgsGeometry(geom)
            try:
                geom_m.transform(to_metric)
            except Exception:
                pass

            try:
                geom_len = geom_m.length()
            except Exception:
                geom_len = 0.0
            if geom_len <= 0.0:
                log_func("  → Geometría con longitud 0, se omite", debug=True)
                return []

            # seleccionar paso: user_step > 0 -> usarlo, si =0 usar heurística
            if user_step and float(user_step) > 0.0:
                step_m = float(user_step)
            else:
                step_m = self.auto_sampling_step(dem_layer, target_scale_m)

            # muestreo 3D sobre toda la línea (controlado)
            pts3d_all = self._sample_points(geom_m, dem_layer, to_dem, step_m, log_func, start_d=0.0, end_d=geom_len)
            if pts3d_all.shape[0] < 6:
                log_func("  → Insuficientes puntos para análisis (muestreo completo).", debug=True)
                return []

            # segmentación inicial (sin fusión)
            segs_init = self.segment_by_changepoints(pts3d_all, step_m, target_scale_m, log_func)
            if len(segs_init) == 0:
                return []

            # fusión: absorber tramos cortos hacia el siguiente
            segs_merged = self._merge_segments_absorb_next(segs_init, step_m, log_func)
            log_func(f"  → Segments merged (count): {len(segs_merged)}", debug=True)

            seg_results = []
            for new_idx, merged_info in enumerate(segs_merged):
                a = merged_info['start_i']
                b = merged_info['end_i']
                orig_idxs = merged_info['orig_idxs']

                # concatenar pts del rango (a:b)
                combined = pts3d_all[a:b]
                if combined.shape[0] < 3:
                    log_func(f"    · Tramo resultante {new_idx} tiene <3 pts, se omite", debug=True)
                    continue

                # PCA sobre combinado
                nrm, planarity = self.pca_normal(combined)
                strike, dip, dip_dir = self.normal_to_angles(nrm)

                start_m = float(a * step_m)
                end_m = float((b - 1) * step_m)
                mid_m = 0.5 * (start_m + end_m)

                # punto medio sobre la línea (CRS métrico)
                try:
                    mid_pt_geom_m = geom_m.interpolate(mid_m)
                except Exception:
                    mid_pt_geom_m = geom_m.centroid()
                mid_pt_geom_contact = QgsGeometry(mid_pt_geom_m)
                try:
                    mid_pt_geom_contact.transform(to_contact)
                except Exception:
                    pass

                # extraer la geometría real del tramo y muestrear Z sobre ella (seg_pts3d)
                seg_pts3d, seg_geom_metric = self._extract_segment_line_and_geom(
                    geom_m, dem_layer, to_dem, start_m, end_m, step_m, log_func
                )
                seg_geom_contact = None
                if isinstance(seg_geom_metric, QgsGeometry):
                    seg_geom_contact = QgsGeometry(seg_geom_metric)
                    try:
                        seg_geom_contact.transform(to_contact)
                    except Exception:
                        pass

                merged_from_txt = ",".join(str(int(ii)) for ii in orig_idxs)
                n_merged = len(orig_idxs)

                seg_results.append({
                    'line_id':    line_id,
                    'seg_id':     int(new_idx),
                    'start_m':    start_m,
                    'mid_m':      mid_m,
                    'end_m':      end_m,
                    'len_m':      float(max(0.0, end_m - start_m)),
                    'npts':       int(combined.shape[0]),
                    'rumbo':      float(strike),
                    'buzamiento': float(dip),
                    'dir_buz':    float(dip_dir),
                    'planarity':  float(planarity),
                    'metodo':     'Best-Fit Plane (PCA) – Changepoints (merged->next)',
                    'pt_geom':    mid_pt_geom_contact,
                    'pts3d':      combined,
                    'normal':     nrm,
                    'center':     np.mean(combined, axis=0),
                    'step_m':     float(step_m),
                    'seg_geom':   seg_geom_contact,
                    'seg_pts3d':  seg_pts3d,
                    'merged_from': merged_from_txt,  # ej: "0,1,2"
                    'n_merged':   int(n_merged)
                })

            return seg_results

        except Exception as e:
            log_func(f"  → EXCEPCIÓN: {e}", debug=True)
            traceback.print_exc()
            return []


# ==========================
# VISOR 3D (matplotlib)
# ==========================
class Plane3DViewerDialog(QDialog):
    def __init__(self, seg_record: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Vista 3D – Línea {seg_record['line_id']} | Segmento {seg_record['seg_id']}")
        self.setMinimumSize(940, 720)

        self.seg = seg_record
        layout = QVBoxLayout(self)

        info = QLabel(
            f"<b>Segmento:</b> L{seg_record['line_id']}–S{seg_record['seg_id']} &nbsp; "
            f"<b>Pts:</b> {seg_record['npts']} &nbsp; "
            f"<b>R:</b> {seg_record['rumbo']:.1f}° &nbsp; "
            f"<b>B:</b> {seg_record['buzamiento']:.1f}° &nbsp; "
            f"<b>Planarity:</b> {seg_record['planarity']:.3f} &nbsp; "
            f"<b>Len:</b> {seg_record['len_m']:.1f} m &nbsp; "
            f"<b>MergedFrom:</b> {seg_record.get('merged_from','-')}"
        )
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        # canvas + toolbar
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 0.6])

        self.sc_points = None
        self.ln_path = None
        self.surf_plane = None

        self._plot_segment(self.seg)

        # controls
        hb = QHBoxLayout()
        btn_reset = QPushButton("Reset vista")
        btn_reset.clicked.connect(self._reset_view)
        hb.addWidget(btn_reset)

        btn_fit = QPushButton("Encajar datos")
        btn_fit.clicked.connect(self._fit_bounds)
        hb.addWidget(btn_fit)

        self.chk_points = QCheckBox("Puntos")
        self.chk_points.setChecked(True)
        self.chk_points.toggled.connect(self._toggle_points)
        hb.addWidget(self.chk_points)

        self.chk_line = QCheckBox("Tramo (línea real)")
        self.chk_line.setChecked(True)
        self.chk_line.toggled.connect(self._toggle_line)
        hb.addWidget(self.chk_line)

        self.chk_plane = QCheckBox("Plano PCA")
        self.chk_plane.setChecked(True)
        self.chk_plane.toggled.connect(self._toggle_plane)
        hb.addWidget(self.chk_plane)

        layout.addLayout(hb)

    def _plot_segment(self, seg):
        self.ax.cla()
        self.ax.set_box_aspect([1, 1, 0.6])

        pts_pca = np.array(seg.get('pts3d', np.empty((0, 3))))
        pts_line = np.array(seg.get('seg_pts3d', np.empty((0, 3))))
        if pts_line is None or pts_line.size == 0:
            pts_line = pts_pca

        if pts_pca.size > 0:
            self.sc_points = self.ax.scatter(
                pts_pca[:, 0], pts_pca[:, 1], pts_pca[:, 2],
                s=20, alpha=0.9, marker='o',
                facecolors='C0',
                edgecolors=(0.0, 0.0, 0.0, 0.75),
                linewidths=0.5
            )

        try:
            if pts_line.size > 0:
                self.ln_path, = self.ax.plot(
                    pts_line[:, 0], pts_line[:, 1], pts_line[:, 2],
                    linewidth=1.6, alpha=0.95, linestyle='-', marker=None
                )
            else:
                self.ln_path = None
        except Exception:
            self.ln_path = None

        # Plano PCA
        if pts_pca.size > 0:
            nrm = np.array(seg.get('normal', [0, 0, 1]))
            ctr = np.array(seg.get('center', np.mean(pts_pca, axis=0)))
            pad = 0.06
            xmin, xmax = pts_pca[:, 0].min(), pts_pca[:, 0].max()
            ymin, ymax = pts_pca[:, 1].min(), pts_pca[:, 1].max()
            dx = (xmax - xmin) if xmax > xmin else 1.0
            dy = (ymax - ymin) if ymax > ymin else 1.0
            xmin -= pad * dx; xmax += pad * dx
            ymin -= pad * dy; ymax += pad * dy
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
            d = -np.dot(nrm, ctr)
            nz = nrm[2] if abs(nrm[2]) >= 1e-9 else (1e-9 if nrm[2] >= 0 else -1e-9)
            zz = (-nrm[0] * xx - nrm[1] * yy - d) / nz
            try:
                self.surf_plane = self.ax.plot_surface(xx, yy, zz, alpha=0.35)
            except Exception:
                self.surf_plane = None
        else:
            self.surf_plane = None

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title(f"Plano PCA – L{seg['line_id']} S{seg['seg_id']}")

        self._fit_bounds()
        self.ax.text2D(0.02, 0.98, "Puntos (◼) + Tramo real (—) + Plano", transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def _reset_view(self):
        self.ax.view_init(elev=20, azim=-60)
        self.canvas.draw_idle()

    def _fit_bounds(self):
        pts = np.array(self.seg.get('seg_pts3d') if self.seg.get('seg_pts3d') is not None else self.seg.get('pts3d', np.empty((0, 3))))
        if pts.size == 0:
            return
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
        px = max(1e-6, 0.05 * (xmax - xmin) if xmax > xmin else 1.0)
        py = max(1e-6, 0.05 * (ymax - ymin) if ymax > ymin else 1.0)
        pz = max(1e-6, 0.05 * (zmax - zmin) if zmax > zmin else 1.0)
        self.ax.set_xlim(xmin - px, xmax + px)
        self.ax.set_ylim(ymin - py, ymax + py)
        self.ax.set_zlim(zmin - pz, zmax + pz)
        self.canvas.draw_idle()

    def _toggle_points(self, on: bool):
        if self.sc_points is not None:
            try:
                self.sc_points.set_visible(bool(on))
            except Exception:
                pass
            self.canvas.draw_idle()

    def _toggle_line(self, on: bool):
        if self.ln_path is not None:
            try:
                self.ln_path.set_visible(bool(on))
            except Exception:
                pass
            self.canvas.draw_idle()

    def _toggle_plane(self, on: bool):
        if self.surf_plane is not None:
            try:
                self.surf_plane.set_visible(bool(on))
            except Exception:
                pass
            self.canvas.draw_idle()


# ==========================
# RESULTS BROWSER
# ==========================
class ResultsBrowserDialog(QDialog):
    def __init__(self, results_all, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resultados – Segmentos detectados")
        self.setMinimumSize(700, 500)
        self.results_all = results_all

        layout = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setSelectionMode(self.list.SingleSelection)
        layout.addWidget(self.list)

        for seg in self.results_all:
            txt = (f"L{seg['line_id']} – S{seg['seg_id']} | "
                   f"len={seg['len_m']:.1f} m, npts={seg['npts']}, "
                   f"R={seg['rumbo']:.1f}°, B={seg['buzamiento']:.1f}°, "
                   f"merged_from={seg.get('merged_from','-')}")
            item = QListWidgetItem(txt)
            item.setData(Qt.UserRole, seg)
            self.list.addItem(item)

        hb = QHBoxLayout()
        self.btn_view = QPushButton("Ver en 3D")
        self.btn_view.setEnabled(self.list.count() > 0)
        self.btn_view.clicked.connect(self._open_selected_3d)
        hb.addWidget(self.btn_view)

        self.btn_close = QPushButton("Cerrar")
        self.btn_close.clicked.connect(self.accept)
        hb.addWidget(self.btn_close)
        layout.addLayout(hb)

    def _open_selected_3d(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.warning(self, "Sin selección", "Selecciona un segmento de la lista.")
            return
        seg = item.data(Qt.UserRole)
        viewer = Plane3DViewerDialog(seg, self)
        viewer.exec_()


# ==========================
# PLUGIN QGIS (con parámetros en UI)
# ==========================
class StructuralAnalysisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.menu = '&Análisis Best-Fit Plane (PCA)'
        self.toolbar = iface.addToolBar('BestFitPlanePCA')
        self.dialog = None

    def tr(self, text):
        return QCoreApplication.translate('StructuralAnalysisPlugin', text)

    def add_action(self, text, callback):
        a = QAction(QIcon(), text, self.iface.mainWindow())
        a.triggered.connect(callback)
        self.iface.addPluginToMenu(self.menu, a)
        self.toolbar.addAction(a)
        self.actions.append(a)

    def initGui(self):
        self.add_action(self.tr('Análisis Best-Fit Plane (PCA)'), self.run)

    def unload(self):
        for a in self.actions:
            self.iface.removePluginMenu(self.menu, a)
            self.iface.removeToolBarIcon(a)

    def run(self):
        if not self.dialog:
            self.dialog = GeologicalAnalysisDialog(self.iface.mainWindow())
            self.dialog.view_results_btn.clicked.connect(self.open_results_browser)
        if self.dialog.exec_():
            self.execute(self.dialog)

    def execute(self, dlg: GeologicalAnalysisDialog):
        dlg.log_message("▶️ Iniciando PCA con segmentación automática", debug=True)
        L = dlg.get_selected_layer(dlg.contact_layer)
        R = dlg.get_selected_layer(dlg.dem_layer)
        out = dlg.output_path.text()

        if not L or not R or not out or out == dlg.tr("Seleccionar…"):
            QMessageBox.warning(None, dlg.tr('Faltan datos'), dlg.tr('Seleccione capas y ruta de salida.'))
            return
        if not out.lower().endswith('.shp'):
            out += '.shp'
        if L.featureCount() == 0:
            QMessageBox.warning(None, dlg.tr('Sin líneas'), dlg.tr('La capa está vacía.'))
            return

        user_step = float(dlg.spin_step.value())
        min_seg_length = float(dlg.spin_minseg.value())
        target_scale = float(dlg.spin_target_scale.value())

        analyzer = StructuralAnalyzer(target_scale_m=target_scale, min_segment_length_m=min_seg_length)
        results_all = []

        dlg.progress.setMaximum(L.featureCount())
        dlg.progress.setVisible(True)

        cnt = 0
        for i, feat in enumerate(L.getFeatures()):
            dlg.progress.setValue(i + 1)
            segs = analyzer.analyze_line(
                feat.geometry(), R, feat.id(),
                L.crs(), dlg.log_message,
                user_step=user_step, target_scale_m=target_scale
            )
            for res in segs:
                results_all.append(res)
                dlg.log_message(
                    f"✔️ L{feat.id()} S{res['seg_id']}: "
                    f"R={res['rumbo']:.1f}°, B={res['buzamiento']:.1f}°, "
                    f"len={res['len_m']:.1f} m (mid={res['mid_m']:.1f} m) merged_from={res.get('merged_from','-')}",
                    debug=True
                )
                cnt += 1
            if not segs:
                dlg.log_message(f"❌ L{feat.id()}: sin segmentos válidos", debug=True)

        dlg.progress.setVisible(False)

        if not results_all:
            QMessageBox.warning(None, dlg.tr('Sin resultados'),
                                dlg.tr('No se generaron segmentos. Ajuste la geometría o el DEM.'))
            return

        # Crear shapefile de puntos con campo merged_from y n_merged
        if self.create_output_file(results_all, out, L.crs()):
            layer_out = QgsVectorLayer(out, out.split('/')[-1], 'ogr')
            if layer_out.isValid():
                QgsProject.instance().addMapLayer(layer_out)
                QMessageBox.information(None, dlg.tr('Listo'),
                                        dlg.tr(f'Análisis completado: {len(results_all)} planos/segmentos.'))
            else:
                QMessageBox.critical(None, dlg.tr('Error'),
                                     dlg.tr('No se pudo cargar la capa de salida.'))
        else:
            QMessageBox.critical(None, dlg.tr('Error'),
                                 dlg.tr('No se pudo crear el archivo de salida.'))

        # crear shapefile de tramos (opcional) con merged_from
        base = out[:-4]
        seg_out = f"{base}_segs.shp"
        if any(r.get('seg_geom') for r in results_all):
            if self.create_segments_file(results_all, seg_out, L.crs()):
                layer_segs = QgsVectorLayer(seg_out, seg_out.split('/')[-1], 'ogr')
                if layer_segs.isValid():
                    QgsProject.instance().addMapLayer(layer_segs)

        dlg.results_all = results_all
        dlg.view_results_btn.setEnabled(True)

    def open_results_browser(self):
        if not self.dialog or not getattr(self.dialog, 'results_all', None):
            QMessageBox.information(None, "Sin resultados",
                                    "Primero ejecuta el análisis para ver segmentos.")
            return
        rb = ResultsBrowserDialog(self.dialog.results_all, self.dialog)
        rb.exec_()

    def create_output_file(self, results, path, crs):
        """Shapefile de puntos (punto medio) incluyendo merged_from y n_merged."""
        fields = QgsFields([
            QgsField('line_id',    QVariant.Int,    'Integer', 10, 0),
            QgsField('seg_id',     QVariant.Int,    'Integer', 10, 0),
            QgsField('start_m',    QVariant.Double, 'Double',  12, 2),
            QgsField('mid_m',      QVariant.Double, 'Double',  12, 2),
            QgsField('end_m',      QVariant.Double, 'Double',  12, 2),
            QgsField('len_m',      QVariant.Double, 'Double',  12, 2),
            QgsField('npts',       QVariant.Int,    'Integer', 10, 0),
            QgsField('rumbo',      QVariant.Double, 'Double',  10, 2),
            QgsField('buzamiento', QVariant.Double, 'Double',  10, 2),
            QgsField('dir_buz',    QVariant.Double, 'Double',  10, 2),
            QgsField('planarity',  QVariant.Double, 'Double',  10, 3),
            QgsField('n_merged',   QVariant.Int,    'Integer', 3, 0),
            QgsField('merged_from',QVariant.String, 'String',  200, 0),
            QgsField('metodo',     QVariant.String, 'String',  80, 0),
        ])

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = 'ESRI Shapefile'
        writer = QgsVectorFileWriter.create(
            path, fields, QgsWkbTypes.Point, crs,
            QgsProject.instance().transformContext(), opts
        )
        if writer.hasError() != QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage(
                f"Error creando shapefile: {writer.errorMessage()}",
                'BestFitPlanePCA', Qgis.Critical
            )
            return False

        for res in results:
            pt_geom = res.get('pt_geom')
            if not isinstance(pt_geom, QgsGeometry):
                continue
            feat = QgsFeature(fields)
            feat.setGeometry(pt_geom)
            feat.setAttributes([
                res['line_id'],
                res['seg_id'],
                res['start_m'],
                res['mid_m'],
                res['end_m'],
                res['len_m'],
                res['npts'],
                res['rumbo'],
                res['buzamiento'],
                res['dir_buz'],
                res['planarity'],
                res.get('n_merged', 1),
                res.get('merged_from', ''),
                res['metodo']
            ])
            writer.addFeature(feat)

        del writer
        return True

    def create_segments_file(self, results, path, crs):
        """Shapefile de tramos (LineString) con merged_from y n_merged (si seg_geom existe)."""
        fields = QgsFields([
            QgsField('line_id',    QVariant.Int,    'Integer', 10, 0),
            QgsField('seg_id',     QVariant.Int,    'Integer', 10, 0),
            QgsField('len_m',      QVariant.Double, 'Double',  12, 2),
            QgsField('npts',       QVariant.Int,    'Integer', 10, 0),
            QgsField('rumbo',      QVariant.Double, 'Double',  10, 2),
            QgsField('buzamiento', QVariant.Double, 'Double',  10, 2),
            QgsField('planarity',  QVariant.Double, 'Double',  10, 3),
            QgsField('n_merged',   QVariant.Int,    'Integer', 3, 0),
            QgsField('merged_from',QVariant.String, 'String',  200, 0),
            QgsField('metodo',     QVariant.String, 'String',  80, 0),
        ])
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = 'ESRI Shapefile'
        writer = QgsVectorFileWriter.create(
            path, fields, QgsWkbTypes.LineString, crs,
            QgsProject.instance().transformContext(), opts
        )
        if writer.hasError() != QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage(
                f"Error creando shapefile de tramos: {writer.errorMessage()}",
                'BestFitPlanePCA', Qgis.Critical
            )
            return False

        for res in results:
            seg_geom = res.get('seg_geom')
            if not isinstance(seg_geom, QgsGeometry):
                continue
            feat = QgsFeature(fields)
            feat.setGeometry(seg_geom)
            feat.setAttributes([
                res['line_id'],
                res['seg_id'],
                res['len_m'],
                res['npts'],
                res['rumbo'],
                res['buzamiento'],
                res['planarity'],
                res.get('n_merged', 1),
                res.get('merged_from', ''),
                res['metodo']
            ])
            writer.addFeature(feat)
        del writer
        return True
