# -*- coding: utf-8 -*-
"""
Plugin QGIS: Análisis Geológico Estructural – Best-Fit Plane (PCA)

Reproyecta internamente la geometría de líneas a CRS métrico (UTM),
densifica en metros, muestrea el DEM en su CRS original y calcula
strike (rumbo), dip (buzamiento) y dip‐direction (dirección de buzamiento)
mediante Análisis de Componentes Principales (PCA), respetando la convención
de azimut geológico y la regla de la mano derecha.
"""

import math
import traceback
import numpy as np

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAction, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QProgressBar, QTextEdit,
    QGroupBox, QGridLayout, QFileDialog, QComboBox, QCheckBox
)
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer, QgsFeature,
    QgsGeometry, QgsPointXY, QgsField, QgsFields,
    QgsVectorFileWriter, QgsWkbTypes, QgsMessageLog, Qgis,
    QgsDistanceArea, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform
)


class GeologicalAnalysisDialog(QDialog):
    """Diálogo para parámetros, opciones y selección de capas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle(self.tr("Análisis Geológico – Best-Fit Plane (PCA)"))
        self.setFixedSize(650, 650)
        layout = QVBoxLayout(self)

        # Parámetros de muestreo
        params = QGroupBox(self.tr("Parámetros de Muestreo"))
        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("Densidad de muestreo (m):")), 0, 0)
        self.sampling_density = QDoubleSpinBox()
        self.sampling_density.setRange(0.1, 100.0)
        self.sampling_density.setDecimals(2)
        self.sampling_density.setValue(1.0)
        grid.addWidget(self.sampling_density, 0, 1)
        params.setLayout(grid)
        layout.addWidget(params)

        # Opciones (debug)
        opts = QGroupBox(self.tr("Opciones"))
        vlay_opts = QVBoxLayout()
        self.show_debug = QCheckBox(self.tr("Mostrar logs detallados (debug)"))
        vlay_opts.addWidget(self.show_debug)
        opts.setLayout(vlay_opts)
        layout.addWidget(opts)

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

        # Progreso y log
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log)

        # Botones
        hb = QHBoxLayout()
        run_btn = QPushButton(self.tr("Ejecutar"))
        run_btn.clicked.connect(self.accept)
        hb.addWidget(run_btn)
        cancel_btn = QPushButton(self.tr("Cancelar"))
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
        self.output_path.setText(fname)

    def get_selected_layer(self, combo):
        return QgsProject.instance().mapLayer(combo.currentData())

    def log_message(self, message, debug=False):
        """Añade texto al log; si es debug solo si está marcado."""
        if debug and not self.show_debug.isChecked():
            return
        self.log.append(message)
        QCoreApplication.processEvents()


class StructuralAnalyzer:
    """Analizador estructural usando ajuste de plano por PCA."""
    def __init__(self, sampling_density: float):
        self.sampling_density = sampling_density
        self.dist_calc = QgsDistanceArea()
        self.dist_calc.setEllipsoid(QgsProject.instance().ellipsoid())

    def analyze_line(self, geom, dem_layer, line_id, contact_crs, log_func):
        try:
            log_func(f"\n--- Analizando Línea ID: {line_id} ---", debug=True)

            # CRS DEM y CRS métrico
            dem_crs    = dem_layer.crs()
            metric_crs = self._get_metric_crs(contact_crs, geom)

            # Transformadores
            ctx       = QgsProject.instance().transformContext()
            to_metric = QgsCoordinateTransform(contact_crs, metric_crs, ctx)
            to_dem    = QgsCoordinateTransform(metric_crs,   dem_crs,    ctx)

            # Reproyección y muestreo
            geom_m = QgsGeometry(geom)
            geom_m.transform(to_metric)
            pts3d = self._sample_points(geom_m, dem_layer, to_dem, log_func)
            if pts3d.shape[0] < 3:
                log_func("  → ERROR: se requieren al menos 3 puntos muestreados", debug=True)
                return None

            # Cálculo PCA
            return self._calculate_pca_method(pts3d, log_func)

        except Exception as e:
            log_func(f"  → EXCEPCIÓN: {e}", debug=True)
            traceback.print_exc()
            return None

    def _sample_points(self, geom_metric, dem_layer, to_dem, log_func):
        """Densifica y muestrea Z."""
        geom_dens = geom_metric.densifyByDistance(self.sampling_density)
        prov      = dem_layer.dataProvider()
        pts       = []
        for v in geom_dens.vertices():
            ptm = QgsPointXY(v.x(), v.y())
            try:
                dem_pt = to_dem.transform(ptm)
            except Exception as e:
                log_func(f"  → ERROR al transformar: {e}", debug=True)
                continue
            z, ok = prov.sample(dem_pt, 1)
            if ok and not math.isnan(z):
                pts.append([ptm.x(), ptm.y(), float(z)])
        log_func(f"  -> {len(pts)} puntos muestreados", debug=True)
        return np.array(pts)

    def _get_metric_crs(self, crs, geom):
        """Si crs es geográfico, retorna UTM; si no, el mismo."""
        if not crs.isGeographic():
            return crs
        p = geom.centroid().asPoint()
        zone = int((p.x() + 180) / 6) + 1
        if zone < 1 or zone > 60:
            zone = 31
        epsg = 32600 + zone if p.y() >= 0 else 32700 + zone
        return QgsCoordinateReferenceSystem(epsg)

    def _calculate_pca_method(self, pts3d, log_func):
        """
        Calcula orientación del plano usando PCA,
        con azimut geológico y regla mano derecha.
        """
        # 1. Centrar datos
        mean_pt = np.mean(pts3d, axis=0)
        data_c  = pts3d - mean_pt

        # 2. Covarianza
        covm = np.cov(data_c, rowvar=False)

        # 3. Eigen descomposition
        vals, vecs = np.linalg.eigh(covm)

        # 4. Normal = eigenvector de menor eigenvalor
        nrm = vecs[:, np.argmin(vals)]
        # Asegurar componente Z positiva
        if nrm[2] < 0:
            nrm = -nrm
        nx, ny, nz = nrm

        # 5. Dip: ángulo entre normal y vertical
        dip = math.degrees(math.acos(nz))

        # 6. Dip direction: dirección downhill = azimut de (−nx, −ny)
        dx, dy = -nx, -ny
        dip_dir = (math.degrees(math.atan2(dx, dy)) + 360) % 360

        # 7. Strike (mano derecha)
        strike = (dip_dir - 90) % 360

        log_func(f"  → Strike={strike:.1f}°, Dip={dip:.1f}°, DipDir={dip_dir:.1f}°", debug=True)
        return {
            'metodo':     'Best-Fit Plane (PCA)',
            'rumbo':      strike,
            'buzamiento': dip,
            'dir_buz':    dip_dir
        }


class StructuralAnalysisPlugin:
    """Plugin principal QGIS."""
    def __init__(self, iface):
        self.iface   = iface
        self.actions = []
        self.menu    = '&Análisis Best-Fit Plane (PCA)'
        self.toolbar = iface.addToolBar('BestFitPlanePCA')
        self.dialog  = None

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
        if self.dialog.exec_():
            self.execute(self.dialog)

    def execute(self, dlg):
        dlg.log_message("▶️ Iniciando Best-Fit Plane (PCA)", debug=True)
        L = dlg.get_selected_layer(dlg.contact_layer)
        R = dlg.get_selected_layer(dlg.dem_layer)
        out = dlg.output_path.text()

        # Validaciones
        if not L or not R or not out or out == dlg.tr("Seleccionar…"):
            QMessageBox.warning(None, dlg.tr('Faltan datos'),
                                dlg.tr('Seleccione capas y ruta de salida.'))
            return
        if not out.lower().endswith('.shp'):
            out += '.shp'
        if L.featureCount() == 0:
            QMessageBox.warning(None, dlg.tr('Sin líneas'),
                                dlg.tr('La capa está vacía.'))
            return

        analyzer = StructuralAnalyzer(dlg.sampling_density.value())
        results, geoms = [], {}
        dlg.progress.setMaximum(L.featureCount())
        dlg.progress.setVisible(True)

        for i, feat in enumerate(L.getFeatures()):
            dlg.progress.setValue(i + 1)
            geoms[feat.id()] = feat.geometry()
            res = analyzer.analyze_line(
                feat.geometry(), R, feat.id(),
                L.crs(), dlg.log_message
            )
            if res:
                res['line_id'] = feat.id()
                results.append(res)
                dlg.log_message(
                    f"✔️ L{feat.id()}: R={res['rumbo']:.1f}°, "
                    f"B={res['buzamiento']:.1f}°", debug=True
                )
            else:
                dlg.log_message(f"❌ L{feat.id()}: fallo", debug=True)

        dlg.progress.setVisible(False)

        if not results:
            QMessageBox.warning(None, dlg.tr('Sin resultados'),
                                dlg.tr('Ajuste los parámetros.'))
            return

        if self.create_output_file(results, geoms, out, L.crs()):
            layer_out = QgsVectorLayer(out, out.split('/')[-1], 'ogr')
            if layer_out.isValid():
                QgsProject.instance().addMapLayer(layer_out)
                QMessageBox.information(None, dlg.tr('Listo'),
                                        dlg.tr(f'Análisis completado: {len(results)} líneas.'))
            else:
                QMessageBox.critical(None, dlg.tr('Error'),
                                     dlg.tr('No se pudo cargar la capa de salida.'))
        else:
            QMessageBox.critical(None, dlg.tr('Error'),
                                 dlg.tr('No se pudo crear el archivo de salida.'))

    def create_output_file(self, results, geometries, path, crs):
        """Crea un shapefile punto con los resultados."""
        fields = QgsFields([
            # line_id
            QgsField('line_id',    QVariant.Int,    'Integer', 10, 0),
            # rumbo (strike)
            QgsField('rumbo',      QVariant.Double, 'Double',  10, 2),
            # buzamiento (dip)
            QgsField('buzamiento', QVariant.Double, 'Double',  10, 2),
            # dirección de buzamiento
            QgsField('dir_buz',    QVariant.Double, 'Double',  10, 2),
            # método
            QgsField('metodo',     QVariant.String, 'String',  50, 0),
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
            geom = geometries.get(res['line_id'])
            if not geom:
                continue
            feat = QgsFeature(fields)
            feat.setGeometry(geom.centroid())
            feat.setAttributes([
                res['line_id'],
                res['rumbo'],
                res['buzamiento'],
                res['dir_buz'],
                res['metodo']
            ])
            writer.addFeature(feat)

        del writer
        return True