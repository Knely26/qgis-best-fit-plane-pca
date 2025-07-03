# -*- coding: utf-8 -*-

def classFactory(iface):
    """
    Carga la clase principal del plugin desde el archivo mainPlugin.
    
    :param iface: Una instancia de la interfaz de QGIS.
    """
    # Se importa la clase 'GeologicalAnalysis' desde el archivo mainPlugin.py
    # (El punto '.' significa "desde la carpeta actual")
    from .mainPlugin import StructuralAnalysisPlugin
    return StructuralAnalysisPlugin(iface)
