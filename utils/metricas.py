"""
Funciones para calcular métricas de evaluación de los algoritmos.
"""
from typing import List, Dict
import numpy as np
from utils.entorno import Entorno


def calcular_metricas(entorno: Entorno, historial: List[Dict] = None) -> Dict:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        entorno: Entorno de simulación
        historial: Historial de rondas (opcional)
        
    Returns:
        Diccionario con todas las métricas
    """
    metricas = {}
    
    # Número de rondas hasta completitud
    metricas['rondas_total'] = entorno.ronda_actual
    
    # Distancia total recorrida
    metricas['distancia_total'] = sum(r.distancia_recorrida for r in entorno.robots)
    
    # Tasa de completitud
    tareas_completadas = sum(1 for t in entorno.tareas.values() if t.estado.value == 'completada')
    metricas['tasa_completitud'] = tareas_completadas / len(entorno.tareas) if len(entorno.tareas) > 0 else 0.0
    
    # Balance de carga (desviación estándar del número de tareas completadas por robot)
    tareas_por_robot = [r.tareas_completadas_count for r in entorno.robots]
    metricas['balance_carga'] = np.std(tareas_por_robot) if len(tareas_por_robot) > 0 else 0.0
    
    # Error de estimación (solo para algoritmo con incertidumbre)
    errores = []
    for tarea in entorno.tareas.values():
        if hasattr(tarea, 'posicion_real_pickup') and tarea.posicion_real_pickup != tarea.posicion_pickup:
            error_pickup = distancia_manhattan(tarea.posicion_real_pickup, tarea.posicion_pickup)
            error_delivery = distancia_manhattan(tarea.posicion_real_delivery, tarea.posicion_delivery)
            errores.append((error_pickup + error_delivery) / 2)
    
    metricas['error_estimacion'] = np.mean(errores) if errores else 0.0
    
    # Tasa de éxito con incertidumbre
    if historial:
        metricas['tasa_exito_incertidumbre'] = metricas['tasa_completitud']
    else:
        metricas['tasa_exito_incertidumbre'] = metricas['tasa_completitud']
    
    return metricas


def distancia_manhattan(pos1: tuple, pos2: tuple) -> int:
    """Calcula la distancia de Manhattan entre dos posiciones."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def imprimir_metricas(metricas: Dict):
    """Imprime las métricas de forma legible."""
    print("\n" + "="*50)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*50)
    print(f"Rondas totales: {metricas['rondas_total']}")
    print(f"Distancia total recorrida: {metricas['distancia_total']}")
    print(f"Tasa de completitud: {metricas['tasa_completitud']:.2%}")
    print(f"Balance de carga (desv. est.): {metricas['balance_carga']:.2f}")
    if metricas.get('error_estimacion', 0) > 0:
        print(f"Error de estimación promedio: {metricas['error_estimacion']:.2f}")
    print("="*50 + "\n")
