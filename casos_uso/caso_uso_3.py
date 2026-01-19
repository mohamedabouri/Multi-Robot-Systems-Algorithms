"""
Caso de uso 3: Comparación directa entre ambos algoritmos
Compara el Algoritmo 1 y el Algoritmo 2 en las mismas condiciones.
"""
import sys
import os
# Añadir el directorio raíz al path de Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import os
from datetime import datetime
from typing import Dict, Tuple
from utils.robot import Robot
from utils.tarea import Tarea
from utils.entorno import Entorno
from utils.metricas import calcular_metricas
from algoritmos.greedy_distribuido import AlgoritmoGreedyDistribuido
from algoritmos.consenso_incertidumbre import AlgoritmoConsensoIncertidumbre


def generar_posicion_aleatoria(dimensiones: tuple) -> tuple:
    """Genera una posición aleatoria dentro de las dimensiones."""
    return (random.randint(0, dimensiones[0] - 1), random.randint(0, dimensiones[1] - 1))


def crear_entorno_comun(num_robots: int = 5,
                        num_tareas: int = 10,
                        dimensiones: tuple = (10, 10),
                        capacidad: int = 1,
                        rango_percepcion: int = 3,
                        rango_comunicacion: int = 5,
                        ruido_percepcion: float = 0.0,
                        probabilidad_fallo_comunicacion: float = 0.0,
                        semilla: int = None) -> Tuple[Entorno, Entorno]:
    """
    Crea dos entornos idénticos para comparar ambos algoritmos.
    
    Returns:
        Tupla (entorno_greedy, entorno_consenso)
    """
    if semilla is not None:
        np.random.seed(semilla)
        random.seed(semilla)
    
    # Crear robots (mismas posiciones para ambos)
    robots_greedy = []
    robots_consenso = []
    for i in range(num_robots):
        posicion_inicial = generar_posicion_aleatoria(dimensiones)
        
        robot_g = Robot(
            id_robot=i,
            posicion_inicial=posicion_inicial,
            capacidad=capacidad,
            rango_percepcion=rango_percepcion,
            rango_comunicacion=rango_comunicacion
        )
        robots_greedy.append(robot_g)
        
        robot_c = Robot(
            id_robot=i,
            posicion_inicial=posicion_inicial,
            capacidad=capacidad,
            rango_percepcion=rango_percepcion,
            rango_comunicacion=rango_comunicacion
        )
        robots_consenso.append(robot_c)
    
    # Crear tareas (mismas para ambos)
    tareas_greedy = []
    tareas_consenso = []
    for j in range(num_tareas):
        posicion_pickup = generar_posicion_aleatoria(dimensiones)
        posicion_delivery = generar_posicion_aleatoria(dimensiones)
        
        while posicion_delivery == posicion_pickup:
            posicion_delivery = generar_posicion_aleatoria(dimensiones)
        
        tarea_g = Tarea(
            id_tarea=j,
            posicion_pickup=posicion_pickup,
            posicion_delivery=posicion_delivery,
            posicion_real_pickup=posicion_pickup,
            posicion_real_delivery=posicion_delivery
        )
        tareas_greedy.append(tarea_g)
        
        tarea_c = Tarea(
            id_tarea=j,
            posicion_pickup=posicion_pickup,
            posicion_delivery=posicion_delivery,
            posicion_real_pickup=posicion_pickup,
            posicion_real_delivery=posicion_delivery
        )
        tareas_consenso.append(tarea_c)
    
    # Crear entornos
    entorno_greedy = Entorno(
        dimensiones=dimensiones,
        robots=robots_greedy,
        tareas=tareas_greedy,
        ruido_percepcion=ruido_percepcion,
        probabilidad_fallo_comunicacion=probabilidad_fallo_comunicacion
    )
    
    entorno_consenso = Entorno(
        dimensiones=dimensiones,
        robots=robots_consenso,
        tareas=tareas_consenso,
        ruido_percepcion=ruido_percepcion,
        probabilidad_fallo_comunicacion=probabilidad_fallo_comunicacion
    )
    
    return entorno_greedy, entorno_consenso


def comparar_algoritmos(ruido_percepcion: float = 0.0,
                       probabilidad_fallo_comunicacion: float = 0.0,
                       semilla: int = None,
                       verbose: bool = True) -> Dict:
    """
    Compara ambos algoritmos en las mismas condiciones.
    
    Args:
        ruido_percepcion: Desviación estándar del ruido
        probabilidad_fallo_comunicacion: Probabilidad de fallo de comunicación
        semilla: Semilla aleatoria
        verbose: Si True, imprime información detallada
        
    Returns:
        Diccionario con métricas de ambos algoritmos
    """
    # Crear entornos idénticos
    entorno_greedy, entorno_consenso = crear_entorno_comun(
        num_robots=5,
        num_tareas=10,
        ruido_percepcion=ruido_percepcion,
        probabilidad_fallo_comunicacion=probabilidad_fallo_comunicacion,
        semilla=semilla
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CASO DE USO 3: Comparación Directa")
        print(f"{'='*60}")
        print(f"Configuración:")
        print(f"  - Robots: 5")
        print(f"  - Tareas: 10")
        print(f"  - Ruido percepción (σ): {ruido_percepcion}")
        print(f"  - Prob. fallo comunicación: {probabilidad_fallo_comunicacion}")
        print(f"  - Semilla: {semilla}")
        print(f"{'='*60}\n")
    
    # Ejecutar Algoritmo 1 (Greedy)
    algoritmo_greedy = AlgoritmoGreedyDistribuido(entorno_greedy)
    rondas_greedy = algoritmo_greedy.ejecutar(max_rondas=1000)
    metricas_greedy = calcular_metricas(entorno_greedy)
    metricas_greedy['rondas_ejecutadas'] = rondas_greedy
    
    # Ejecutar Algoritmo 2 (Consenso)
    algoritmo_consenso = AlgoritmoConsensoIncertidumbre(
        entorno_consenso,
        epsilon=0.2,
        lambda_penalizacion=0.3,
        iteraciones_consenso=5,
        sigma_umbral=2.0
    )
    rondas_consenso = algoritmo_consenso.ejecutar(max_rondas=1000)
    metricas_consenso = calcular_metricas(entorno_consenso)
    metricas_consenso['rondas_ejecutadas'] = rondas_consenso
    
    if verbose:
        print("ALGORITMO 1 (Greedy Distribuido):")
        print(f"  Rondas: {metricas_greedy['rondas_ejecutadas']}")
        print(f"  Distancia total: {metricas_greedy['distancia_total']}")
        print(f"  Tasa completitud: {metricas_greedy['tasa_completitud']:.2%}")
        print(f"  Balance carga: {metricas_greedy['balance_carga']:.2f}")
        
        print("\nALGORITMO 2 (Consenso con Incertidumbre):")
        print(f"  Rondas: {metricas_consenso['rondas_ejecutadas']}")
        print(f"  Distancia total: {metricas_consenso['distancia_total']}")
        print(f"  Tasa completitud: {metricas_consenso['tasa_completitud']:.2%}")
        print(f"  Balance carga: {metricas_consenso['balance_carga']:.2f}")
        if metricas_consenso.get('error_estimacion', 0) > 0:
            print(f"  Error estimación: {metricas_consenso['error_estimacion']:.2f}")
        
        print("\nCOMPARACIÓN:")
        print(f"  Diferencia rondas: {metricas_consenso['rondas_ejecutadas'] - metricas_greedy['rondas_ejecutadas']}")
        print(f"  Diferencia distancia: {metricas_consenso['distancia_total'] - metricas_greedy['distancia_total']}")
        print(f"  Diferencia completitud: {(metricas_consenso['tasa_completitud'] - metricas_greedy['tasa_completitud']):.2%}")
    
    return {
        'greedy': metricas_greedy,
        'consenso': metricas_consenso
    }


def guardar_resultados(resultados: Dict, nombre_archivo: str):
    """
    Guarda los resultados en un archivo de texto.
    
    Args:
        resultados: Diccionario con los resultados de los experimentos
        nombre_archivo: Nombre del archivo donde guardar
    """
    # Crear directorio de resultados si no existe
    directorio_resultados = "resultados"
    if not os.path.exists(directorio_resultados):
        os.makedirs(directorio_resultados)
    
    ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)
    
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CASO DE USO 3: Comparación Directa entre Algoritmos\n")
        f.write("="*80 + "\n")
        f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for escenario_nombre, escenario_resultados in resultados.items():
            f.write("\n" + "="*80 + "\n")
            f.write(f"{escenario_nombre.upper().replace('_', ' ')}\n")
            f.write("="*80 + "\n\n")
            
            if escenario_resultados:
                # Calcular estadísticas para Greedy
                promedios_greedy = {}
                desviaciones_greedy = {}
                for key in escenario_resultados[0]['greedy'].keys():
                    valores = [r['greedy'][key] for r in escenario_resultados]
                    promedios_greedy[key] = np.mean(valores)
                    desviaciones_greedy[key] = np.std(valores)
                
                # Calcular estadísticas para Consenso
                promedios_consenso = {}
                desviaciones_consenso = {}
                for key in escenario_resultados[0]['consenso'].keys():
                    valores = [r['consenso'][key] for r in escenario_resultados]
                    promedios_consenso[key] = np.mean(valores)
                    desviaciones_consenso[key] = np.std(valores)
                
                f.write("ALGORITMO 1 (Greedy Distribuido) - Promedio (10 ejecuciones):\n")
                f.write("-"*80 + "\n")
                for key in sorted(promedios_greedy.keys()):
                    f.write(f"  {key:30s}: {promedios_greedy[key]:10.2f} ± {desviaciones_greedy[key]:.2f}\n")
                
                f.write("\nALGORITMO 2 (Consenso con Incertidumbre) - Promedio (10 ejecuciones):\n")
                f.write("-"*80 + "\n")
                for key in sorted(promedios_consenso.keys()):
                    f.write(f"  {key:30s}: {promedios_consenso[key]:10.2f} ± {desviaciones_consenso[key]:.2f}\n")
                
                f.write("\nCOMPARACIÓN (Consenso - Greedy):\n")
                f.write("-"*80 + "\n")
                for key in sorted(promedios_greedy.keys()):
                    if key in promedios_consenso:
                        diferencia = promedios_consenso[key] - promedios_greedy[key]
                        f.write(f"  {key:30s}: {diferencia:10.2f}\n")
                
                f.write("\nResultados individuales:\n")
                f.write("-"*80 + "\n")
                for i, resultado in enumerate(escenario_resultados):
                    f.write(f"\nEjecución {i+1}:\n")
                    f.write("  Greedy:\n")
                    for key in sorted(resultado['greedy'].keys()):
                        f.write(f"    {key:30s}: {resultado['greedy'][key]:10.2f}\n")
                    f.write("  Consenso:\n")
                    for key in sorted(resultado['consenso'].keys()):
                        f.write(f"    {key:30s}: {resultado['consenso'][key]:10.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Resultados guardados en: {ruta_archivo}")


def ejecutar_caso_uso_3():
    """Ejecuta todos los experimentos del Caso de uso 3."""
    resultados = {}
    
    # Escenario 3.1: Sin incertidumbre
    print("\n" + "="*60)
    print("ESCENARIO 3.1: Sin incertidumbre (condiciones ideales para Algoritmo 1)")
    print("="*60)
    resultados_3_1 = []
    for semilla in range(10):
        resultado = comparar_algoritmos(
            ruido_percepcion=0.0,
            probabilidad_fallo_comunicacion=0.0,
            semilla=semilla,
            verbose=False
        )
        resultados_3_1.append(resultado)
    
    resultados['escenario_3_1'] = resultados_3_1
    
    # Calcular promedios
    print("\nResultados promedio Algoritmo 1 (10 ejecuciones):")
    promedios_greedy = {}
    for key in resultados_3_1[0]['greedy'].keys():
        valores = [r['greedy'][key] for r in resultados_3_1]
        promedios_greedy[key] = np.mean(valores)
        print(f"  {key}: {promedios_greedy[key]:.2f}")
    
    print("\nResultados promedio Algoritmo 2 (10 ejecuciones):")
    promedios_consenso = {}
    for key in resultados_3_1[0]['consenso'].keys():
        valores = [r['consenso'][key] for r in resultados_3_1]
        promedios_consenso[key] = np.mean(valores)
        print(f"  {key}: {promedios_consenso[key]:.2f}")
    
    # Escenario 3.2: Incertidumbre moderada
    print("\n" + "="*60)
    print("ESCENARIO 3.2: Incertidumbre moderada (σ = 0.5)")
    print("="*60)
    resultados_3_2 = []
    for semilla in range(10):
        resultado = comparar_algoritmos(
            ruido_percepcion=0.5,
            probabilidad_fallo_comunicacion=0.1,
            semilla=semilla,
            verbose=False
        )
        resultados_3_2.append(resultado)
    
    resultados['escenario_3_2'] = resultados_3_2
    
    # Calcular promedios
    print("\nResultados promedio Algoritmo 1 (10 ejecuciones):")
    promedios_greedy = {}
    for key in resultados_3_2[0]['greedy'].keys():
        valores = [r['greedy'][key] for r in resultados_3_2]
        promedios_greedy[key] = np.mean(valores)
        print(f"  {key}: {promedios_greedy[key]:.2f}")
    
    print("\nResultados promedio Algoritmo 2 (10 ejecuciones):")
    promedios_consenso = {}
    for key in resultados_3_2[0]['consenso'].keys():
        valores = [r['consenso'][key] for r in resultados_3_2]
        promedios_consenso[key] = np.mean(valores)
        print(f"  {key}: {promedios_consenso[key]:.2f}")
    
    # Escenario 3.3: Incertidumbre alta
    print("\n" + "="*60)
    print("ESCENARIO 3.3: Incertidumbre alta (σ = 1.5)")
    print("="*60)
    resultados_3_3 = []
    for semilla in range(10):
        resultado = comparar_algoritmos(
            ruido_percepcion=1.5,
            probabilidad_fallo_comunicacion=0.1,
            semilla=semilla,
            verbose=False
        )
        resultados_3_3.append(resultado)
    
    resultados['escenario_3_3'] = resultados_3_3
    
    # Calcular promedios
    print("\nResultados promedio Algoritmo 1 (10 ejecuciones):")
    promedios_greedy = {}
    for key in resultados_3_3[0]['greedy'].keys():
        valores = [r['greedy'][key] for r in resultados_3_3]
        promedios_greedy[key] = np.mean(valores)
        print(f"  {key}: {promedios_greedy[key]:.2f}")
    
    print("\nResultados promedio Algoritmo 2 (10 ejecuciones):")
    promedios_consenso = {}
    for key in resultados_3_3[0]['consenso'].keys():
        valores = [r['consenso'][key] for r in resultados_3_3]
        promedios_consenso[key] = np.mean(valores)
        print(f"  {key}: {promedios_consenso[key]:.2f}")
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    guardar_resultados(resultados, f"caso_uso_3_{timestamp}.txt")
    
    return resultados


if __name__ == "__main__":
    ejecutar_caso_uso_3()
