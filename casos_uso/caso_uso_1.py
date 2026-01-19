"""
Caso de uso 1: Algoritmo Greedy Distribuido (sin incertidumbre)
Evalúa el Algoritmo 1 en un entorno determinista.
"""
import sys
import os
# Añadir el directorio raíz al path de Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import os
from datetime import datetime
from typing import Dict
from utils.robot import Robot
from utils.tarea import Tarea
from utils.entorno import Entorno
from utils.metricas import calcular_metricas, imprimir_metricas
from algoritmos.greedy_distribuido import AlgoritmoGreedyDistribuido


def generar_posicion_aleatoria(dimensiones: tuple) -> tuple:
    """Genera una posición aleatoria dentro de las dimensiones."""
    return (random.randint(0, dimensiones[0] - 1), random.randint(0, dimensiones[1] - 1))


def crear_entorno_experimento(num_robots: int = 5,
                              num_tareas: int = 10,
                              dimensiones: tuple = (10, 10),
                              capacidad: int = 1,
                              rango_percepcion: int = 3,
                              rango_comunicacion: int = 5,
                              semilla: int = None) -> Entorno:
    """
    Crea un entorno para el experimento.
    
    Args:
        num_robots: Número de robots
        num_tareas: Número de tareas
        dimensiones: Dimensiones del espacio
        capacidad: Capacidad de cada robot
        rango_percepcion: Rango de percepción
        rango_comunicacion: Rango de comunicación
        semilla: Semilla aleatoria para reproducibilidad
    """
    if semilla is not None:
        np.random.seed(semilla)
        random.seed(semilla)
    
    # Crear robots
    robots = []
    for i in range(num_robots):
        posicion_inicial = generar_posicion_aleatoria(dimensiones)
        robot = Robot(
            id_robot=i,
            posicion_inicial=posicion_inicial,
            capacidad=capacidad,
            rango_percepcion=rango_percepcion,
            rango_comunicacion=rango_comunicacion
        )
        robots.append(robot)
    
    # Crear tareas
    tareas = []
    for j in range(num_tareas):
        posicion_pickup = generar_posicion_aleatoria(dimensiones)
        posicion_delivery = generar_posicion_aleatoria(dimensiones)
        
        # Asegurar que pickup y delivery sean diferentes
        while posicion_delivery == posicion_pickup:
            posicion_delivery = generar_posicion_aleatoria(dimensiones)
        
        tarea = Tarea(
            id_tarea=j,
            posicion_pickup=posicion_pickup,
            posicion_delivery=posicion_delivery
        )
        tareas.append(tarea)
    
    # Crear entorno (sin ruido, sin fallos de comunicación)
    entorno = Entorno(
        dimensiones=dimensiones,
        robots=robots,
        tareas=tareas,
        ruido_percepcion=0.0,
        probabilidad_fallo_comunicacion=0.0
    )
    
    return entorno


def ejecutar_experimento(num_robots: int = 5,
                         num_tareas: int = 10,
                         semilla: int = None,
                         verbose: bool = True) -> Dict:
    """
    Ejecuta un experimento del Caso de uso 1.
    
    Args:
        num_robots: Número de robots
        num_tareas: Número de tareas
        semilla: Semilla aleatoria
        verbose: Si True, imprime información detallada
        
    Returns:
        Diccionario con métricas del experimento
    """
    # Crear entorno
    entorno = crear_entorno_experimento(
        num_robots=num_robots,
        num_tareas=num_tareas,
        semilla=semilla
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CASO DE USO 1: Algoritmo Greedy Distribuido")
        print(f"{'='*60}")
        print(f"Configuración:")
        print(f"  - Robots: {num_robots}")
        print(f"  - Tareas: {num_tareas}")
        print(f"  - Semilla: {semilla}")
        print(f"{'='*60}\n")
    
    # Crear y ejecutar algoritmo
    algoritmo = AlgoritmoGreedyDistribuido(entorno)
    rondas = algoritmo.ejecutar(max_rondas=1000)
    
    # Calcular métricas
    metricas = calcular_metricas(entorno)
    metricas['rondas_ejecutadas'] = rondas
    
    if verbose:
        imprimir_metricas(metricas)
        print(f"Tareas completadas por robot:")
        for robot in entorno.robots:
            print(f"  Robot {robot.id_robot}: {robot.tareas_completadas_count} tareas")
    
    return metricas


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
        f.write("CASO DE USO 1: Algoritmo Greedy Distribuido\n")
        f.write("="*80 + "\n")
        f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for exp_nombre, exp_resultados in resultados.items():
            f.write("\n" + "="*80 + "\n")
            f.write(f"{exp_nombre.upper().replace('_', ' ')}\n")
            f.write("="*80 + "\n\n")
            
            # Calcular estadísticas
            if exp_resultados:
                promedios = {}
                desviaciones = {}
                
                for key in exp_resultados[0].keys():
                    valores = [r[key] for r in exp_resultados]
                    promedios[key] = np.mean(valores)
                    desviaciones[key] = np.std(valores)
                
                f.write("Resultados promedio (10 ejecuciones):\n")
                f.write("-"*80 + "\n")
                for key in sorted(promedios.keys()):
                    f.write(f"  {key:30s}: {promedios[key]:10.2f} ± {desviaciones[key]:.2f}\n")
                
                f.write("\nResultados individuales:\n")
                f.write("-"*80 + "\n")
                for i, resultado in enumerate(exp_resultados):
                    f.write(f"\nEjecución {i+1}:\n")
                    for key in sorted(resultado.keys()):
                        f.write(f"  {key:30s}: {resultado[key]:10.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Resultados guardados en: {ruta_archivo}")


def ejecutar_caso_uso_1():
    """Ejecuta todos los experimentos del Caso de uso 1."""
    resultados = {}
    
    # Experimento 1.1: Configuración base (5 robots, 10 tareas)
    print("\n" + "="*60)
    print("EXPERIMENTO 1.1: Configuración base (5 robots, 10 tareas)")
    print("="*60)
    resultados_exp_1_1 = []
    for semilla in range(10):
        metricas = ejecutar_experimento(num_robots=5, num_tareas=10, semilla=semilla, verbose=False)
        resultados_exp_1_1.append(metricas)
    
    resultados['experimento_1_1'] = resultados_exp_1_1
    
    # Calcular promedios
    print("\nResultados promedio (10 ejecuciones):")
    promedios = {}
    for key in resultados_exp_1_1[0].keys():
        valores = [r[key] for r in resultados_exp_1_1]
        promedios[key] = np.mean(valores)
        print(f"  {key}: {promedios[key]:.2f}")
    
    # Experimento 1.2: Más tareas (5 robots, 15 tareas)
    print("\n" + "="*60)
    print("EXPERIMENTO 1.2: Más tareas (5 robots, 15 tareas)")
    print("="*60)
    resultados_exp_1_2 = []
    for semilla in range(10):
        metricas = ejecutar_experimento(num_robots=5, num_tareas=15, semilla=semilla, verbose=False)
        resultados_exp_1_2.append(metricas)
    
    resultados['experimento_1_2'] = resultados_exp_1_2
    
    # Calcular promedios
    print("\nResultados promedio (10 ejecuciones):")
    promedios = {}
    for key in resultados_exp_1_2[0].keys():
        valores = [r[key] for r in resultados_exp_1_2]
        promedios[key] = np.mean(valores)
        print(f"  {key}: {promedios[key]:.2f}")
    
    # Experimento 1.3: Más robots (8 robots, 10 tareas)
    print("\n" + "="*60)
    print("EXPERIMENTO 1.3: Más robots (8 robots, 10 tareas)")
    print("="*60)
    resultados_exp_1_3 = []
    for semilla in range(10):
        metricas = ejecutar_experimento(num_robots=8, num_tareas=10, semilla=semilla, verbose=False)
        resultados_exp_1_3.append(metricas)
    
    resultados['experimento_1_3'] = resultados_exp_1_3
    
    # Calcular promedios
    print("\nResultados promedio (10 ejecuciones):")
    promedios = {}
    for key in resultados_exp_1_3[0].keys():
        valores = [r[key] for r in resultados_exp_1_3]
        promedios[key] = np.mean(valores)
        print(f"  {key}: {promedios[key]:.2f}")
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    guardar_resultados(resultados, f"caso_uso_1_{timestamp}.txt")
    
    return resultados


if __name__ == "__main__":
    ejecutar_caso_uso_1()
