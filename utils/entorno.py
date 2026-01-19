"""
Clase para representar el entorno de simulación.
"""
from typing import List, Tuple, Dict
import numpy as np
from utils.robot import Robot
from utils.tarea import Tarea, EstadoTarea


class Entorno:
    """Representa el entorno de simulación para el sistema multi-robot."""
    
    def __init__(self,
                 dimensiones: Tuple[int, int] = (10, 10),
                 robots: List[Robot] = None,
                 tareas: List[Tarea] = None,
                 ruido_percepcion: float = 0.0,
                 probabilidad_fallo_comunicacion: float = 0.0):
        """
        Inicializa el entorno.
        
        Args:
            dimensiones: Dimensiones del espacio (ancho, alto)
            robots: Lista de robots en el entorno
            tareas: Lista de tareas en el entorno
            ruido_percepcion: Desviación estándar del ruido gaussiano en la percepción
            probabilidad_fallo_comunicacion: Probabilidad de que un mensaje falle
        """
        self.dimensiones = dimensiones
        self.robots = robots if robots else []
        self.tareas = {t.id_tarea: t for t in (tareas if tareas else [])}
        self.ruido_percepcion = ruido_percepcion
        self.probabilidad_fallo_comunicacion = probabilidad_fallo_comunicacion
        
        # Estadísticas
        self.ronda_actual = 0
        self.historial: List[Dict] = []
    
    def obtener_vecinos_comunicacion(self, robot_id: int) -> List[Robot]:
        """
        Obtiene los robots vecinos dentro del rango de comunicación.
        
        Args:
            robot_id: ID del robot
            
        Returns:
            Lista de robots vecinos (excluyendo el propio robot)
        """
        robot = self.robots[robot_id]
        vecinos = []
        for otro_robot in self.robots:
            if otro_robot.id_robot != robot_id and robot.puede_comunicarse_con(otro_robot):
                # Simular fallo de comunicación
                if np.random.random() > self.probabilidad_fallo_comunicacion:
                    vecinos.append(otro_robot)
        return vecinos
    
    def detectar_tareas(self, robot_id: int) -> List[Tarea]:
        """
        Detecta tareas dentro del rango de percepción del robot.
        
        Args:
            robot_id: ID del robot
            
        Returns:
            Lista de tareas detectadas (con ruido si aplica)
        """
        robot = self.robots[robot_id]
        tareas_detectadas = []
        
        for tarea in self.tareas.values():
            if robot.puede_detectar_tarea(tarea):
                # Crear una copia de la tarea con posible ruido
                if self.ruido_percepcion > 0:
                    # Añadir ruido gaussiano
                    ruido_x = np.random.normal(0, self.ruido_percepcion)
                    ruido_y = np.random.normal(0, self.ruido_percepcion)
                    
                    pickup_estimado = (
                        int(round(tarea.posicion_real_pickup[0] + ruido_x)),
                        int(round(tarea.posicion_real_pickup[1] + ruido_y))
                    )
                    delivery_estimado = (
                        int(round(tarea.posicion_real_delivery[0] + ruido_x)),
                        int(round(tarea.posicion_real_delivery[1] + ruido_y))
                    )
                    
                    # Crear tarea con estimación
                    tarea_detectada = Tarea(
                        id_tarea=tarea.id_tarea,
                        posicion_pickup=pickup_estimado,
                        posicion_delivery=delivery_estimado,
                        posicion_real_pickup=tarea.posicion_real_pickup,
                        posicion_real_delivery=tarea.posicion_real_delivery
                    )
                    tarea_detectada.estado = tarea.estado
                    tareas_detectadas.append(tarea_detectada)
                else:
                    tareas_detectadas.append(tarea)
        
        return tareas_detectadas
    
    def todas_tareas_completadas(self) -> bool:
        """Verifica si todas las tareas han sido completadas."""
        return all(t.estado == EstadoTarea.COMPLETADA for t in self.tareas.values())
    
    def obtener_estadisticas(self) -> Dict:
        """Obtiene estadísticas del estado actual del entorno."""
        return {
            'ronda': self.ronda_actual,
            'tareas_completadas': sum(1 for t in self.tareas.values() if t.estado == EstadoTarea.COMPLETADA),
            'tareas_totales': len(self.tareas),
            'distancia_total': sum(r.distancia_recorrida for r in self.robots),
            'tareas_por_robot': {r.id_robot: r.tareas_completadas_count for r in self.robots}
        }
    
    def avanzar_ronda(self):
        """Avanza una ronda en la simulación."""
        self.ronda_actual += 1
        stats = self.obtener_estadisticas()
        self.historial.append(stats)
