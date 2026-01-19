"""
Clase para representar una tarea de recogida y entrega.
"""
from enum import Enum
from typing import Tuple, Optional


class EstadoTarea(Enum):
    """Estados posibles de una tarea."""
    PENDIENTE = "pendiente"
    ASIGNADA = "asignada"
    EN_PROGRESO = "en_progreso"
    COMPLETADA = "completada"


class Tarea:
    """Representa una tarea de recogida y entrega."""
    
    def __init__(self, 
                 id_tarea: int,
                 posicion_pickup: Tuple[int, int],
                 posicion_delivery: Tuple[int, int],
                 posicion_real_pickup: Optional[Tuple[int, int]] = None,
                 posicion_real_delivery: Optional[Tuple[int, int]] = None):
        """
        Inicializa una tarea.
        
        Args:
            id_tarea: Identificador único de la tarea
            posicion_pickup: Posición de recogida (puede ser estimada)
            posicion_delivery: Posición de entrega (puede ser estimada)
            posicion_real_pickup: Posición real de recogida (si es None, usa posicion_pickup)
            posicion_real_delivery: Posición real de entrega (si es None, usa posicion_delivery)
        """
        self.id_tarea = id_tarea
        self.posicion_pickup = posicion_pickup
        self.posicion_delivery = posicion_delivery
        self.posicion_real_pickup = posicion_real_pickup if posicion_real_pickup else posicion_pickup
        self.posicion_real_delivery = posicion_real_delivery if posicion_real_delivery else posicion_delivery
        self.estado = EstadoTarea.PENDIENTE
        self.robot_asignado: Optional[int] = None
        
        # Para el algoritmo con incertidumbre
        self.estimaciones: dict = {}  # {robot_id: (pos_estimada, sigma, confianza)}
        self.sigma_pickup = 0.0
        self.sigma_delivery = 0.0
        self.confianza_total = 0.0
    
    def distancia_manhattan(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcula la distancia de Manhattan entre dos posiciones."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def coste_total(self, posicion_robot: Tuple[int, int]) -> float:
        """
        Calcula el coste total de realizar esta tarea desde una posición.
        
        Args:
            posicion_robot: Posición actual del robot
            
        Returns:
            Distancia total a recorrer
        """
        dist_pickup = self.distancia_manhattan(posicion_robot, self.posicion_pickup)
        dist_delivery = self.distancia_manhattan(self.posicion_pickup, self.posicion_delivery)
        return dist_pickup + dist_delivery
    
    def __repr__(self):
        return f"Tarea(id={self.id_tarea}, estado={self.estado.value}, pickup={self.posicion_pickup}, delivery={self.posicion_delivery})"
