"""
Clase para representar un robot en el sistema multi-robot.
"""
from typing import Tuple, Set
from utils.tarea import Tarea


class Robot:
    """Representa un robot en el sistema multi-robot."""
    
    def __init__(self,
                 id_robot: int,
                 posicion_inicial: Tuple[int, int],
                 capacidad: int = 1,
                 rango_percepcion: int = 3,
                 rango_comunicacion: int = 5):
        """
        Inicializa un robot.
        
        Args:
            id_robot: Identificador único del robot
            posicion_inicial: Posición inicial del robot
            capacidad: Número máximo de objetos que puede transportar simultáneamente
            rango_percepcion: Rango de percepción (distancia de Manhattan)
            rango_comunicacion: Rango de comunicación (distancia de Manhattan)
        """
        self.id_robot = id_robot
        self.posicion = posicion_inicial
        self.capacidad = capacidad
        self.carga_actual = 0
        self.rango_percepcion = rango_percepcion
        self.rango_comunicacion = rango_comunicacion
        
        # Conjuntos de tareas
        self.tareas_conocidas: Set[int] = set()  # IDs de tareas conocidas
        self.tareas_asignadas: Set[int] = set()  # IDs de tareas asignadas
        self.tareas_completadas: Set[int] = set()  # IDs de tareas completadas
        
        # Para el algoritmo con incertidumbre
        self.confianza_tareas: dict = {}  # {tarea_id: confianza}
        self.asignaciones_propuestas: dict = {}  # {tarea_id: probabilidad_asignacion}
        
        # Estadísticas
        self.distancia_recorrida = 0
        self.tareas_completadas_count = 0
    
    def distancia_manhattan(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcula la distancia de Manhattan entre dos posiciones."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def puede_detectar_tarea(self, tarea: Tarea) -> bool:
        """Verifica si el robot puede detectar una tarea dentro de su rango de percepción."""
        dist = self.distancia_manhattan(self.posicion, tarea.posicion_pickup)
        return dist <= self.rango_percepcion
    
    def puede_comunicarse_con(self, otro_robot: 'Robot') -> bool:
        """Verifica si este robot puede comunicarse con otro robot."""
        dist = self.distancia_manhattan(self.posicion, otro_robot.posicion)
        return dist <= self.rango_comunicacion
    
    def tiene_capacidad(self) -> bool:
        """Verifica si el robot tiene capacidad disponible."""
        return self.carga_actual < self.capacidad
    
    def mover_hacia(self, destino: Tuple[int, int]) -> int:
        """
        Mueve el robot un paso hacia el destino.
        
        Args:
            destino: Posición objetivo
            
        Returns:
            Distancia recorrida en este paso
        """
        if self.posicion == destino:
            return 0
        
        # Movimiento en un paso (distancia de Manhattan = 1)
        dx = 0 if self.posicion[0] == destino[0] else (1 if destino[0] > self.posicion[0] else -1)
        dy = 0 if self.posicion[1] == destino[1] else (1 if destino[1] > self.posicion[1] else -1)
        
        self.posicion = (self.posicion[0] + dx, self.posicion[1] + dy)
        self.distancia_recorrida += 1
        return 1
    
    def asignar_tarea(self, tarea_id: int):
        """Asigna una tarea al robot."""
        self.tareas_asignadas.add(tarea_id)
    
    def completar_tarea(self, tarea_id: int):
        """Marca una tarea como completada."""
        if tarea_id in self.tareas_asignadas:
            self.tareas_asignadas.remove(tarea_id)
        self.tareas_completadas.add(tarea_id)
        self.tareas_completadas_count += 1
        self.carga_actual = max(0, self.carga_actual - 1)
    
    def recoger_objeto(self):
        """El robot recoge un objeto."""
        if self.carga_actual < self.capacidad:
            self.carga_actual += 1
    
    def __repr__(self):
        return f"Robot(id={self.id_robot}, pos={self.posicion}, carga={self.carga_actual}/{self.capacidad})"
