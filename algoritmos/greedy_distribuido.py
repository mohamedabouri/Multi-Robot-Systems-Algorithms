"""
Algoritmo 1: Asignación Greedy Distribuida
Algoritmo distribuido para asignación de tareas en entornos deterministas.
"""
from typing import List, Dict, Set
from utils.tarea import EstadoTarea
from utils.entorno import Entorno


class AlgoritmoGreedyDistribuido:
    """
    Algoritmo greedy distribuido para asignación de tareas.
    Opera en rondas donde los robots intercambian información y realizan asignaciones.
    """
    
    def __init__(self, entorno: Entorno):
        """
        Inicializa el algoritmo.
        
        Args:
            entorno: Entorno de simulación
        """
        self.entorno = entorno
        self.intenciones: Dict[int, Set[int]] = {}  # {robot_id: {tarea_ids}}
        self.asignaciones_confirmadas: Dict[int, int] = {}  # {tarea_id: robot_id}
    
    def inicializar(self):
        """Inicializa el estado de todos los robots."""
        for robot in self.entorno.robots:
            # Detectar tareas iniciales
            tareas_detectadas = self.entorno.detectar_tareas(robot.id_robot)
            robot.tareas_conocidas = {t.id_tarea for t in tareas_detectadas}
            robot.tareas_asignadas = set()
            robot.tareas_completadas = set()
            robot.carga_actual = 0
            self.intenciones[robot.id_robot] = set()
    
    def ejecutar_ronda(self) -> bool:
        """
        Ejecuta una ronda del algoritmo.
        
        Returns:
            True si hay progreso, False si no hay cambios
        """
        # Paso 1: Comunicación - intercambiar información con vecinos
        self._comunicar_con_vecinos()
        
        # Paso 2: Actualizar conocimiento local
        self._actualizar_conocimiento()
        
        # Paso 3: Selección de tareas
        self._seleccionar_tareas()
        
        # Paso 4: Resolución de conflictos
        self._resolver_conflictos()
        
        # Paso 5: Ejecución de tareas asignadas
        progreso = self._ejecutar_tareas()
        
        return progreso
    
    def _comunicar_con_vecinos(self):
        """Los robots intercambian información con sus vecinos."""
        # En un sistema real, aquí se enviarían mensajes
        # Por simplicidad, simulamos que todos los robots conocen la información
        # de sus vecinos después de la comunicación
        pass  # La comunicación se maneja implícitamente en _actualizar_conocimiento
    
    def _actualizar_conocimiento(self):
        """Cada robot actualiza su conocimiento basándose en la información de vecinos."""
        for robot in self.entorno.robots:
            vecinos = self.entorno.obtener_vecinos_comunicacion(robot.id_robot)
            
            # Actualizar tareas conocidas
            for vecino in vecinos:
                robot.tareas_conocidas.update(vecino.tareas_conocidas)
                robot.tareas_conocidas.update(vecino.tareas_asignadas)
            
            # Detectar nuevas tareas en rango
            tareas_detectadas = self.entorno.detectar_tareas(robot.id_robot)
            robot.tareas_conocidas.update({t.id_tarea for t in tareas_detectadas})
    
    def _seleccionar_tareas(self):
        """Cada robot selecciona la mejor tarea disponible."""
        self.intenciones = {}
        
        for robot in self.entorno.robots:
            self.intenciones[robot.id_robot] = set()
            
            if not robot.tiene_capacidad():
                continue
            
            # Obtener tareas disponibles (no asignadas, no completadas)
            tareas_disponibles = []
            for tarea_id in robot.tareas_conocidas:
                if tarea_id in robot.tareas_completadas:
                    continue
                
                tarea = self.entorno.tareas.get(tarea_id)
                if not tarea:
                    continue
                
                # Verificar si la tarea está asignada a otro robot
                if tarea.estado == EstadoTarea.ASIGNADA and tarea.robot_asignado != robot.id_robot:
                    continue
                
                if tarea.estado == EstadoTarea.PENDIENTE or tarea.estado == EstadoTarea.ASIGNADA:
                    tareas_disponibles.append(tarea)
            
            if not tareas_disponibles:
                continue
            
            # Calcular coste para cada tarea disponible
            costes = {}
            for tarea in tareas_disponibles:
                coste = tarea.coste_total(robot.posicion)
                costes[tarea.id_tarea] = coste
            
            # Seleccionar la tarea con menor coste
            if costes:
                mejor_tarea_id = min(costes.keys(), key=lambda tid: costes[tid])
                self.intenciones[robot.id_robot].add(mejor_tarea_id)
    
    def _resolver_conflictos(self):
        """Resuelve conflictos cuando múltiples robots quieren la misma tarea."""
        # Agrupar intenciones por tarea
        tareas_solicitadas: Dict[int, List[int]] = {}
        for robot_id, tareas in self.intenciones.items():
            for tarea_id in tareas:
                if tarea_id not in tareas_solicitadas:
                    tareas_solicitadas[tarea_id] = []
                tareas_solicitadas[tarea_id].append(robot_id)
        
        # Para cada tarea con múltiples solicitantes, asignar al robot con mayor utilidad
        for tarea_id, robots_solicitantes in tareas_solicitadas.items():
            if len(robots_solicitantes) == 1:
                # Sin conflicto
                robot_id = robots_solicitantes[0]
                self._asignar_tarea(robot_id, tarea_id)
            else:
                # Hay conflicto - calcular utilidad para cada robot
                utilidades = {}
                tarea = self.entorno.tareas[tarea_id]
                
                for robot_id in robots_solicitantes:
                    robot = self.entorno.robots[robot_id]
                    coste = tarea.coste_total(robot.posicion)
                    utilidad = 1.0 / coste if coste > 0 else float('inf')
                    utilidades[robot_id] = utilidad
                
                # Asignar al robot con mayor utilidad
                robot_ganador = max(utilidades.keys(), key=lambda rid: utilidades[rid])
                self._asignar_tarea(robot_ganador, tarea_id)
                
                # Los demás robots eliminan esta tarea de sus candidatos
                for robot_id in robots_solicitantes:
                    if robot_id != robot_ganador:
                        self.intenciones[robot_id].discard(tarea_id)
    
    def _asignar_tarea(self, robot_id: int, tarea_id: int):
        """Asigna una tarea a un robot."""
        robot = self.entorno.robots[robot_id]
        tarea = self.entorno.tareas[tarea_id]
        
        robot.asignar_tarea(tarea_id)
        tarea.estado = EstadoTarea.ASIGNADA
        tarea.robot_asignado = robot_id
        self.asignaciones_confirmadas[tarea_id] = robot_id
    
    def _ejecutar_tareas(self) -> bool:
        """
        Ejecuta las tareas asignadas.
        
        Returns:
            True si hubo progreso, False en caso contrario
        """
        progreso = False
        
        for robot in self.entorno.robots:
            if not robot.tareas_asignadas:
                continue
            
            # Obtener la tarea asignada más prioritaria (la primera en la lista)
            tarea_id = next(iter(robot.tareas_asignadas))
            tarea = self.entorno.tareas[tarea_id]
            
            if tarea.estado == EstadoTarea.ASIGNADA:
                # Mover hacia el punto de recogida
                if robot.posicion != tarea.posicion_pickup:
                    robot.mover_hacia(tarea.posicion_pickup)
                    progreso = True
                else:
                    # Llegó al punto de recogida
                    robot.recoger_objeto()
                    tarea.estado = EstadoTarea.EN_PROGRESO
                    progreso = True
            
            elif tarea.estado == EstadoTarea.EN_PROGRESO:
                # Mover hacia el punto de entrega
                if robot.posicion != tarea.posicion_delivery:
                    robot.mover_hacia(tarea.posicion_delivery)
                    progreso = True
                else:
                    # Llegó al punto de entrega - completar tarea
                    robot.completar_tarea(tarea_id)
                    tarea.estado = EstadoTarea.COMPLETADA
                    tarea.robot_asignado = None
                    progreso = True
        
        return progreso
    
    def ejecutar(self, max_rondas: int = 1000) -> int:
        """
        Ejecuta el algoritmo hasta completar todas las tareas o alcanzar el máximo de rondas.
        
        Args:
            max_rondas: Número máximo de rondas
            
        Returns:
            Número de rondas ejecutadas
        """
        self.inicializar()
        
        rondas_sin_progreso = 0
        max_sin_progreso = 10
        
        for ronda in range(max_rondas):
            self.entorno.ronda_actual = ronda
            progreso = self.ejecutar_ronda()
            
            if progreso:
                rondas_sin_progreso = 0
            else:
                rondas_sin_progreso += 1
                if rondas_sin_progreso >= max_sin_progreso:
                    break
            
            if self.entorno.todas_tareas_completadas():
                break
            
            self.entorno.avanzar_ronda()
        
        return self.entorno.ronda_actual
