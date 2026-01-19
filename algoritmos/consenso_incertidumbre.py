"""
Algoritmo 2: Asignación basada en Consenso Distribuido con Incertidumbre
Algoritmo distribuido para asignación de tareas en entornos con incertidumbre.
"""
from typing import Dict
from utils.tarea import EstadoTarea
from utils.entorno import Entorno


class AlgoritmoConsensoIncertidumbre:
    """
    Algoritmo de consenso distribuido para asignación de tareas con incertidumbre.
    Utiliza fusión de información y consenso distribuido para manejar incertidumbre.
    """
    
    def __init__(self, 
                 entorno: Entorno,
                 epsilon: float = 0.2,
                 lambda_penalizacion: float = 0.3,
                 iteraciones_consenso: int = 5,
                 sigma_umbral: float = 2.0):
        """
        Inicializa el algoritmo.
        
        Args:
            entorno: Entorno de simulación
            epsilon: Parámetro de paso para el consenso
            lambda_penalizacion: Parámetro de penalización por incertidumbre
            iteraciones_consenso: Número de iteraciones de consenso por ronda
            sigma_umbral: Umbral de incertidumbre para liberar tareas
        """
        self.entorno = entorno
        self.epsilon = epsilon
        self.lambda_penalizacion = lambda_penalizacion
        self.iteraciones_consenso = iteraciones_consenso
        self.sigma_umbral = sigma_umbral
        
        # Estado del algoritmo
        self.estimaciones_tareas: Dict[int, Dict] = {}  # {tarea_id: {robot_id: (pos, sigma, confianza)}}
        self.asignaciones_probabilisticas: Dict[int, Dict[int, float]] = {}  # {tarea_id: {robot_id: prob}}
    
    def inicializar(self):
        """Inicializa el estado de todos los robots."""
        for robot in self.entorno.robots:
            robot.tareas_conocidas = set()
            robot.tareas_asignadas = set()
            robot.tareas_completadas = set()
            robot.carga_actual = 0
            robot.confianza_tareas = {}
            robot.asignaciones_propuestas = {}
        
        # Inicializar estimaciones
        self.estimaciones_tareas = {}
        self.asignaciones_probabilisticas = {}
    
    def ejecutar_ronda(self) -> bool:
        """
        Ejecuta una ronda del algoritmo.
        
        Returns:
            True si hay progreso, False si no hay cambios
        """
        # Paso 1: Percepción y actualización local
        self._percepcion_local()
        
        # Paso 2: Comunicación y fusión de información
        self._fusionar_informacion()
        
        # Paso 3: Consenso sobre asignaciones
        self._consenso_asignaciones()
        
        # Paso 4: Selección de tareas con incertidumbre
        self._seleccionar_tareas_incertidumbre()
        
        # Paso 5: Ejecución adaptativa
        progreso = self._ejecutar_tareas_adaptativo()
        
        return progreso
    
    def _percepcion_local(self):
        """Cada robot detecta tareas y actualiza sus estimaciones locales."""
        for robot in self.entorno.robots:
            tareas_detectadas = self.entorno.detectar_tareas(robot.id_robot)
            
            for tarea_detectada in tareas_detectadas:
                tarea_id = tarea_detectada.id_tarea
                robot.tareas_conocidas.add(tarea_id)
                
                # Actualizar estimación local usando filtrado bayesiano simple
                if tarea_id not in self.estimaciones_tareas:
                    self.estimaciones_tareas[tarea_id] = {}
                
                # Calcular sigma basado en el ruido de percepción
                sigma = self.entorno.ruido_percepcion
                
                # Actualizar confianza (aumenta con más observaciones)
                if tarea_id in robot.confianza_tareas:
                    robot.confianza_tareas[tarea_id] += 0.1
                else:
                    robot.confianza_tareas[tarea_id] = 1.0
                
                # Guardar estimación
                self.estimaciones_tareas[tarea_id][robot.id_robot] = {
                    'pickup': tarea_detectada.posicion_pickup,
                    'delivery': tarea_detectada.posicion_delivery,
                    'sigma': sigma,
                    'confianza': robot.confianza_tareas[tarea_id]
                }
    
    def _fusionar_informacion(self):
        """Fusiona estimaciones de múltiples robots usando promedio ponderado por confianza."""
        for tarea_id, estimaciones_robots in self.estimaciones_tareas.items():
            if not estimaciones_robots:
                continue
            
            # Calcular posiciones fusionadas usando promedio ponderado
            confianza_total = sum(est['confianza'] for est in estimaciones_robots.values())
            
            if confianza_total == 0:
                continue
            
            # Fusionar pickup
            pickup_fusionado_x = sum(est['pickup'][0] * est['confianza'] 
                                     for est in estimaciones_robots.values()) / confianza_total
            pickup_fusionado_y = sum(est['pickup'][1] * est['confianza'] 
                                     for est in estimaciones_robots.values()) / confianza_total
            
            # Fusionar delivery
            delivery_fusionado_x = sum(est['delivery'][0] * est['confianza'] 
                                       for est in estimaciones_robots.values()) / confianza_total
            delivery_fusionado_y = sum(est['delivery'][1] * est['confianza'] 
                                        for est in estimaciones_robots.values()) / confianza_total
            
            # Actualizar la tarea con las estimaciones fusionadas
            tarea = self.entorno.tareas.get(tarea_id)
            if tarea:
                tarea.posicion_pickup = (int(round(pickup_fusionado_x)), int(round(pickup_fusionado_y)))
                tarea.posicion_delivery = (int(round(delivery_fusionado_x)), int(round(delivery_fusionado_y)))
                
                # Calcular sigma fusionado (promedio ponderado)
                sigma_fusionado = sum(est['sigma'] * est['confianza'] 
                                     for est in estimaciones_robots.values()) / confianza_total
                tarea.sigma_pickup = sigma_fusionado
                tarea.sigma_delivery = sigma_fusionado
                tarea.confianza_total = confianza_total
    
    def _consenso_asignaciones(self):
        """Realiza consenso distribuido sobre las asignaciones propuestas."""
        # Inicializar asignaciones probabilísticas si no existen
        for tarea_id in self.estimaciones_tareas.keys():
            if tarea_id not in self.asignaciones_probabilisticas:
                self.asignaciones_probabilisticas[tarea_id] = {}
                for robot in self.entorno.robots:
                    self.asignaciones_probabilisticas[tarea_id][robot.id_robot] = 0.0
        
        # Iteraciones de consenso
        for _ in range(self.iteraciones_consenso):
            nuevas_asignaciones = {}
            
            for tarea_id, asignaciones_actuales in self.asignaciones_probabilisticas.items():
                nuevas_asignaciones[tarea_id] = {}
                
                for robot in self.entorno.robots:
                    robot_id = robot.id_robot
                    valor_actual = asignaciones_actuales.get(robot_id, 0.0)
                    
                    # Obtener valores de vecinos
                    vecinos = self.entorno.obtener_vecinos_comunicacion(robot_id)
                    suma_diferencias = 0.0
                    
                    for vecino in vecinos:
                        valor_vecino = asignaciones_actuales.get(vecino.id_robot, 0.0)
                        suma_diferencias += (valor_vecino - valor_actual)
                    
                    # Actualizar usando consenso de promedio
                    nuevo_valor = valor_actual + self.epsilon * suma_diferencias
                    nuevas_asignaciones[tarea_id][robot_id] = max(0.0, min(1.0, nuevo_valor))
            
            self.asignaciones_probabilisticas = nuevas_asignaciones
    
    def _seleccionar_tareas_incertidumbre(self):
        """Selecciona tareas considerando incertidumbre."""
        for robot in self.entorno.robots:
            if not robot.tiene_capacidad():
                continue
            
            # Obtener tareas disponibles
            tareas_disponibles = []
            for tarea_id in robot.tareas_conocidas:
                if tarea_id in robot.tareas_completadas:
                    continue
                
                tarea = self.entorno.tareas.get(tarea_id)
                if not tarea:
                    continue
                
                if tarea.estado == EstadoTarea.ASIGNADA and tarea.robot_asignado != robot.id_robot:
                    continue
                
                if tarea.estado == EstadoTarea.PENDIENTE or tarea.estado == EstadoTarea.ASIGNADA:
                    tareas_disponibles.append(tarea)
            
            if not tareas_disponibles:
                continue
            
            # Calcular utilidad esperada ajustada por incertidumbre
            utilidades = {}
            for tarea in tareas_disponibles:
                # Coste esperado (simplificado - usando distancia a estimación)
                coste_esperado = tarea.coste_total(robot.posicion)
                
                # Penalización por incertidumbre
                sigma_total = (tarea.sigma_pickup + tarea.sigma_delivery) / 2
                penalizacion = self.lambda_penalizacion * sigma_total
                
                # Utilidad ajustada
                utilidad = (1.0 / coste_esperado if coste_esperado > 0 else float('inf')) - penalizacion
                utilidades[tarea.id_tarea] = utilidad
            
            # Seleccionar tarea con mayor utilidad
            if utilidades:
                mejor_tarea_id = max(utilidades.keys(), key=lambda tid: utilidades[tid])
                
                # Verificar si hay conflicto
                tarea = self.entorno.tareas[mejor_tarea_id]
                if tarea.estado == EstadoTarea.PENDIENTE:
                    # Asignar directamente si no hay conflicto
                    self._asignar_tarea(robot.id_robot, mejor_tarea_id)
                elif tarea.estado == EstadoTarea.ASIGNADA and tarea.robot_asignado == robot.id_robot:
                    # Ya está asignada a este robot
                    pass
    
    def _asignar_tarea(self, robot_id: int, tarea_id: int):
        """Asigna una tarea a un robot."""
        robot = self.entorno.robots[robot_id]
        tarea = self.entorno.tareas[tarea_id]
        
        # Verificar incertidumbre
        sigma_total = (tarea.sigma_pickup + tarea.sigma_delivery) / 2
        if sigma_total > self.sigma_umbral:
            # Incertidumbre demasiado alta - no asignar
            return
        
        robot.asignar_tarea(tarea_id)
        tarea.estado = EstadoTarea.ASIGNADA
        tarea.robot_asignado = robot_id
    
    def _ejecutar_tareas_adaptativo(self) -> bool:
        """
        Ejecuta las tareas asignadas con actualización adaptativa.
        
        Returns:
            True si hubo progreso, False en caso contrario
        """
        progreso = False
        
        for robot in self.entorno.robots:
            if not robot.tareas_asignadas:
                continue
            
            tarea_id = next(iter(robot.tareas_asignadas))
            tarea = self.entorno.tareas[tarea_id]
            
            # Verificar si la incertidumbre ha crecido demasiado
            sigma_total = (tarea.sigma_pickup + tarea.sigma_delivery) / 2
            if sigma_total > self.sigma_umbral and tarea.estado == EstadoTarea.ASIGNADA:
                # Liberar tarea
                robot.tareas_asignadas.discard(tarea_id)
                tarea.estado = EstadoTarea.PENDIENTE
                tarea.robot_asignado = None
                progreso = True
                continue
            
            if tarea.estado == EstadoTarea.ASIGNADA:
                # Mover hacia el punto de recogida (usando estimación)
                if robot.posicion != tarea.posicion_pickup:
                    robot.mover_hacia(tarea.posicion_pickup)
                    progreso = True
                else:
                    # Verificar si realmente llegó (comparar con posición real)
                    if robot.posicion == tarea.posicion_real_pickup:
                        robot.recoger_objeto()
                        tarea.estado = EstadoTarea.EN_PROGRESO
                        progreso = True
                    else:
                        # La estimación era incorrecta - actualizar
                        tarea.posicion_pickup = tarea.posicion_real_pickup
                        progreso = True
            
            elif tarea.estado == EstadoTarea.EN_PROGRESO:
                # Mover hacia el punto de entrega
                if robot.posicion != tarea.posicion_delivery:
                    robot.mover_hacia(tarea.posicion_delivery)
                    progreso = True
                else:
                    # Verificar si realmente llegó
                    if robot.posicion == tarea.posicion_real_delivery:
                        robot.completar_tarea(tarea_id)
                        tarea.estado = EstadoTarea.COMPLETADA
                        tarea.robot_asignado = None
                        progreso = True
                    else:
                        # La estimación era incorrecta - actualizar
                        tarea.posicion_delivery = tarea.posicion_real_delivery
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
