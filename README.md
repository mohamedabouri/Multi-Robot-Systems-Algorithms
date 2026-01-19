# Sistema Multi-Robot: Algoritmos de Asignación de Tareas Distribuidos

Este proyecto implementa dos algoritmos distribuidos de asignación de tareas para sistemas multi-robot, junto con casos de uso para su evaluación.

## Estructura del Proyecto

```
ROB-IA/
├── algoritmos/
│   ├── __init__.py
│   ├── greedy_distribuido.py          # Algoritmo 1: Asignación Greedy Distribuida
│   └── consenso_incertidumbre.py      # Algoritmo 2: Consenso Distribuido con Incertidumbre
├── casos_uso/
│   ├── __init__.py
│   ├── caso_uso_1.py                  # Caso de uso 1: Greedy sin incertidumbre
│   ├── caso_uso_2.py                  # Caso de uso 2: Consenso con incertidumbre
│   └── caso_uso_3.py                  # Caso de uso 3: Comparación directa
├── utils/
│   ├── __init__.py
│   ├── robot.py                       # Clase Robot
│   ├── tarea.py                       # Clase Tarea
│   ├── entorno.py                     # Clase Entorno
│   └── metricas.py                    # Funciones de cálculo de métricas
└── README.md
```

## Requisitos

- Python 3.7+
- NumPy

## Instalación

```bash
pip install numpy
```

## Algoritmos Implementados

### Algoritmo 1: Asignación Greedy Distribuida

Algoritmo distribuido para entornos deterministas que:
- Opera en rondas de comunicación y asignación
- Selecciona tareas basándose en coste local mínimo
- Resuelve conflictos mediante utilidad
- Es eficiente computacionalmente

**Uso:**
```python
from algoritmos.greedy_distribuido import AlgoritmoGreedyDistribuido
from utils.entorno import Entorno

entorno = Entorno(...)
algoritmo = AlgoritmoGreedyDistribuido(entorno)
rondas = algoritmo.ejecutar(max_rondas=1000)
```

### Algoritmo 2: Consenso Distribuido con Incertidumbre

Algoritmo distribuido para entornos con incertidumbre que:
- Utiliza fusión de información para reducir incertidumbre
- Aplica consenso distribuido para coordinar asignaciones
- Maneja ruido en percepción y fallos de comunicación
- Es más robusto pero computacionalmente más costoso

**Uso:**
```python
from algoritmos.consenso_incertidumbre import AlgoritmoConsensoIncertidumbre
from utils.entorno import Entorno

entorno = Entorno(ruido_percepcion=0.5, probabilidad_fallo_comunicacion=0.1)
algoritmo = AlgoritmoConsensoIncertidumbre(
    entorno,
    epsilon=0.2,
    lambda_penalizacion=0.3,
    iteraciones_consenso=5
)
rondas = algoritmo.ejecutar(max_rondas=1000)
```

## Casos de Uso

### Caso de Uso 1: Algoritmo Greedy (sin incertidumbre)

Evalúa el Algoritmo 1 en un entorno determinista con tres experimentos:
- **Experimento 1.1:** Configuración base (5 robots, 10 tareas)
- **Experimento 1.2:** Más tareas (5 robots, 15 tareas)
- **Experimento 1.3:** Más robots (8 robots, 10 tareas)

**Ejecución:**
```bash
python casos_uso/caso_uso_1.py
```

### Caso de Uso 2: Algoritmo de Consenso (con incertidumbre)

Evalúa el Algoritmo 2 en un entorno con incertidumbre:
- **Experimento 2.1:** Incertidumbre baja (σ = 0.3)
- **Experimento 2.2:** Incertidumbre alta (σ = 1.0)

**Ejecución:**
```bash
python casos_uso/caso_uso_2.py
```

### Caso de Uso 3: Comparación Directa

Compara ambos algoritmos en las mismas condiciones:
- **Escenario 3.1:** Sin incertidumbre
- **Escenario 3.2:** Incertidumbre moderada (σ = 0.5)
- **Escenario 3.3:** Incertidumbre alta (σ = 1.5)

**Ejecución:**
```bash
python casos_uso/caso_uso_3.py
```

## Métricas de Evaluación

Los algoritmos se evalúan mediante las siguientes métricas:

1. **Rondas totales:** Número de rondas hasta completar todas las tareas
2. **Distancia total:** Suma de todas las distancias recorridas por todos los robots
3. **Tasa de completitud:** Fracción de tareas completadas
4. **Balance de carga:** Desviación estándar del número de tareas completadas por robot
5. **Error de estimación:** Error promedio entre posiciones reales y estimadas (solo Algoritmo 2)

## Ejemplo de Uso Básico

```python
import numpy as np
from utils.robot import Robot
from utils.tarea import Tarea
from utils.entorno import Entorno
from algoritmos.greedy_distribuido import AlgoritmoGreedyDistribuido

# Crear robots
robots = [
    Robot(id_robot=0, posicion_inicial=(0, 0)),
    Robot(id_robot=1, posicion_inicial=(5, 5))
]

# Crear tareas
tareas = [
    Tarea(id_tarea=0, posicion_pickup=(2, 2), posicion_delivery=(8, 8)),
    Tarea(id_tarea=1, posicion_pickup=(3, 3), posicion_delivery=(7, 7))
]

# Crear entorno
entorno = Entorno(
    dimensiones=(10, 10),
    robots=robots,
    tareas=tareas
)

# Ejecutar algoritmo
algoritmo = AlgoritmoGreedyDistribuido(entorno)
rondas = algoritmo.ejecutar(max_rondas=1000)

print(f"Completado en {rondas} rondas")
```
