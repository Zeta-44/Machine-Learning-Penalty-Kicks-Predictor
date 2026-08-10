# Predicción de Penales mediante Transfer Learning

> **Trabajo Práctico Final** — Introducción a la Inteligencia Artificial (IIA)  
> **Integrantes:** Agustín López y Baltazar Lynn Bosio

---

## Descripción del Proyecto

El objetivo de este proyecto es predecir la dirección y posición final de un tiro penal a partir de una única imagen capturada en el momento exacto del impacto entre el pie del pateador y la pelota. 

El modelo se evalúa frente al **baseline humano** (la decisión real tomada por el arquero durante el penal) para analizar si la visión por computadora permite anticipar la trayectoria del disparo con mayor precisión que un guardameta profesional.

---

## Experimentos y Modelos

Se entrenan tres modelos independientes basados en la arquitectura **InceptionV3** (preentrenada en ImageNet):

| Experimento | Tipo de Tarea | Salida | Descripción |
| :--- | :--- | :--- | :--- |
| **Experimento A** | Regresión | `(x, y)` | Coordenadas continuas normalizadas del cruce por la línea de gol ($x \in [0,1]$, $y \in [0,1]$). |
| **Experimento B** | Clasificación 3 Clases | `L / C / R` | Izquierda ($x < 0.4$), Centro ($0.4 \le x \le 0.6$), Derecha ($x > 0.6$). |
| **Experimento C** | Clasificación 2 Clases | `L / R` | Izquierda ($x < 0.5$) vs. Derecha ($x \ge 0.5$). |

> En la convención de coordenadas: `(0, 0)` representa el poste inferior izquierdo del arco desde la perspectiva de la cámara, y `(1, 1)` representa el ángulo superior derecho.

---

## Metodología y Estrategia de Validación

1. **Split & Grouping (`StratifiedGroupKFold`)**:
   - Un mismo tiro penal puede contar con múltiples frames de cámara (frame base + frames extras).
   - Se utiliza **5-Fold Cross Validation** agrupando por `penal_id` para garantizar que **todos** los frames de un mismo penal queden en el mismo fold (evitando *data leakage* entre entrenamiento y validación).
   - Se reserva un **15% del dataset como Test Hold-Out** totalmente independiente.

2. **Data Augmentation Específico de Dominio**:
   - **Horizontal Flip**: Se voltea la imagen horizontalmente y se ajustan dinámicamente sus etiquetas:
     $$x_{new} = 1 - x$$
     $$\text{Etiquetas: } L \leftrightarrow R$$

3. **Entrenamiento en Dos Etapas (Transfer Learning)**:
   - **Etapa 1**: Congelamiento de las capas convolucionales de InceptionV3 y entrenamiento de la cabeza clasificadora/regresora (Learning Rate alto).
   - **Etapa 2**: *Fine-tuning* de las últimas capas convolucionales con Learning Rate reducido y *Early Stopping* sobre `val_loss`.

4. **Agregación por Penal & Ensemble**:
   - **Nivel Frame**: Predicción individual para cada crop de imagen.
   - **Nivel Penal**: Promedio de las predicciones (coordenadas o probabilidades) de todos los frames pertenecientes a un mismo disparo. Reduciendo el ruido inter-frame y permitiendo la comparación directa contra la decisión del arquero.
   - Para la evaluación final en Test, se aplica un **Ensemble** promediando las 5 redes entrenadas en la validación cruzada.

---

## Estructura del Repositorio

```text
.
├── IIA_Trabajo_Práctico_Final.ipynb  # Notebook principal con pipeline de carga, training, evaluation y plots
├── IIA.pdf                           # Documentación / Consignas / Informe del trabajo
├── README.md                         # Descripción del proyecto y guía de uso
└── dataset/
    ├── audit_dataset.py         # Script de auditoría e integridad del dataset
    ├── labels.csv               # Metadatos y anotaciones (penal, x, y, arquero)
    └── img/                     # Imágenes de los penales (frames base y extras)
```

---

## Dataset y Auditoría

El dataset contiene información de penales de competencias internacionales (Eurocopa, Nations League, etc.).

- `labels.csv`: Contiene las columnas `penal`, `x`, `y` y `arquero` (dirección a la que se tiró el golero: `L`, `CL`, `CR`, `R`).
- **Frames Secuenciales**:
  - Frame base: `penal_..._01.png`
  - Frames extras: `penal_..._01_1.png`, `penal_..._01_2.png`, etc.

### Script de Auditoría

El proyecto incluye una herramienta CLI para validar la calidad e integridad de las imágenes y anotaciones:

```bash
python dataset/audit_dataset.py
```

Opciones:
- `--strict`: Termina con código de error `1` en caso de detectar inconsistencias o imágenes faltantes.

---

## Requisitos e Instalación

### Prerrequisitos
- Python 3.10+ (Recomendado 3.11)
- CUDA / GPU compatible con TensorFlow (Opcional, pero recomendado para acelerar el entrenamiento).

### Instalación de Dependencias

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/nombre-del-repo.git
   cd nombre-del-repo
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/macOS:
   source .venv/bin/activate
   ```

3. Instalar los paquetes necesarios:
   ```bash
   pip install tensorflow numpy pandas matplotlib scipy scikit-learn pillow
   ```

---

## 🚀 Uso

1. **Auditar el dataset**:
   ```bash
   python dataset/audit_dataset.py
   ```

2. **Ejecutar el Notebook**:
   Abrir `IIA_Trabajo_Práctico_Final.ipynb` en VS Code, JupyterLab o Google Colab y ejecutar todas las celdas secuencialmente para reproducir el entrenamiento, los resultados Out-of-Fold (OOF) y la evaluación final sobre el conjunto de test.

---

## Resultados y Evaluación

El notebook genera análisis completos que incluyen:
- **Curvas de Aprendizaje**: Evolución de Loss/Accuracy (media $\pm$ desvío estándar entre los 5 folds).
- **Métricas de Regresión**: MAE (Mean Absolute Error) y $std_x$ en $x$ e $y$.
- **Métricas de Clasificación**: Accuracy, F1-Score, Matriz de Confusión y test estadístico de **McNemar** para comparar el desempeño del modelo vs. el baseline del arquero humano.
