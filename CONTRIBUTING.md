# Guía de Contribución — IBioML

¡Gracias por tu interés en contribuir! Esta guía explica cómo configurar el entorno de desarrollo, el flujo de trabajo con Git y cómo publicar una nueva versión.

---

## Índice

1. [Configurar el entorno de desarrollo](#1-configurar-el-entorno-de-desarrollo)
2. [Flujo de trabajo con Git](#2-flujo-de-trabajo-con-git)
3. [Convenciones de commits](#3-convenciones-de-commits)
4. [Estilo de código](#4-estilo-de-código)
5. [Tests](#5-tests)
6. [Publicar una nueva versión](#6-publicar-una-nueva-versión)
7. [Reportar bugs y sugerir mejoras](#7-reportar-bugs-y-sugerir-mejoras)

---

## 1. Configurar el entorno de desarrollo

Para trabajar en IBioML sin necesidad de reinstalar la librería cada vez que hacés un cambio, usá la **instalación en modo editable** (`pip install -e .`). Esto crea un enlace simbólico al código fuente local: cualquier modificación en `ibioml/` se refleja de inmediato.

```bash
# 1. Clonar el repositorio
git clone https://github.com/mariburginlab-labPrograms/IBioML.git
cd IBioML

# 2. Crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instalar en modo editable con dependencias de desarrollo
pip install -e ".[dev]"
```

El extra `[dev]` instala automáticamente `pytest`, `black`, `flake8`, `mypy` y `pre-commit`.

> **Verificación:** `import ibioml` en cualquier script importará el código de tu clon local.

---

## 2. Flujo de trabajo con Git

Este proyecto sigue **GitHub Flow**: toda contribución parte de una rama propia y llega a `main` mediante un Pull Request.

**Nunca se hace commit directo a `main`.**

### Pasos para una contribución

```bash
# 1. Asegurarte de tener main actualizado
git checkout main
git pull origin main

# 2. Crear una rama descriptiva desde main
git checkout -b feat/nombre-de-la-feature
# o para correcciones:
git checkout -b fix/descripcion-del-bug

# 3. Hacer los cambios y commitear
git add <archivos>
git commit -m "feat(models): agregar soporte para transformer"

# 4. Pushear la rama al remote
git push -u origin feat/nombre-de-la-feature

# 5. Abrir un Pull Request en GitHub hacia main
```

### Convención de nombres de ramas

| Prefijo | Uso |
|---|---|
| `feat/` | Nueva funcionalidad |
| `fix/` | Corrección de bug |
| `docs/` | Cambios en documentación |
| `refactor/` | Refactorización sin cambio de comportamiento |
| `test/` | Agregar o mejorar tests |
| `chore/` | Tareas de mantenimiento (dependencias, CI, etc.) |

### Pull Requests

- Describí claramente qué cambia y por qué.
- Relacioná el PR con el issue correspondiente si existe (`Fixes #123`).
- El PR debe tener al menos **una revisión aprobada** antes de mergearse.
- Asegurate de que los tests pasen antes de pedir revisión.

---

## 3. Convenciones de commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
tipo(ámbito): descripción breve en imperativo

Cuerpo opcional con más detalle.

Fixes #123
```

**Tipos principales:**

| Tipo | Cuándo usarlo |
|---|---|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `docs` | Documentación |
| `style` | Formato (sin cambio de lógica) |
| `refactor` | Refactorización |
| `test` | Tests |
| `chore` | Mantenimiento (versiones, CI, etc.) |

**Ejemplos:**

```
feat(models): agregar modelo GRU bidireccional
fix(tuner): corregir índices en validación cruzada anidada
docs(preprocessing): mejorar descripción de parámetros
chore: bump versión a 0.1.7
```

---

## 4. Estilo de código

IBioML sigue [PEP 8](https://pep8.org/) con formateo automático via `black`.

```bash
# Formatear código
black ibioml/

# Verificar estilo
flake8 ibioml/

# Verificar tipos
mypy ibioml/
```

Documentá las funciones públicas con docstrings estilo Google:

```python
def run_study(X, y, T, model_space, num_trials=10, outer_folds=5):
    """
    Ejecuta un estudio de optimización con validación cruzada anidada.

    Args:
        X: Datos de entrada de forma (n_samples, ...).
        y: Targets de forma (n_samples, n_outputs).
        T: Índices de tiempo por trial.
        model_space: Diccionario con la configuración del modelo.
        num_trials: Número de configuraciones a probar por Optuna.
        outer_folds: Número de folds externos.

    Returns:
        dict: Resultados del estudio con métricas por fold.
    """
```

---

## 5. Tests

```bash
# Correr todos los tests
pytest

# Correr un archivo específico
pytest tests/test_models.py

# Con reporte de cobertura
pytest --cov=ibioml
```

Cada nueva funcionalidad debería ir acompañada de su test correspondiente en `tests/`.

---

## 6. Publicar una nueva versión

Las releases **solo se pueden hacer desde la rama `main`**. El script `release.sh` automatiza todo el proceso:

```bash
# Primero asegurate de estar en main y tener todo mergeado
git checkout main
git pull origin main

# Probar en TestPyPI (por defecto)
./release.sh 0.1.7

# Publicar en PyPI real
./release.sh 0.1.7 pypi
```

El script hace automáticamente:

1. Verifica que estés en la rama `main` (aborta si no).
2. Actualiza la versión en `setup.py`.
3. Limpia builds anteriores.
4. Genera el paquete con `python -m build`.
5. Sube el paquete con `twine upload`.
6. Hace commit, tag `vX.Y.Z` y push.

### Prerrequisitos para publicar

```bash
pip install build twine
```

Configurá tus credenciales de PyPI en `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
```

---

## 7. Reportar bugs y sugerir mejoras

Usá el [sistema de issues de GitHub](https://github.com/mariburginlab-labPrograms/IBioML/issues):

- **Bug:** Describí el comportamiento esperado vs. el observado, incluí un ejemplo mínimo reproducible y la versión de IBioML (`pip show ibioml`).
- **Feature:** Describí el caso de uso, qué problema resuelve y, si es posible, una propuesta de API.

---

## Contacto

[jiponce@ibioba-mpsp-conicet.gov.ar](mailto:jiponce@ibioba-mpsp-conicet.gov.ar)
