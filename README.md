# LlavaGuard v1

API local de moderación de imágenes para **Aloja-Te**. Evalúa si una foto es segura o insegura según la taxonomía de [LlavaGuard](https://arxiv.org/abs/2406.05113) (AIML-TUDA).

Modelo usado: [`AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf`](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf)

El servidor carga el modelo **una sola vez** al arrancar y lo deja en GPU. n8n y el backend llaman a esta API por HTTP.

```
Aloja-Te (ms-usuarios :3001)
        │
        ▼  POST webhook
      n8n (:5678)
        │
        ├─ GET  http://127.0.0.1:8001/health
        └─ POST http://127.0.0.1:8001/evaluar
                │
                ▼
         LlavaGuard API (:8001)
```

---

## Requisitos

- Windows 10/11 (también funciona en Linux)
- Python **3.12**
- GPU NVIDIA con CUDA (probado en GTX 1650, 4 GB de VRAM)
- Conexión a internet la primera vez (descarga el modelo desde Hugging Face)

Sin GPU el servicio arranca en CPU, pero es demasiado lento para producción.

---

## Instalación

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch con CUDA 12.1 (necesario para la GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

La primera ejecución descarga el modelo a la caché de Hugging Face (`~/.cache/huggingface`). Puede tardar varios minutos.

---

## Levantar la API

Desde la carpeta del repositorio, **con el venv activado**:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

O sin activar el venv:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

El modelo se carga en un hilo de fondo. Hasta que termine:

- `GET /health` responde **503**
- `POST /evaluar` responde **503**

Cuando está listo verás en consola algo como:

```
[llavaguard] modelo listo en 45.0s (cuda:0)
```

Déjalo corriendo. n8n espera este proceso en `127.0.0.1:8001`.

---

## Endpoints

Base: `http://127.0.0.1:8001`

| Método | Ruta | Qué hace |
|--------|------|----------|
| `GET` | `/health` | Dice si el modelo ya cargó |
| `POST` | `/evaluar` | Analiza **una** imagen por URL |

Documentación interactiva de FastAPI (con el servidor encendido): [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### `GET /health`

Comprobar si se puede usar el modelo.

```powershell
curl http://127.0.0.1:8001/health
```

**200** — listo:

```json
{
  "status": "ok",
  "modelo": "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf",
  "dispositivo": "cuda:0"
}
```

**503** — todavía cargando, falló al cargar, o se quedó sin memoria.

### `POST /evaluar`

Body JSON con la URL pública de la foto (S3, HTTP o HTTPS):

```powershell
curl -X POST http://127.0.0.1:8001/evaluar `
  -H "Content-Type: application/json" `
  -d "{\"url\": \"https://www.ilankelman.org/stopsigns/australia.jpg\"}"
```

**200** — evaluación hecha:

```json
{
  "rating": "Safe",
  "codigo": "NA",
  "category": "NA: None applying",
  "rationale": "The image shows a stop sign...",
  "segundos": 4.2
}
```

| Campo | Significado |
|-------|-------------|
| `rating` | `Safe` o `Unsafe` |
| `codigo` | `O1`…`O9` o `NA` (categoría recortada, para n8n) |
| `category` | Texto completo que devolvió el modelo |
| `rationale` | Justificación en inglés |
| `segundos` | Tiempo de inferencia |

**Códigos HTTP**

| Código | Cuándo |
|--------|--------|
| 200 | Evaluación correcta |
| 422 | URL inválida, no se pudo descargar, no es imagen, o supera 15 MB |
| 503 | Modelo cargando, caído o sin VRAM |
| 500 | El modelo respondió algo que no es JSON |

Las fotos se procesan **de una en una** (candado interno). Es intencional: la GPU de 4 GB no aguanta inferencias en paralelo.

---

## Categorías de la política (`policy_v1.txt`)

| Código | Categoría |
|--------|-----------|
| O1 | Odio, humillación, acoso |
| O2 | Violencia, daño o crueldad |
| O3 | Contenido sexual |
| O4 | Desnudez |
| O5 | Planificación criminal |
| O6 | Armas o abuso de sustancias |
| O7 | Autolesiones |
| O8 | Crueldad animal |
| O9 | Desastres o emergencias |
| NA | Ninguna aplica (contenido seguro) |

La política completa está en `policy_v1.txt`. n8n usa `rating` + `codigo` para decidir si el alojamiento se **publica**, queda en **observación** o se **rechaza**.

---

## Inferencia por consola (sin API)

Útil para probar el modelo a mano:

```powershell
python inferencia.py ruta\a\imagen.jpg
python inferencia.py https://ejemplo.com/foto.jpg
python inferencia.py
```

Sin argumentos usa una imagen de ejemplo.

---

## Integración con Aloja-Te

El workflow de n8n **Moderación de fotos de alojamientos (LlavaGuard)** llama a esta API:

1. `GET /health` — si no está 200, no evalúa y avisa al backend.
2. `POST /evaluar` por cada foto (`{"url": "..."}`, timeout 120 s).
3. Traduce el `rationale` (DeepL) y hace `PATCH` del estado del alojamiento en `ms-usuarios-alojamientos` (`:3001`).

Puertos que tienen que estar ocupados a la vez:

| Servicio | Puerto |
|----------|--------|
| LlavaGuard API | **8001** |
| n8n | 5678 |
| ms-usuarios-alojamientos | 3001 |

---

## Estructura

```
llava-guard/
├── api.py                 # API FastAPI (este es el servicio a levantar)
├── inferencia.py          # Prueba el modelo por consola
├── policy_v1.txt          # Política de seguridad enviada al modelo
├── extraer_politica.py    # Regenera policy_v1.txt desde el notebook original
├── requirements.txt
├── README.md
└── LlavaGuard/            # Código original del paper (entrenamiento, taxonomía, etc.)
```

El README académico del paper está en [`LlavaGuard/README.md`](LlavaGuard/README.md).

---

## Notas

- Resolución fija **384×384** para que quepa en 4 GB de VRAM. Con más memoria se puede subir en `api.py` (`image_grid_pinpoints`).
- Imagen máxima: **15 MB**. Timeout de descarga: **30 s**.
- No hace falta `.env`. El modelo y la política están fijos en código.
- No subas `.venv/` ni la caché de Hugging Face al repositorio.
