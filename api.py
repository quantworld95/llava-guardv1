"""API local de moderacion de imagenes con LlavaGuard-v1.2-0.5B-OV-hf.

Arranque (desde la carpeta llava-guard):
    .venv\\Scripts\\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001

Endpoints:
    GET  /health   -> 200 si el modelo esta listo, 503 si aun carga o fallo.
    POST /evaluar  -> body {"url": "https://..."}; analiza UNA imagen.
        200: {"rating": "Safe|Unsafe", "codigo": "O1..O9|NA", "category": "texto del modelo",
              "rationale": "...", "segundos": 4.2}
        422: la imagen no se pudo descargar o abrir (error de la foto).
        503: el modelo no esta disponible (cargando, caido o sin memoria).
        500: el modelo respondio algo que no es JSON valido.

Las evaluaciones se procesan de una en una (candado), por la GPU de 4 GB.
"""
import json
import re
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import requests
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

MODEL_ID = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
POLICY_PATH = Path(__file__).parent / "policy_v1.txt"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30

# Estado compartido del modelo (se llena en un hilo al arrancar).
estado = {"listo": False, "error": None, "model": None, "processor": None,
          "policy": None, "device": None, "dtype": None}
candado_inferencia = threading.Lock()


def cargar_modelo():
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        t0 = time.time()
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        # GPU de 4 GB: una sola resolucion base de 384x384 para evitar OOM.
        pinpoints = [[384, 384]]
        processor.image_processor.image_grid_pinpoints = pinpoints
        model.config.image_grid_pinpoints = pinpoints
        estado.update(
            model=model, processor=processor, device=device, dtype=dtype,
            policy=POLICY_PATH.read_text(encoding="utf8"), listo=True,
        )
        print(f"[llavaguard] modelo listo en {time.time() - t0:.1f}s ({device})")
    except Exception as exc:  # noqa: BLE001
        estado["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[llavaguard] fallo al cargar el modelo: {estado['error']}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=cargar_modelo, daemon=True).start()
    yield


app = FastAPI(title="LlavaGuard API", lifespan=lifespan)


class EvaluarRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    if estado["listo"]:
        return {"status": "ok", "modelo": MODEL_ID, "dispositivo": estado["device"]}
    detalle = estado["error"] or "El modelo aun esta cargando."
    raise HTTPException(status_code=503, detail=detalle)


def descargar_imagen(url: str) -> Image.Image:
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="La URL debe empezar con http:// o https://")
    try:
        with requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True,
                          headers={"User-Agent": "Mozilla/5.0"}) as resp:
            resp.raise_for_status()
            contenido = bytearray()
            for trozo in resp.iter_content(64 * 1024):
                contenido.extend(trozo)
                if len(contenido) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=422, detail="La imagen supera los 15 MB.")
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo descargar la imagen: {exc}")
    try:
        return Image.open(BytesIO(bytes(contenido))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="El archivo descargado no es una imagen valida.")


def extraer_json(texto: str) -> dict:
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        coincidencia = re.search(r"\{.*\}", texto, re.DOTALL)
        if coincidencia:
            return json.loads(coincidencia.group(0))
        raise


@app.post("/evaluar")
def evaluar(req: EvaluarRequest):
    if not estado["listo"]:
        raise HTTPException(status_code=503, detail=estado["error"] or "El modelo aun esta cargando.")

    image = descargar_imagen(req.url)

    model, processor = estado["model"], estado["processor"]
    device, dtype = estado["device"], estado["dtype"]
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": estado["policy"]}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    with candado_inferencia:
        t0 = time.time()
        try:
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                ids = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False, num_beams=1, use_cache=True
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="Sin memoria de GPU para analizar la imagen.")
        nuevos = ids[:, inputs["input_ids"].shape[1]:]
        salida = processor.batch_decode(
            nuevos, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        segundos = round(time.time() - t0, 2)

    try:
        datos = extraer_json(salida)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Respuesta del modelo no es JSON: {salida[:300]}")

    categoria = datos.get("category")
    codigo = None
    if isinstance(categoria, str):
        m = re.match(r"\s*(O\d+|NA)\b", categoria, re.IGNORECASE)
        codigo = m.group(1).upper() if m else None

    return {
        "rating": datos.get("rating"),
        "codigo": codigo,
        "category": categoria,
        "rationale": datos.get("rationale"),
        "segundos": segundos,
    }
