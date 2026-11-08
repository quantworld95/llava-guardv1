"""Inferencia con LlavaGuard-v1.2-0.5B-OV-hf (Transformers, GPU de 4 GB).

Uso:
    python inferencia.py ruta\\a\\imagen.jpg
    python inferencia.py https://url/de/imagen.jpg
    python inferencia.py            # usa una imagen de ejemplo
"""
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

MODEL_ID = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
DEFAULT_URL = "https://www.ilankelman.org/stopsigns/australia.jpg"
POLICY_PATH = Path(__file__).parent / "policy_v1.txt"


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Dispositivo: {device} | dtype: {dtype}")

    print(f"Cargando modelo {MODEL_ID} ...")
    t0 = time.time()
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # GPU de 4 GB: por defecto OneVision divide la imagen en muchos recortes
    # (anyres) y genera >10k tokens visuales -> OOM. Se limita a una sola
    # resolucion base de 384x384 (~1.5k tokens). Menos detalle, pero cabe.
    pinpoints = [[384, 384]]
    processor.image_processor.image_grid_pinpoints = pinpoints
    model.config.image_grid_pinpoints = pinpoints
    print(f"Modelo cargado en {time.time() - t0:.1f}s")

    policy = POLICY_PATH.read_text(encoding="utf8")
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": policy}],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image = load_image(source)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    inputs = {
        k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
        for k, v in inputs.items()
    }

    hyperparameters = {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "num_beams": 1,  # 1 para ahorrar VRAM (el notebook usa 2)
        "use_cache": True,
    }

    print(f"Evaluando: {source}")
    t0 = time.time()
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, **hyperparameters)
    # Decodifica solo los tokens nuevos (sin repetir el prompt)
    new_tokens = generate_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"Inferencia en {time.time() - t0:.1f}s\n")
    print(out)
    if device.startswith("cuda"):
        print(f"\nVRAM pico: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
