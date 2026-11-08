import json
from pathlib import Path

base = Path(__file__).parent
nb = json.loads((base / "LlavaGuard" / "scripts" / "inference" / "tranformers.ipynb").read_text(encoding="utf8"))

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "policy_v1 =" in src:
        ns = {}
        # Ejecuta solo las asignaciones de las politicas (hasta antes de usar processor)
        code = src.split("policy = policy_v1")[0]
        exec(code, ns)
        (base / "policy_v1.txt").write_text(ns["policy_v1"], encoding="utf8")
        print("policy_v1.txt escrito:", len(ns["policy_v1"]), "caracteres")
        break
else:
    raise SystemExit("No se encontro policy_v1 en el notebook")
