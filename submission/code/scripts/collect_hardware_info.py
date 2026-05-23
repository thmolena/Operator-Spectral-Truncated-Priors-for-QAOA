#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from uq_qaoa.hardware import collect_hardware_info
out=ROOT/"results"/"hardware_info.json"; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(collect_hardware_info(), indent=2), encoding="utf-8")
print(f"wrote {out}")
