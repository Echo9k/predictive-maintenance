"""Patch the local notebook into a Kaggle-ready notebook.

Changes:
  - Cell 3: detect Kaggle env, set DATA (input) and DERIVED (working) accordingly,
    write build_features.py to disk under src/ so the rest of the notebook's
    `from src.build_features import ...` imports work unchanged.
  - Cell 11: change features.parquet target to DERIVED (Kaggle input is read-only).

Run from repo root: python kaggle_kernel/patch.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "predictive_maintenance.ipynb"
DST = REPO / "kaggle_kernel" / "predictive_maintenance.ipynb"

# Read the source module verbatim, then patch DATA to be configurable.
bf_src = (REPO / "src" / "build_features.py").read_text()
# Replace hardcoded DATA path with one that respects an env override.
bf_patched = bf_src.replace(
    'DATA = Path(__file__).resolve().parents[1] / "data"',
    'import os\n'
    'DATA = Path(os.environ.get("PDM_DATA", '
    'Path(__file__).resolve().parents[1] / "data"))\n'
    'DERIVED = Path(os.environ.get("PDM_DERIVED", DATA))'
).replace(
    'out_path = DATA / "features.parquet"',
    'DERIVED.mkdir(parents=True, exist_ok=True)\n'
    '    out_path = DERIVED / "features.parquet"'
)

CELL3_SRC = (
    'import sys, os, platform, warnings, numpy as np, pandas as pd\n'
    'import matplotlib.pyplot as plt, seaborn as sns\n'
    'warnings.filterwarnings("ignore")\n'
    'plt.rcParams["figure.figsize"] = (9, 4); sns.set_style("whitegrid")\n'
    'SEED = 42; np.random.seed(SEED)\n'
    'from pathlib import Path\n'
    '\n'
    '# --- environment detection ----------------------------------------------\n'
    'IS_KAGGLE = Path("/kaggle/input").exists()\n'
    'if IS_KAGGLE:\n'
    '    # Resolve the dataset mount even if Kaggle renames the folder.\n'
    '    _expected = Path("/kaggle/input/microsoft-azure-predictive-maintenance")\n'
    '    if (_expected / "PdM_telemetry.csv").exists():\n'
    '        DATA = _expected\n'
    '    else:\n'
    '        _hits = list(Path("/kaggle/input").rglob("PdM_telemetry.csv"))\n'
    '        if not _hits:\n'
    '            raise FileNotFoundError(\n'
    '                "PdM_telemetry.csv not found under /kaggle/input. "\n'
    '                "Attach dataset arnabbiswas1/microsoft-azure-predictive-maintenance to this kernel."\n'
    '            )\n'
    '        DATA = _hits[0].parent\n'
    '    DERIVED = Path("/kaggle/working")\n'
    '    # materialize src/build_features.py so existing `from src...` imports work\n'
    '    SRC_DIR = Path("/kaggle/working/src"); SRC_DIR.mkdir(exist_ok=True)\n'
    '    (SRC_DIR / "__init__.py").write_text("")\n'
    '    (SRC_DIR / "build_features.py").write_text(_BUILD_FEATURES_SRC)\n'
    '    sys.path.insert(0, "/kaggle/working")\n'
    '    os.environ["PDM_DATA"]    = str(DATA)\n'
    '    os.environ["PDM_DERIVED"] = str(DERIVED)\n'
    'else:\n'
    '    sys.path.append("..")  # allow `import src...` when run from repo root or src/\n'
    '    ROOT = Path.cwd() if (Path.cwd()/"data").exists() else Path.cwd().parent\n'
    '    DATA = ROOT / "data"\n'
    '    DERIVED = DATA\n'
    'print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)\n'
    'import lightgbm, sklearn; print("lightgbm", lightgbm.__version__, "| sklearn", sklearn.__version__)\n'
    'print("env:", "Kaggle" if IS_KAGGLE else "local", "| DATA =", DATA)\n'
)

# Embed the patched build_features module source as a Python triple-quoted string
# in a new cell that runs before cell 3 so the constant exists when cell 3 needs it.
BF_BOOT_CELL = (
    '# Inline copy of src/build_features.py — written to disk inside the next cell\n'
    '# so the rest of the notebook can `from src.build_features import ...` unchanged.\n'
    '_BUILD_FEATURES_SRC = r\'\'\'' + bf_patched + '\'\'\'\n'
)

nb = json.loads(SRC.read_text())

def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

# Replace cell 3 (env/imports) with the new env-aware version.
nb["cells"][3] = code_cell(CELL3_SRC)
# Insert the inline build_features source cell right before cell 3.
nb["cells"].insert(3, code_cell(BF_BOOT_CELL))
# Insert a Kaggle-only install cell at the very top so packages are ready.
INSTALL_CELL = (
    '# Kaggle setup: ensure non-default deps are available. No-op locally.\n'
    'import sys, subprocess\n'
    'from pathlib import Path\n'
    'if Path("/kaggle/input").exists():\n'
    '    subprocess.run([sys.executable, "-m", "pip", "install", "-q",\n'
    '                    "dice-ml==0.12", "lifelines>=0.30", "shap"],\n'
    '                   check=False)\n'
)
nb["cells"].insert(0, code_cell(INSTALL_CELL))

# Locate the features.parquet cell by content (index shifts as cells get inserted).
cell11 = next(c for c in nb["cells"]
              if c["cell_type"] == "code"
              and '_fpath = DATA/"features.parquet"' in "".join(c["source"]))
src11 = "".join(cell11["source"])
new11 = src11.replace(
    '_fpath = DATA/"features.parquet"',
    '_fpath = DERIVED/"features.parquet"',
)
cell11["source"] = new11.splitlines(keepends=True)
cell11["outputs"] = []
cell11["execution_count"] = None

# Strip all outputs and execution counts — Kaggle re-runs from scratch.
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Drop heavy metadata.
nb.get("metadata", {}).pop("widgets", None)

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote {DST} ({DST.stat().st_size:,} bytes, {len(nb['cells'])} cells)")
