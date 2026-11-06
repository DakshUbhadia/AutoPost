import ast
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import autopost.interfaces  # noqa: F401
    import autopost.eval.stats  # noqa: F401
    import autopost.verify.exact_scorer  # noqa: F401


def test_flat_fallback(tmp_path):
    shutil.copy(ROOT / "src/autopost/interfaces.py", tmp_path / "interfaces.py")
    shutil.copy(ROOT / "src/autopost/eval/stats.py", tmp_path / "stats.py")
    shutil.copytree(ROOT / "src/autopost/contracts", tmp_path / "contracts")
    code = 'import sys; sys.modules["autopost"] = None; sys.path.insert(0, r"%s"); import interfaces, stats' % tmp_path
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exact_scorer_imports_only_stdlib():
    path = ROOT / "src/autopost/verify/exact_scorer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= set(sys.stdlib_module_names)
    assert not any("train" in name for name in imports)
