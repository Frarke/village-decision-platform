"""只读盘点项目业务材料；环境、Git对象和缓存不视为研究数据。"""
import ast
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".venv_codex", ".pytest_cache", "__pycache__", "output", "tmp", "temp"}


def describe(path, content):
    suffix = path.suffix.lower()
    if suffix in {".json", ".geojson"}:
        value = json.loads(content.decode("utf-8-sig"))
        return f"JSON实际解析；顶层键={','.join(value) if isinstance(value, dict) else '数组'}"
    if suffix == ".csv":
        rows = list(csv.reader(content.decode("utf-8-sig").splitlines()))
        return f"CSV实际解析；数据行={max(0,len(rows)-1)}；列={len(rows[0]) if rows else 0}"
    if suffix == ".py":
        tree = ast.parse(content.decode("utf-8-sig"))
        functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return "Python语法树解析；入口=" + ",".join(functions[:12])
    if suffix in {".docx", ".pptx"}:
        with ZipFile(path) as archive:
            names = ["word/document.xml"] if suffix == ".docx" else [n for n in archive.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            texts = ["".join(n.text or "" for n in ET.fromstring(archive.read(name)).iter() if n.tag.endswith("}t")) for name in names]
            return f"Office XML实际解析；正文/幻灯片={len(names)}；文本字符={sum(map(len,texts))}"
    if suffix in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        with Image.open(path) as picture:
            return f"图片实际读取；{picture.format}；{picture.size[0]}×{picture.size[1]}；普通图片不是GIS数据"
    if suffix == ".mat":
        from scipy.io import whosmat
        return "MAT实际解析；" + ";".join(f"{name}:{dimensions}" for name, dimensions, _ in whosmat(path))
    if suffix in {".md", ".m", ".ps1", ".bat", ".txt", ".toml"} or path.name == ".gitignore":
        text = content.decode("utf-8-sig")
        return f"文本实际读取；行={len(text.splitlines())}；字符={len(text)}"
    return "二进制内容读取并计算SHA256；未作为计算输入"


def main():
    rows = []
    target = ROOT / "docs/V1_AUDIT_INVENTORY.csv"
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts) or not path.is_file() or path == target:
            continue
        content = path.read_bytes()
        try:
            inspection = describe(path, content)
        except (ValueError, UnicodeDecodeError, SyntaxError, ImportError) as exc:
            inspection = f"未完整解析：{type(exc).__name__}: {exc}"
        rows.append({"path": str(relative), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                     "inspection": inspection, "private_original": "是" if relative.parts[0] == "references_private" else "否"})
    with target.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已盘点{len(rows)}个业务文件；未修改原始材料。")


if __name__ == "__main__":
    main()
