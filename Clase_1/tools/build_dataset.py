from __future__ import annotations
from pathlib import Path
import random
import shutil

from PIL import Image
from icrawler.builtin import BingImageCrawler
import ssl
import os
import certifi
import pip_system_certs.bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "flowers_vs_coins"


CLASSES = {
    "flowers": "flower",
    "coins": "coin",
}


def download_raw_images(limit_per_class: int = 30) -> None:
    """
    Downloads images using Bing via icrawler.

    This repository runs on Windows where Python may lack access to system
    CA certificates (CERTIFICATE_VERIFY_FAILED). We bootstrap system certs
    via pip-system-certs, and keep certifi fallback.
    """
    try:
        pip_system_certs.bootstrap.bootstrap()
    except Exception:
        pass

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    try:
        ssl._create_default_https_context = ssl.create_default_context(
            cafile=certifi.where()
        )
    except Exception:
        pass

    raw_root = DATASET_ROOT / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    for cls, query in CLASSES.items():
        out_dir = raw_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        crawler = BingImageCrawler(
            storage={"root_dir": str(out_dir)},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )
        crawler.crawl(keyword=query, max_num=limit_per_class)


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        # Re-open to ensure it can be loaded (verify() is not enough in some cases)
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def clean_raw_images() -> dict[str, int]:
    raw_root = DATASET_ROOT / "raw"
    removed = 0
    kept = 0

    for cls in CLASSES.keys():
        cls_dir = raw_root / cls
        if not cls_dir.exists():
            continue

        for p in cls_dir.rglob("*"):
            if p.is_dir():
                continue
            # Keep only common image extensions
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
                try:
                    p.unlink()
                except Exception:
                    pass
                removed += 1
                continue

            if not is_valid_image(p):
                try:
                    p.unlink()
                except Exception:
                    pass
                removed += 1
            else:
                kept += 1

    return {"kept": kept, "removed": removed}


def split_train_val(train_ratio: float = 0.8, seed: int = 42) -> dict[str, dict[str, int]]:
    raw_root = DATASET_ROOT / "raw"
    train_root = DATASET_ROOT / "train"
    test_root = DATASET_ROOT / "test"

    def _on_rm_error(func, path, exc_info):
        # Windows/OneDrive can mark files/folders read-only; try to fix perms
        try:
            os.chmod(path, 0o777)
        except Exception:
            pass
        try:
            func(path)
        except Exception:
            pass

    # Recreate train/val
    if train_root.exists():
        shutil.rmtree(train_root, onerror=_on_rm_error)
    if test_root.exists():
        shutil.rmtree(test_root, onerror=_on_rm_error)

    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    counts: dict[str, dict[str, int]] = {}
    for cls in CLASSES.keys():
        files = [p for p in (raw_root / cls).glob("*") if p.is_file()]
        rng.shuffle(files)

        n_train = int(len(files) * train_ratio)
        train_files = files[:n_train]
        test_files = files[n_train:]

        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (test_root / cls).mkdir(parents=True, exist_ok=True)

        for src in train_files:
            shutil.copy2(src, train_root / cls / src.name)
        for src in test_files:
            shutil.copy2(src, test_root / cls / src.name)

        counts[cls] = {"raw": len(files), "train": len(train_files), "val": len(test_files)}

    return counts


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    print("Downloading raw images...")
    download_raw_images(limit_per_class=30)

    print("Cleaning raw images...")
    clean_stats = clean_raw_images()
    print(f"Clean stats: {clean_stats}")

    print("Splitting train/val...")
    counts = split_train_val(train_ratio=0.8, seed=42)
    print("Counts:")
    for cls, c in counts.items():
        print(f"  {cls}: raw={c['raw']} train={c['train']} val={c['val']}")

    readme = DATASET_ROOT / "README.md"
    readme.write_text(
        "# flowers_vs_airplanes dataset\n\n"
        "Two-class image dataset: **flowers** vs **airplanes**.\n\n"
        "## Source\n"
        "- Downloaded automatically using `icrawler` (BingImageCrawler)\n"
        "- Queries: `flower`, `airplane`\n\n"
        "## Structure\n"
        "```\n"
        "datasets/flowers_vs_airplanes/\n"
        "  raw/\n"
        "    flowers/\n"
        "    airplanes/\n"
        "  train/\n"
        "    flowers/\n"
        "    airplanes/\n"
        "  val/\n"
        "    flowers/\n"
        "    airplanes/\n"
        "```\n\n"
        "## Notes\n"
        "- The script removes non-image files and corrupted images.\n"
        "- Split: 80/20 (train/val).\n\n"
        f"## Final counts\n{counts}\n",
        encoding="utf-8",
    )

    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
