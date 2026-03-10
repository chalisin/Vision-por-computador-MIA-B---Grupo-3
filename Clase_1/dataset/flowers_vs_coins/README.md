# flowers_vs_airplanes dataset

Two-class image dataset: **flowers** vs **airplanes**.

## Source
- Downloaded automatically using `icrawler` (BingImageCrawler)
- Queries: `flower`, `airplane`

## Structure
```
datasets/flowers_vs_airplanes/
  raw/
    flowers/
    airplanes/
  train/
    flowers/
    airplanes/
  val/
    flowers/
    airplanes/
```

## Notes
- The script removes non-image files and corrupted images.
- Split: 80/20 (train/val).

## Final counts
{'flowers': {'raw': 20, 'train': 16, 'val': 4}, 'coins': {'raw': 36, 'train': 28, 'val': 8}}
