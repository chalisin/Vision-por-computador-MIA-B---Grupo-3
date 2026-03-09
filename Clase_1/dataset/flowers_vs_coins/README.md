# flowers_vs_coins dataset

Two-class image dataset: **flowers** vs **coins**.

## Source
- Downloaded automatically using `icrawler` (BingImageCrawler)
- Queries: `flower`, `coins`

## Structure
```
datasets/flowers_vs_coins/
  raw/
    flowers/
    coins/
  train/
    flowers/
    coins/
  val/
    flowers/
    coins/
```

## Notes
- The script removes non-image files and corrupted images.
- Split: 80/20 (train/test).

## Final counts
{'flowers': {'raw': 22, 'train': 17, 'test': 5}, 'coins': {'raw': 30, 'train': 24, 'test': 6}}
