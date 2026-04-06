# This file documents the intended SwimVision data collection workflow and source commands.
# Data Collection Notes

SwimVision expects raw clips in `data/raw/` and metadata in `data/labels.csv`.

## Example `yt-dlp` Commands
Use one command per source video and rename outputs to match the corresponding `clip_id`.

```bash
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "data/raw/elite_start_01.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID_1"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "data/raw/elite_start_02.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID_2"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "data/raw/elite_start_03.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID_3"
```

Replace each `VIDEO_ID_*` placeholder with the exact YouTube source used for the clip. For reproducibility, record the final downloaded filename, swimmer identity if known, competition context, and camera angle in `data/labels.csv`.
