"""Read competition pages, all discussion pages, and full replies via the Kaggle SDK."""

import json
from pathlib import Path

from kaggle import api

out = Path("references/official")
out.mkdir(parents=True, exist_ok=True)
competition = "filament-segmentation-2026"
pages = api.competition_list_pages(competition)
(out / "competition-pages.json").write_text("[" + ",".join(str(p) for p in pages) + "]")
for page in pages:
    value = json.loads(str(page))
    (out / (value["name"] + ".md")).write_text(value["content"])
topics = []
page = 1
while True:
    response = json.loads(str(api.competition_list_topics(competition, page=page)))
    batch = response.get("topics", [])
    if not batch:
        break
    topics.extend(batch)
    if len(topics) >= response.get("totalCount", len(topics)):
        break
    page += 1
(out / "topics.json").write_text(json.dumps(topics, indent=2))
for topic in topics:
    response = api.competition_list_topic_messages(competition, topic["id"], page_size=-1)
    folder = out / "discussions"
    folder.mkdir(exist_ok=True)
    (folder / f"{topic['id']}.json").write_text(str(response))
print(f"Saved {len(pages)} pages and {len(topics)} discussions.")
