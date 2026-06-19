import argparse
import html
import io
import json
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline


ROOT = Path(__file__).resolve().parent
HF_HOME = ROOT / ".hf-cache"
os.environ.setdefault("HF_HOME", str(HF_HOME))

ROYALROAD_BASE = "https://www.royalroad.com"
WUXIAWORLD_BASE = "https://www.wuxiaworld.com"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 LocalTTSReader/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

VOICES = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
]

LANGUAGES = [
    ("a", "American English"),
    ("b", "British English"),
]

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kokoro Local Reader Test</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #202522;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: #f7f7f4;
    }

    main {
      width: min(980px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.15;
      font-weight: 760;
    }

    .status {
      min-width: 190px;
      padding: 9px 12px;
      border: 1px solid #c9d0ca;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
      font-size: 13px;
      text-align: center;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      align-items: start;
    }

    textarea {
      width: 100%;
      min-height: 360px;
      resize: vertical;
      padding: 18px;
      border: 1px solid #b9c5bd;
      border-radius: 8px;
      background: #fffef9;
      color: #202522;
      font: 18px/1.7 Georgia, "Times New Roman", serif;
    }

    aside {
      display: grid;
      gap: 12px;
      padding: 14px;
      border: 1px solid #c9d0ca;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 650;
    }

    select,
    input {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid #b9c5bd;
      border-radius: 7px;
      background: #fffef9;
      color: #202522;
      font: inherit;
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 11px 14px;
      background: #245e4f;
      color: white;
      font-weight: 740;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.64;
    }

    .secondary { background: #626c66; }

    audio {
      width: 100%;
      margin-top: 16px;
    }

    pre {
      min-height: 112px;
      margin: 0;
      padding: 12px;
      overflow: auto;
      border: 1px solid #d3d9d5;
      border-radius: 8px;
      background: #1f2523;
      color: #dfe8e2;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    @media (max-width: 760px) {
      header,
      .layout {
        display: grid;
        grid-template-columns: 1fr;
      }

      .status { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Kokoro Local Reader Test</h1>
      <div class="status" id="status">Checking server...</div>
    </header>

    <div class="layout">
      <section>
        <textarea id="text">Hello. This is Kokoro running locally.</textarea>
        <audio id="audio" controls></audio>
      </section>

      <aside>
        <label>
          Voice
          <select id="voice"></select>
        </label>
        <label>
          Language
          <select id="language"></select>
        </label>
        <label>
          Speed
          <input id="speed" type="number" min="0.6" max="1.5" step="0.05" value="1.0" />
        </label>
        <button id="speak">Generate & Play</button>
        <button id="tone" type="button" class="secondary">Test Tone</button>
        <pre id="log"></pre>
      </aside>
    </div>
  </main>

  <script>
    const voices = __VOICES__;
    const languages = __LANGUAGES__;
    const $ = (id) => document.getElementById(id);

    function fillSelect(id, values, selected) {
      const select = $(id);
      select.innerHTML = values.map((item) => {
        const value = Array.isArray(item) ? item[0] : item;
        const label = Array.isArray(item) ? item[1] : item;
        const attr = value === selected ? " selected" : "";
        return `<option value="${value}"${attr}>${label}</option>`;
      }).join("");
    }

    function log(message) {
      $("log").textContent = message;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        $("status").textContent = data.pipeline_loaded ? "Pipeline loaded" : "Ready";
      } catch (error) {
        $("status").textContent = "Server unavailable";
      }
    }

    async function speak() {
      const button = $("speak");
      button.disabled = true;
      $("status").textContent = "Generating...";
      log("Sending request.");

      const started = performance.now();
      try {
        const response = await fetch("/api/speak", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: $("text").value,
            voice: $("voice").value,
            lang_code: $("language").value,
            speed: Number($("speed").value)
          })
        });

        if (!response.ok) {
          throw new Error(await response.text());
        }

        const elapsed = ((performance.now() - started) / 1000).toFixed(1);
        const metadata = response.headers.get("X-TTS-Metadata");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        $("audio").src = url;
        await $("audio").play();
        $("status").textContent = "Playing";
        log(`Generated ${Math.round(blob.size / 1024)} KB in ${elapsed}s.\n${metadata || ""}`);
      } catch (error) {
        $("status").textContent = "Error";
        log(String(error.message || error));
      } finally {
        button.disabled = false;
        refreshHealth();
      }
    }

    async function testTone() {
      $("status").textContent = "Playing tone";
      try {
        const response = await fetch("/api/tone");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        $("audio").src = url;
        await $("audio").play();
        log("If you hear a beep, browser audio works.");
      } catch (error) {
        $("status").textContent = "Tone error";
        log(String(error.message || error));
      }
    }

    fillSelect("voice", voices, "af_heart");
    fillSelect("language", languages, "a");
    $("speak").addEventListener("click", speak);
    $("tone").addEventListener("click", testTone);
    refreshHealth();
  </script>
</body>
</html>
"""


def make_test_tone(sample_rate=24000, seconds=1.0, frequency=440.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    wav = (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    audio = io.BytesIO()
    sf.write(audio, wav, sample_rate, format="WAV", subtype="PCM_16")
    return audio.getvalue()


def clean_html_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style|noscript|iframe|svg|canvas).*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</(p|div|li|h1|h2|h3|tr)>", "\n", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = html.unescape(value).replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def absolutize_url(url: str, base: str = ROYALROAD_BASE) -> str:
    return urljoin(base, html.unescape(url))


def fetch_public_html(url_or_path: str, base: str = ROYALROAD_BASE) -> str:
    url = absolutize_url(url_or_path, base)
    parsed = urlparse(url)
    allowed = urlparse(base).netloc
    if parsed.scheme not in {"https", "http"} or parsed.netloc != allowed:
        raise ValueError(f"Unsupported source URL: {url}")
    request = Request(url, headers=HTTP_HEADERS)
    with urlopen(request, timeout=25) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "charset" not in content_type:
            raise ValueError(f"Unexpected source response type: {content_type}")
        return response.read().decode("utf-8", errors="replace")


def parse_royalroad_cards(page_html: str, limit: int = 30) -> list[dict]:
    blocks = re.split(r'<div class="(?:row )?fiction-list-item(?: row)?">', page_html)
    items = []
    seen = set()
    for block in blocks[1:]:
        title_match = re.search(r'<h2 class="fiction-title">\s*<a href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not title_match:
            continue
        url = absolutize_url(title_match.group(1))
        if url in seen:
            continue
        seen.add(url)
        cover_match = re.search(r'<img[^>]+src="([^"]+)"', block, re.S)
        description_match = re.search(r'<div id="description-[^"]+"[^>]*>(.*?)</div>\s*</div>', block, re.S)
        tags = [clean_html_text(match) for match in re.findall(r'class="[^"]*fiction-tag[^"]*"[^>]*>(.*?)</a>', block, re.S)]
        stats = [clean_html_text(match) for match in re.findall(r'<span>([^<]*(?:Followers|Pages|Views|Chapters)[^<]*)</span>', block, re.I)]
        items.append(
            {
                "source": "royalroad",
                "title": clean_html_text(title_match.group(2)),
                "url": url,
                "cover": absolutize_url(cover_match.group(1)) if cover_match else "",
                "summary": clean_html_text(description_match.group(1))[:900] if description_match else "",
                "tags": tags[:8],
                "stats": stats[:6],
            }
        )
        if len(items) >= limit:
            break
    return items


def parse_royalroad_detail(page_html: str, url: str) -> dict:
    book = {}
    script_match = re.search(r'<script type="application/ld\+json">(.*?)</script>', page_html, re.S)
    if script_match:
        try:
            book = json.loads(html.unescape(script_match.group(1)))
        except json.JSONDecodeError:
            book = {}

    title = clean_html_text(str(book.get("name") or ""))
    author = ""
    if isinstance(book.get("author"), dict):
        author = str(book["author"].get("name") or "")
    description = clean_html_text(str(book.get("description") or ""))
    cover = str(book.get("image") or book.get("thumbnailUrl") or "")

    if not title:
        match = re.search(r'<meta property="og:title" content="([^"]+)"', page_html)
        title = clean_html_text(match.group(1)) if match else "RoyalRoad fiction"
    if not description:
        desc_match = re.search(r'<div class="description">\s*(.*?)\s*<label', page_html, re.S)
        description = clean_html_text(desc_match.group(1)) if desc_match else ""

    tags = [clean_html_text(match) for match in re.findall(r'class="[^"]*fiction-tag[^"]*"[^>]*>(.*?)</a>', page_html, re.S)]
    chapters = []
    for match in re.finditer(r'<tr(?P<attrs>[^>]*)>(?P<body>.*?)</tr>', page_html, re.S):
        attrs = match.group("attrs")
        if "chapter-row" not in attrs:
            continue
        url_match = re.search(r'data-url="([^"]+)"', attrs)
        if not url_match:
            continue
        chapter_url = absolutize_url(url_match.group(1))
        cell = match.group("body")
        title_match = re.search(r'<td>\s*<a[^>]+>(.*?)</a>', cell, re.S)
        date_match = re.search(r'<time[^>]+datetime="([^"]+)"', cell, re.S)
        chapters.append(
            {
                "title": clean_html_text(title_match.group(1)) if title_match else f"Chapter {len(chapters) + 1}",
                "url": chapter_url,
                "date": html.unescape(date_match.group(1)) if date_match else "",
            }
        )

    return {
        "ok": True,
        "source": "royalroad",
        "title": title,
        "author": author,
        "url": url,
        "cover": cover,
        "summary": description,
        "tags": tags[:16],
        "chapters": chapters,
    }


def parse_royalroad_chapter(page_html: str, url: str) -> dict:
    title_match = re.search(r'<h1[^>]*class="[^"]*font-white[^"]*"[^>]*>(.*?)</h1>', page_html, re.S)
    fiction_match = re.search(r'<h2[^>]*class="[^"]*font-white[^"]*"[^>]*>(.*?)</h2>', page_html, re.S)
    body_match = re.search(
        r'<div class="chapter-inner chapter-content">\s*(.*?)(?:<div class="portlet light d|<div class="portlet solid author-note|<div class="portlet-footer)',
        page_html,
        re.S,
    )
    if not body_match:
        raise ValueError("Could not find chapter content on this RoyalRoad page.")
    text = clean_html_text(body_match.group(1))
    if len(text) < 200:
        raise ValueError("Extracted chapter text is too short.")
    return {
        "ok": True,
        "source": "royalroad",
        "title": clean_html_text(title_match.group(1)) if title_match else "RoyalRoad chapter",
        "novel": clean_html_text(fiction_match.group(1)) if fiction_match else "",
        "url": url,
        "text": text,
        "chars": len(text),
    }


def get_value(value, default=""):
    if isinstance(value, dict):
        if "value" in value:
            return value.get("value") or default
        if "units" in value:
            return value.get("units") or default
    return value if value not in (None, "") else default


def get_int(value, default: int = 0) -> int:
    value = get_value(value, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_wuxiaworld_state(page_html: str) -> dict:
    marker = "window.__REACT_QUERY_STATE__ = "
    start = page_html.find(marker)
    if start < 0:
        raise ValueError("Could not find WuxiaWorld page data.")
    start += len(marker)
    end = page_html.find("window.__APP_CONTEXT__", start)
    if end < 0:
        script_end = page_html.find("</script>", start)
        end = script_end if script_end >= 0 else len(page_html)
    blob = page_html[start:end].strip()
    if blob.endswith(";"):
        blob = blob[:-1].strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError("Could not parse WuxiaWorld page data.") from exc


def find_wuxiaworld_query(page_html: str, *prefix) -> dict:
    state = parse_wuxiaworld_state(page_html)
    for query in state.get("queries", []):
        key = query.get("queryKey") or []
        if len(key) >= len(prefix) and all(key[index] == value for index, value in enumerate(prefix)):
            return query.get("state", {}).get("data", {}) or {}
    raise ValueError(f"Could not find WuxiaWorld data for {prefix[0] if prefix else 'page'}.")


def wuxiaworld_novel_from_item(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    if item.get("oneofKind") == "novel" and isinstance(item.get("novel"), dict):
        return item["novel"]
    series_item = (((item.get("series") or {}).get("item") or {}) if isinstance(item.get("series"), dict) else {})
    if isinstance(series_item, dict) and series_item.get("oneofKind") == "novel":
        return series_item.get("novel") or {}
    if isinstance(item.get("item"), dict):
        return wuxiaworld_novel_from_item(item["item"])
    if item.get("slug") and item.get("name"):
        return item
    return {}


def wuxiaworld_free_chapter_count(novel: dict, chapter_count: int) -> int:
    counts = []
    series = novel.get("series") if isinstance(novel.get("series"), dict) else {}
    pricing_models = list(series.get("activePricingModels") or [])
    if isinstance(series.get("pricingModel"), dict):
        pricing_models.append(series["pricingModel"])
    karma = novel.get("karmaInfo") if isinstance(novel.get("karmaInfo"), dict) else {}
    if get_int(karma.get("maxFreeChapter")):
        counts.append(get_int(karma.get("maxFreeChapter")))
    for model in pricing_models:
        pricing = model.get("pricing") if isinstance(model, dict) else {}
        if not isinstance(pricing, dict):
            continue
        oneof = pricing.get("oneofKind")
        pricing_data = pricing.get(oneof) if oneof else None
        if isinstance(pricing_data, dict):
            free_chapters = get_int(pricing_data.get("freeChapters"))
            if free_chapters:
                counts.append(free_chapters)
        if oneof == "free":
            counts.append(chapter_count)
    return max([0, *counts])


def wuxiaworld_card(novel: dict) -> dict:
    title = str(novel.get("name") or "WuxiaWorld novel")
    slug = str(novel.get("slug") or "").strip("/")
    chapter_count = get_int((novel.get("chapterInfo") or {}).get("chapterCount")) if isinstance(novel.get("chapterInfo"), dict) else 0
    author = str(get_value(novel.get("authorName")) or "")
    translator = str(get_value(novel.get("translatorName")) or "")
    tags = [str(tag) for tag in (novel.get("genres") or novel.get("tags") or []) if tag]
    stats = []
    if chapter_count:
        stats.append(f"{chapter_count} chapters")
    if author:
        stats.append(f"Author: {author}")
    if translator:
        stats.append(f"Translator: {translator}")
    return {
        "source": "wuxiaworld",
        "title": title,
        "url": absolutize_url(f"/novel/{slug}", WUXIAWORLD_BASE),
        "cover": str(get_value(novel.get("coverUrl")) or ""),
        "summary": clean_html_text(str(get_value(novel.get("synopsis")) or ""))[:900],
        "tags": tags[:8],
        "stats": stats[:6],
    }


def parse_wuxiaworld_cards(page_html: str, query: str = "", limit: int = 30) -> list[dict]:
    state = parse_wuxiaworld_state(page_html)
    items = []
    seen = set()
    query = query.casefold().strip()
    for entry in state.get("queries", []):
        data = entry.get("state", {}).get("data", {}) or {}
        raw_items = list(data.get("items", []) or [])
        for page in data.get("pages", []) or []:
            if isinstance(page, dict):
                raw_items.extend(page.get("items", []) or [])
        for raw_item in raw_items:
            novel = wuxiaworld_novel_from_item(raw_item)
            if not novel:
                continue
            title = str(novel.get("name") or "")
            slug = str(novel.get("slug") or "")
            if not title or not slug or (query and query not in title.casefold()):
                continue
            url = absolutize_url(f"/novel/{slug}", WUXIAWORLD_BASE)
            if url in seen:
                continue
            seen.add(url)
            items.append(wuxiaworld_card(novel))
            if len(items) >= limit:
                return items
    return items


def parse_wuxiaworld_detail(page_html: str, url: str) -> dict:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1]
    data = find_wuxiaworld_query(page_html, "novel", slug)
    novel = wuxiaworld_novel_from_item(data.get("item") or data)
    if not novel:
        raise ValueError("Could not find WuxiaWorld novel details.")

    chapter_info = novel.get("chapterInfo") if isinstance(novel.get("chapterInfo"), dict) else {}
    chapter_count = get_int(chapter_info.get("chapterCount"))
    first = chapter_info.get("firstChapter") if isinstance(chapter_info.get("firstChapter"), dict) else {}
    latest = chapter_info.get("latestChapter") if isinstance(chapter_info.get("latestChapter"), dict) else {}
    first_offset = get_int(first.get("offset") or first.get("number"), 1)
    latest_offset = get_int(latest.get("offset") or latest.get("number"), chapter_count)
    public_count = wuxiaworld_free_chapter_count(novel, chapter_count)
    public_count = min(chapter_count or public_count, public_count or chapter_count or latest_offset)

    anchor_chapters = parse_wuxiaworld_anchor_chapters(page_html, slug)
    chapters = anchor_chapters if len(anchor_chapters) > 1 else []
    first_slug = str(first.get("slug") or "")
    slug_prefix = ""
    match = re.match(r"^(.*?)(\d+)$", first_slug)
    if chapters:
        pass
    elif match:
        slug_prefix = match.group(1)
    if not chapters and slug_prefix and public_count:
        for number in range(first_offset, public_count + 1):
            chapter_slug = f"{slug_prefix}{number}"
            if number == first_offset and first.get("name"):
                title = str(first["name"])
            elif number == latest_offset and latest.get("name"):
                title = str(latest["name"])
            else:
                title = f"Chapter {number}"
            chapters.append(
                {
                    "title": title,
                    "url": absolutize_url(f"/novel/{slug}/{chapter_slug}", WUXIAWORLD_BASE),
                    "date": "",
                }
            )
    elif not chapters:
        for chapter in (first, latest):
            chapter_slug = chapter.get("slug")
            if not chapter_slug:
                continue
            chapters.append(
                {
                    "title": str(chapter.get("name") or f"Chapter {len(chapters) + 1}"),
                    "url": absolutize_url(f"/novel/{slug}/{chapter_slug}", WUXIAWORLD_BASE),
                    "date": "",
                }
            )

    return {
        "ok": True,
        "source": "wuxiaworld",
        "title": str(novel.get("name") or "WuxiaWorld novel"),
        "author": str(get_value(novel.get("authorName")) or ""),
        "url": url,
        "cover": str(get_value(novel.get("coverUrl")) or ""),
        "summary": clean_html_text(str(get_value(novel.get("synopsis")) or "")),
        "tags": [str(tag) for tag in (novel.get("genres") or novel.get("tags") or []) if tag][:16],
        "chapters": chapters,
    }


def parse_wuxiaworld_chapter(page_html: str, url: str) -> dict:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) < 3:
        raise ValueError("Unsupported WuxiaWorld chapter URL.")
    novel_slug, chapter_slug = path_parts[1], path_parts[2]
    chapter = {}
    try:
        data = find_wuxiaworld_query(page_html, "chapter", novel_slug, chapter_slug)
        chapter = data.get("item") if isinstance(data.get("item"), dict) else data
    except ValueError:
        chapter = {}
    content = str(get_value(chapter.get("content")) or "")
    if not content:
        content = extract_wuxiaworld_chapter_content_div(page_html)
    if not content:
        pricing = chapter.get("pricingInfo") if isinstance(chapter.get("pricingInfo"), dict) else {}
        if pricing and pricing.get("isFree") is False:
            raise ValueError("This WuxiaWorld chapter appears locked and did not expose public text.")
        raise ValueError("Could not find public WuxiaWorld chapter content.")
    text = clean_html_text(content)
    if len(text) < 200:
        raise ValueError("Extracted WuxiaWorld chapter text is too short.")
    novel_info = chapter.get("novelInfo") if isinstance(chapter.get("novelInfo"), dict) else {}
    return {
        "ok": True,
        "source": "wuxiaworld",
        "title": str(chapter.get("name") or "WuxiaWorld chapter"),
        "novel": str(novel_info.get("name") or ""),
        "url": url,
        "text": text,
        "chars": len(text),
    }


def parse_wuxiaworld_anchor_chapters(page_html: str, novel_slug: str) -> list[dict]:
    chapters = []
    seen = set()
    pattern = re.compile(r'<a\b(?P<attrs>[^>]*\bhref="(?P<href>/novel/%s/[^"]+)"[^>]*)>(?P<title>.*?)</a>' % re.escape(novel_slug), re.S)
    for match in pattern.finditer(page_html):
        href = html.unescape(match.group("href"))
        if href in seen:
            continue
        seen.add(href)
        title = clean_html_text(match.group("title")) or href.rstrip("/").split("/")[-1]
        chapters.append(
            {
                "title": title,
                "url": absolutize_url(href, WUXIAWORLD_BASE),
                "date": "",
            }
        )
    return chapters


def extract_wuxiaworld_chapter_content_div(page_html: str) -> str:
    match = re.search(r'<div[^>]+id=["\']chapter-content["\'][^>]*>(.*?)</div>\s*(?:</div>|<script|<footer)', page_html, re.S | re.I)
    return match.group(1) if match else ""


class KokoroState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pipelines: dict[str, KPipeline] = {}
        self.lock = threading.Lock()
        self.import_lock = threading.Lock()
        self.imported_document: dict | None = None

    @property
    def loaded(self) -> bool:
        return bool(self.pipelines)

    def get_pipeline(self, lang_code: str) -> KPipeline:
        if lang_code not in {"a", "b"}:
            raise ValueError("Only American English (a) and British English (b) are configured.")

        if lang_code in self.pipelines:
            return self.pipelines[lang_code]

        with self.lock:
            if lang_code in self.pipelines:
                return self.pipelines[lang_code]

            print(f"Loading Kokoro pipeline lang={lang_code} device={self.args.device}", flush=True)
            pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=self.args.device)
            self.pipelines[lang_code] = pipeline
            print("Kokoro pipeline loaded.", flush=True)
            return pipeline

    def speak(self, payload: dict) -> tuple[bytes, dict]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Text is required.")

        voice = str(payload.get("voice") or "af_heart")
        if voice not in VOICES:
            raise ValueError(f"Unsupported voice: {voice}")

        lang_code = str(payload.get("lang_code") or self.args.lang_code)
        speed = float(payload.get("speed") or self.args.speed)
        speed = max(0.6, min(1.5, speed))

        pipeline = self.get_pipeline(lang_code)
        started = time.perf_counter()
        chunks = []
        with self.lock:
            for result in pipeline(text, voice=voice, speed=speed):
                chunks.append(np.asarray(result.audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")

        wav = np.concatenate(chunks)
        audio = io.BytesIO()
        sf.write(audio, wav, 24000, format="WAV", subtype="PCM_16")
        elapsed = time.perf_counter() - started
        metadata = {
            "sample_rate": 24000,
            "seconds": len(wav) / 24000,
            "elapsed": elapsed,
            "voice": voice,
            "lang_code": lang_code,
            "speed": speed,
            "peak": float(np.max(np.abs(wav))) if wav.size else 0.0,
            "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0,
        }
        return audio.getvalue(), metadata

    def set_imported_document(self, payload: dict) -> dict:
        title = str(payload.get("title", "")).strip()[:240]
        text = str(payload.get("text", "")).strip()
        url = str(payload.get("url", "")).strip()[:1000]
        source = str(payload.get("source", "browser")).strip()[:80]
        if not text:
            raise ValueError("Imported text is empty.")
        if len(text) > 2_000_000:
            raise ValueError("Imported text is too large.")

        document = {
            "ok": True,
            "title": title or "Imported chapter",
            "text": text,
            "url": url,
            "source": source,
            "chars": len(text),
            "imported_at": time.time(),
        }
        with self.import_lock:
            self.imported_document = document
        return {key: value for key, value in document.items() if key != "text"}

    def get_imported_document(self) -> dict:
        with self.import_lock:
            if not self.imported_document:
                return {"ok": False, "error": "No browser import is available yet."}
            return self.imported_document

    def source_feed(self, source: str, kind: str) -> dict:
        if source == "wuxiaworld":
            feed_path = "/" if (kind or "trending") == "trending" else "/novels"
            page = fetch_public_html(feed_path, WUXIAWORLD_BASE)
            return {"ok": True, "source": "wuxiaworld", "kind": kind or "trending", "items": parse_wuxiaworld_cards(page)}
        if source != "royalroad":
            raise ValueError(f"Unsupported source: {source}")
        feed_path = {
            "trending": "/fictions/trending",
            "latest": "/fictions/latest-updates",
            "best": "/fictions/best-rated",
            "complete": "/fictions/complete",
        }.get(kind or "trending", "/fictions/trending")
        page = fetch_public_html(feed_path)
        return {"ok": True, "source": "royalroad", "kind": kind or "trending", "items": parse_royalroad_cards(page)}

    def source_search(self, source: str, query: str) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("Search query is required.")
        if source == "wuxiaworld":
            page = fetch_public_html("/novels", WUXIAWORLD_BASE)
            return {"ok": True, "source": "wuxiaworld", "query": query, "items": parse_wuxiaworld_cards(page, query)}
        if source != "royalroad":
            raise ValueError(f"Unsupported source: {source}")
        page = fetch_public_html(f"/fictions/search?title={quote_plus(query)}")
        return {"ok": True, "source": "royalroad", "query": query, "items": parse_royalroad_cards(page)}

    def source_novel(self, source: str, url: str) -> dict:
        if source == "wuxiaworld":
            full_url = absolutize_url(url, WUXIAWORLD_BASE)
            path = urlparse(full_url).path
            if not re.match(r"^/novel/[^/]+/?$", path):
                raise ValueError("Unsupported WuxiaWorld novel URL.")
            page = fetch_public_html(full_url, WUXIAWORLD_BASE)
            return parse_wuxiaworld_detail(page, full_url)
        if source != "royalroad":
            raise ValueError(f"Unsupported source: {source}")
        full_url = absolutize_url(url)
        path = urlparse(full_url).path
        if not re.match(r"^/fiction/\d+/", path):
            raise ValueError("Unsupported RoyalRoad novel URL.")
        page = fetch_public_html(full_url)
        return parse_royalroad_detail(page, full_url)

    def source_chapter(self, source: str, url: str) -> dict:
        if source == "wuxiaworld":
            full_url = absolutize_url(url, WUXIAWORLD_BASE)
            path = urlparse(full_url).path
            if not re.match(r"^/novel/[^/]+/[^/]+/?$", path):
                raise ValueError("Unsupported WuxiaWorld chapter URL.")
            page = fetch_public_html(full_url, WUXIAWORLD_BASE)
            return parse_wuxiaworld_chapter(page, full_url)
        if source != "royalroad":
            raise ValueError(f"Unsupported source: {source}")
        full_url = absolutize_url(url)
        path = urlparse(full_url).path
        if not re.match(r"^/fiction/\d+/.+/chapter/\d+/", path):
            raise ValueError("Unsupported RoyalRoad chapter URL.")
        page = fetch_public_html(full_url)
        return parse_royalroad_chapter(page, full_url)


class Handler(BaseHTTPRequestHandler):
    state: KokoroState

    def _send_bytes(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if headers:
            for name, value in headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: dict) -> None:
        self._send_bytes(status, json.dumps(data).encode("utf-8"), "application/json")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/":
            html_path = ROOT / "kokoro_reader.html"
            html = html_path.read_text(encoding="utf-8") if html_path.exists() else INDEX_HTML
            html = html.replace("__VOICES__", json.dumps(VOICES)).replace("__LANGUAGES__", json.dumps(LANGUAGES))
            self._send_bytes(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "engine": "kokoro",
                    "pipeline_loaded": self.state.loaded,
                    "device": self.state.args.device,
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
            )
            return

        if path == "/api/tone":
            self._send_bytes(HTTPStatus.OK, make_test_tone(), "audio/wav")
            return

        if path == "/api/imported":
            self._send_json(HTTPStatus.OK, self.state.get_imported_document())
            return

        try:
            if path == "/api/sources":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "sources": [
                            {"id": "royalroad", "label": "RoyalRoad", "supports": ["feed", "search", "novel", "chapter"]},
                            {"id": "wuxiaworld", "label": "WuxiaWorld", "supports": ["feed", "search", "novel", "chapter"]},
                        ],
                    },
                )
                return
            if path == "/api/source/feed":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.source_feed(query.get("source", ["royalroad"])[0], query.get("kind", ["trending"])[0]),
                )
                return
            if path == "/api/source/search":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.source_search(query.get("source", ["royalroad"])[0], query.get("q", [""])[0]),
                )
                return
            if path == "/api/source/novel":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.source_novel(query.get("source", ["royalroad"])[0], query.get("url", [""])[0]),
                )
                return
            if path == "/api/source/chapter":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.source_chapter(query.get("source", ["royalroad"])[0], query.get("url", [""])[0]),
                )
                return
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(error)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/speak", "/api/imported"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                payload = json.loads(raw.decode("utf-8") or "{}")
            else:
                payload = {key: values[0] for key, values in parse_qs(raw.decode("utf-8")).items()}

            if path == "/api/imported":
                self._send_json(HTTPStatus.OK, self.state.set_imported_document(payload))
                return

            audio, metadata = self.state.speak(payload)
            headers = {"X-TTS-Metadata": json.dumps(metadata)}
            self._send_bytes(HTTPStatus.OK, audio, "audio/wav", headers=headers)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "CUDA ran out of memory. Try less text or --device cpu.", "details": str(error)},
            )
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local browser server for Kokoro TTS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lang-code", default="a", choices=["a", "b"])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--preload", action="store_true", help="Load the default pipeline before accepting requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    HF_HOME.mkdir(exist_ok=True)
    state = KokoroState(args)

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.preload:
        state.get_pipeline(args.lang_code)

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
