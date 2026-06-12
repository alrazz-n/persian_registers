import json
import os
import random
import urllib.request
import zstandard as zstd

# Configuration
LINES_PER_FILE = 100000
SAMPLES_PER_FILE = 10000
MAX_TOTAL_SAMPLES = 30000
MAP_URL = "https://data.hplt-project.org/three/sorted/pes_Arab.map"

#label searching for
WEBREG_FILTER_KEYS = ["SP"]
WEBREG_THRESHOLD_1 = 0.40
WEBREG_THRESHOLD_2 = 0.36

#label/s to avoid
OTHER_LABELS = ["ne", "it"]  # Add as much as label needed
OTHER_LABEL_MAX = 0.25


webreg_suffix = "-".join(WEBREG_FILTER_KEYS)

job_id = os.getenv("SLURM_JOB_ID", "manual")

OUTPUT_DIR = "/scratch/project_2005092/nima/samples"
OUTPUT_FILE = f"{OUTPUT_DIR}/sample_{webreg_suffix}_{job_id}.jsonl"
PROGRESS_FILE = f"{OUTPUT_DIR}/progress_{webreg_suffix}_{job_id}.txt"

blocked_domains = (
    "netgarmi", "netct.ir", "nahamta.ir", "topdars.com", "sibfun.ir",
    "chefsona.com", "30m30.com", "dlkon.ir", "patoghy.ir", "sargarmifull.ir",
    "parsipatogh.ir", "gizmotoon.ir", "1tvmarket.ir", "campec.ir",
    "radnetworking.com", "funtarin.ir", "azcloob.ir", "cnnic.in",
    "avaliha.ir", "ahang.bottega--veneta.net", "film-serial1.b19.ir",
    "vazeh.com", "sbmstudio.biz", "kancelaria-radomsko.pl", "sirooz.com",
    "foodbaran.com", "jafo.ir", "irancloob.com", "niazerooz.com",
    "jeje2.blogreader.ir", "softestan.com", "persiankhatoon.com",
    "lindyhopamersfoort.nl", "hofruiters.nl", "sadowahouse.pl",
    "fewo-gruss-aus-partenkirchen.de", "hein-vom-rhein.de", "sidonline",
    "ferienwohnung-stueck.de", "orthopediebosdam.be", "kwaekensteyn.nl"
)

def item_contains_blocked_domain(item):
    text = json.dumps(item)
    return any(bad in text for bad in blocked_domains)

def passes_webreg_filter(item):
    if "web-register" not in item:
        return False

    wr = item["web-register"]

    # Main label must be in range
    target_ok = any(
        key in wr and WEBREG_THRESHOLD_2 <= wr[key] <= WEBREG_THRESHOLD_1
        for key in WEBREG_FILTER_KEYS
    )

    if not target_ok:
        return False

    # Other labels must stay below threshold
    others_ok = all(
        wr.get(key, 0.0) < OTHER_LABEL_MAX
        for key in OTHER_LABELS
    )

    return others_ok

print("Downloading MAP file…")
urls = [
    u.strip()
    for u in urllib.request.urlopen(MAP_URL).read().decode().split("\n")
    if u.strip()
]

# Remove blocked domains from URLs
urls = [u for u in urls if not any(bad in u for bad in blocked_domains)]

# Shuffle for more stable runtime
random.shuffle(urls)

print(f"Found {len(urls)} valid files after filtering.")

start_index = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        start_index = int(f.read().strip())

total_written = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "a") as output:

    for i in range(start_index, len(urls)):
        if total_written >= MAX_TOTAL_SAMPLES:
            break

        url = urls[i]
        print(f"File {i + 1}/{len(urls)}: {url.split('/')[-1]}")

        collected = []
        lines_read = 0

        with urllib.request.urlopen(url) as response:
            reader = zstd.ZstdDecompressor().stream_reader(response)
            buffer = b""

            while lines_read < LINES_PER_FILE:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk

                while b"\n" in buffer and lines_read < LINES_PER_FILE:
                    line, buffer = buffer.split(b"\n", 1)

                    if not line.strip():
                        continue

                    item = json.loads(line.decode())
                    lines_read += 1

                    # Filtering
                    if item_contains_blocked_domain(item):
                        continue
                    if not passes_webreg_filter(item):
                        continue

                    # Reservoir sampling
                    if len(collected) < SAMPLES_PER_FILE:
                        collected.append(item)
                    else:
                        j = random.randint(0, lines_read - 1)
                        if j < SAMPLES_PER_FILE:
                            collected[j] = item

                    # Early stop per file
                    if len(collected) >= SAMPLES_PER_FILE:
                        break

                if len(collected) >= SAMPLES_PER_FILE:
                    break

        if collected:
            remaining = MAX_TOTAL_SAMPLES - total_written
            selected = collected[:remaining]

            for item in selected:
                output.write(json.dumps(item) + "\n")

            output.flush()

            total_written += len(selected)
            print(f"  Read {lines_read}, kept {len(collected)}, wrote {len(selected)} (total = {total_written})")
        else:
            print(f"  Read {lines_read}, kept 0")

        # Save progress
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(i + 1))

print(f"Done! Total written: {total_written}")

# Cleanup
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
