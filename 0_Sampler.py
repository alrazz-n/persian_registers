import json
import os
import random
import urllib.request
import zstandard as zstd

# Configuration
LINES_PER_FILE = 7000 #20000
SAMPLES_PER_FILE = 700 #20000
MAX_TOTAL_SAMPLES = 2000
MAP_URL = "https://data.hplt-project.org/three/sorted/pes_Arab.map"

WEBREG_FILTER_KEYS = ["sr"] #can be anything, do each one sepratly
WEBREG_THRESHOLD_1 = 0.4 #used to be 0.8 but it was so certain
WEBREG_THRESHOLD_2 = 0.2 #add after 

webreg_suffix = "-".join(WEBREG_FILTER_KEYS)


job_id = os.getenv("SLURM_JOB_ID", "manual")


OUTPUT_DIR = "/scratch/project_2005092/nima/samples"
OUTPUT_FILE = f"{OUTPUT_DIR}/sample_{webreg_suffix}_{job_id}.jsonl"
PROGRESS_FILE = f"{OUTPUT_DIR}/progress_{webreg_suffix}_{job_id}.txt"

blocked_domains = (
    "netgarmi.ir", "netct.ir", "nahamta.ir", "topdars.com", "sibfun.ir",
    "chefsona.com", "30m30.com", "dlkon.ir", "patoghy.ir", "sargarmifull.ir",
    "parsipatogh.ir", "gizmotoon.ir", "1tvmarket.ir", "campec.ir",
    "radnetworking.com", "funtarin.ir", "azcloob.ir", "cnnic.in",
    "avaliha.ir", "ahang.bottega--veneta.net", "film-serial1.b19.ir",
    "vazeh.com", "sbmstudio.biz", "kancelaria-radomsko.pl", "sirooz.com",
    "foodbaran.com", "jafo.ir", "irancloob.com", "niazerooz.com",
    "jeje2.blogreader.ir", "softestan.com", "persiankhatoon.com",
    "lindyhopamersfoort.nl", "hofruiters.nl", "sadowahouse.pl",
    "fewo-gruss-aus-partenkirchen.de", "hein-vom-rhein.de", "sidonline",
    "ferienwohnung-stueck.de", "orthopediebosdam.be"
)

def item_contains_blocked_domain(item):
    text = json.dumps(item)
    return any(bad in text for bad in blocked_domains)

def passes_webreg_filter(item):
    if "web-register" not in item:
        return False
    wr = item["web-register"]
    return any(
        key in wr and wr[key] <= WEBREG_THRESHOLD_1 and wr[key] >= WEBREG_THRESHOLD_2  #used to be >= only
        for key in WEBREG_FILTER_KEYS
    )

print("Downloading MAP file…")
urls = [
    u.strip()
    for u in urllib.request.urlopen(MAP_URL).read().decode().split("\n")
    if u.strip()
]

urls = [u for u in urls if not any(bad in u for bad in blocked_domains)]
print(f"Found {len(urls)} valid files after filtering.")

start_index = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        start_index = int(f.read().strip())

total_written = 0

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "a") as output:

    for i in range(start_index, len(urls)):
        if total_written >= MAX_TOTAL_SAMPLES:
            break

        url = urls[i]
        print(f"File {i + 1}/{len(urls)}: {url.split('/')[-1]}")

        # Read at most LINES_PER_FILE lines
        collected = []
        with urllib.request.urlopen(url) as response:
            reader = zstd.ZstdDecompressor().stream_reader(response)
            buffer = b""
            lines_read = 0

            while lines_read < LINES_PER_FILE:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk

                while b"\n" in buffer and lines_read < LINES_PER_FILE:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        item = json.loads(line.decode())
                        lines_read += 1

                        # Immediately filter
                        if item_contains_blocked_domain(item):
                            continue
                        if not passes_webreg_filter(item):
                            continue

                        collected.append(item)

        # Now sample *after* filtering
        if collected:
            take = min(SAMPLES_PER_FILE, len(collected))
            selected = random.sample(collected, take)

            # Enforce global limit
            remaining = MAX_TOTAL_SAMPLES - total_written
            selected = selected[:remaining]

            for item in selected:
                output.write(json.dumps(item) + "\n")
            output.flush()

            total_written += len(selected)
            print(f"  Wrote {len(selected)} items (total = {total_written})")

        # Save progress
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(i + 1))

print(f"Done! Total written: {total_written}")

# Clean up
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
