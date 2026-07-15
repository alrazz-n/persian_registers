import json
import os
#import random
import urllib.request
import zstandard as zstd

# Configuration
LINES_PER_FILE = None #20000
#SAMPLES_PER_FILE = 2000 #20000
MAX_TOTAL_SAMPLES = 5000
MAP_URL = "https://data.hplt-project.org/three/sorted/pes_Arab.map"



job_id = os.getenv("SLURM_JOB_ID", "manual")


OUTPUT_DIR = "/scratch/project_2005092/nima/samples"
OUTPUT_FILE = f"{OUTPUT_DIR}/sample_JNK_{job_id}.jsonl"
PROGRESS_FILE = f"{OUTPUT_DIR}/progress_JNK_{job_id}.txt"

blocked_domains = (
    "nahamta.ir", "sibfun.ir",
    "chefsona.com", "dlkon.ir", "sargarmifull.ir",
    "gizmotoon.ir", "1tvmarket.ir", "campec.ir",
    "radnetworking.com", "funtarin.ir", "azcloob.ir", "cnnic.in",
    "ahang.bottega--veneta.net", "film-serial1.b19.ir",
    "sbmstudio.biz", "kancelaria-radomsko.pl",
    "foodbaran.com", "jafo.ir", "irancloob.com",
    "jeje2.blogreader.ir","persiankhatoon.com",
    "lindyhopamersfoort.nl", "hofruiters.nl", "sadowahouse.pl",
    "fewo-gruss-aus-partenkirchen.de", "hein-vom-rhein.de", "sidonline",
    "orthopediebosdam.be", "kwaekensteyn.nl"
)

def item_contains_blocked_domain(item):
    text = json.dumps(item).lower()
    return any(bad.lower() in text for bad in blocked_domains)


print("Downloading MAP file…")


urls = [
    u.strip()
    for u in urllib.request.urlopen(MAP_URL).read().decode().split("\n")
    if u.strip()
]

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
        #collected = []
        with urllib.request.urlopen(url) as response:
            reader = zstd.ZstdDecompressor().stream_reader(response)
            buffer = b""
            lines_read = 0

            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)

                    if line.strip():
                        item = json.loads(line.decode())
                        lines_read += 1

                        if item_contains_blocked_domain(item):
                            output.write(json.dumps(item) + "\n")
                            output.flush()

                            total_written += 1
                            print(f"Found {total_written}/{MAX_TOTAL_SAMPLES}")

                            if total_written >= MAX_TOTAL_SAMPLES:
                                break


        # Now sample *after* filtering
        #if collected:
            #remaining = MAX_TOTAL_SAMPLES - total_written
            #to_write = collected[:remaining]

            #for item in to_write:
                #output.write(json.dumps(item) + "\n")
            #output.flush()

            #total_written += len(to_write)
            #print(f"  Wrote {len(to_write)} items (total = {total_written})")

        # Save progress
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(i + 1))

print(f"Done! Total written: {total_written}")

# Clean up
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
