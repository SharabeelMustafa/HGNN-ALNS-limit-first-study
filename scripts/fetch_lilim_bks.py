"""Scrape Li & Lim best-known solutions (vehicles, distance) from SINTEF TOP pages -> data/bks/lilim_bks.csv"""
import csv, html, re, sys, urllib.request

URL = "https://www.sintef.no/projectweb/top/pdptw/{}-customers/"
rows = []
for size in [100, 200, 400, 600, 800, 1000]:
    req = urllib.request.Request(URL.format(size), headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(req, timeout=60).read().decode("utf8", "ignore")
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if len(cells) >= 3 and re.match(r"^l[cr]", cells[0], re.I):
            name = cells[0].rstrip("s").strip()
            try:
                rows.append({"size": size, "instance": name, "vehicles": int(cells[1]), "distance": float(cells[2]),
                             "reference": cells[3] if len(cells) > 3 else "", "date": cells[4] if len(cells) > 4 else ""})
            except ValueError:
                pass
with open(sys.argv[1] if len(sys.argv) > 1 else "data/bks/lilim_bks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["size", "instance", "vehicles", "distance", "reference", "date"])
    w.writeheader(); w.writerows(rows)
print(len(rows), "BKS rows")
