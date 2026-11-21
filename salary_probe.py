"""Show raw table rows of a RotoGuru NBA DraftKings day page."""
import re
import requests
UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
t = requests.get("http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=1&day=15&year=2019", headers=UA, timeout=30).text
rows = re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S | re.I)
sal = [r for r in rows if re.search(r"\$\d{1,2},\d{3}", r)]
print(len(rows), "rows,", len(sal), "with salaries")
for r in sal[:3] + sal[-2:]:
    print("RAW:", r[:600])
    cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S | re.I)]
    print("CELLS:", cells)
