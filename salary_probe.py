"""Is historical DraftKings salary data reachable? Variants of RotoGuru pages."""
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
C = {
    "nba_scsv_2019-01-15": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=1&day=15&year=2019&scsv=1",
    "nba_scsv_2021-03-10": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=3&day=10&year=2021&scsv=1",
    "nba_scsv_2017-12-05": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=12&day=5&year=2017&scsv=1",
    "nfl_scsv_2021_w10": "http://rotoguru1.com/cgi-bin/fyday.pl?week=10&year=2021&game=dk&scsv=1",
    "nfl_scsv_2018_w5": "http://rotoguru1.com/cgi-bin/fyday.pl?week=5&year=2018&game=dk&scsv=1",
}
for k, u in C.items():
    try:
        r = requests.get(u, headers=UA, timeout=30)
        t = r.text
        rows = [l for l in re.sub(r"<[^>]+>", "\n", t).splitlines() if l.count(";") >= 5]
        sal = re.findall(r"\$\d{1,2},\d{3}", t)
        print(f"=== {k}: HTTP {r.status_code} {len(t)}B, scsv rows {len(rows)}, $salaries {len(sal)}")
        for l in rows[:4]:
            print("   ", l[:200])
        if not rows:
            body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))
            i = body.find("Salary")
            print("   ", body[max(0, i - 300):i + 600])
    except Exception as exc:
        print(f"=== {k}: ERROR {exc}")
