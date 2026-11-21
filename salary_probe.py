"""Is historical DraftKings salary data reachable? Variants of RotoGuru pages."""
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
C = {
    "nba_html_2024-01-15": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=1&day=15&year=2024",
    "nba_scsv_2026-11-15": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=11&day=15&year=2026&scsv=1",
    "nba_scsv_2025-03-10": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=3&day=10&year=2025&scsv=1",
    "nba_html_2025-03-10": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=3&day=10&year=2025",
    "nfl_html_2024_w10": "http://rotoguru1.com/cgi-bin/fyday.pl?week=10&year=2024&game=dk",
    "nfl_scsv_2025_w10": "http://rotoguru1.com/cgi-bin/fyday.pl?week=10&year=2025&game=dk&scsv=1",
    "nfl_scsv_2026_w10": "http://rotoguru1.com/cgi-bin/fyday.pl?week=10&year=2026&game=dk&scsv=1",
    "nhl_index": "http://rotoguru1.com/hockey/",
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
