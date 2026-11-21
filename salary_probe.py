"""Is historical DraftKings salary data reachable? Prints the first lines of
each candidate RotoGuru daily/weekly page (semicolon-CSV mode)."""
import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
CANDIDATES = {
    "nba_dk_2024-01-15": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=1&day=15&year=2024&scsv=1",
    "nba_dk_2022-12-01": "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=12&day=1&year=2022&scsv=1",
    "nhl_dk_2024-01-15_hoday": "http://rotoguru1.com/cgi-bin/hoday.pl?game=dk&mon=1&day=15&year=2024&scsv=1",
    "nhl_dk_2024-01-15_hkyday": "http://rotoguru1.com/cgi-bin/hkyday.pl?game=dk&mon=1&day=15&year=2024&scsv=1",
    "nfl_dk_2023_w5": "http://rotoguru1.com/cgi-bin/fyday.pl?week=5&year=2023&game=dk&scsv=1",
    "nfl_dk_2024_w10": "http://rotoguru1.com/cgi-bin/fyday.pl?week=10&year=2024&game=dk&scsv=1",
    "mlb_dk_2024-06-01": "http://rotoguru1.com/cgi-bin/byday.pl?game=dk&mon=6&day=1&year=2024&scsv=1",
}
for k, u in CANDIDATES.items():
    try:
        r = requests.get(u, headers=UA, timeout=30)
        txt = r.text
        i = txt.find("Date;") if "Date;" in txt else txt.find("Week;")
        body = txt[i:i + 700] if i >= 0 else txt[:400]
        print(f"=== {k}: HTTP {r.status_code}, {len(txt)} bytes, semicolon rows {txt.count(';')}")
        print(body.replace("<br>", "\n")[:700])
    except Exception as exc:
        print(f"=== {k}: ERROR {exc}")
