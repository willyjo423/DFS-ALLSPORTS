"""(1) Every FanDuel player-prop market type per sport (all tabs the event
page offers), with a count. (2) The odds API plan's remaining credits, read
from a free call (/v4/sports costs nothing); never prints the key."""
import os, re, requests, time
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
      "Accept": "application/json"}
AK = "FhMFpcPWXMeyZxOx"
BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
for sport in ("nba", "nhl", "nfl"):
    js = requests.get(f"{BASE}/content-managed-page", params={"page": "CUSTOM", "customPageId": sport, "_ak": AK}, headers=UA, timeout=30).json()
    ev = sorted([e for e in (js.get("attachments") or {}).get("events", {}).values() if " @ " in str(e.get("name")) and "(" not in str(e.get("name"))],
                key=lambda e: str(e.get("openDate")))
    if not ev:
        print("==", sport, "no games"); continue
    e = ev[0]
    main = requests.get(f"{BASE}/event-page", params={"eventId": e["eventId"], "_ak": AK}, headers=UA, timeout=30).json()
    tabs = [t.get("title") or t.get("id") for t in ((main.get("layout") or {}).get("tabs") or {}).values()] if isinstance((main.get("layout") or {}).get("tabs"), dict) else []
    tabids = []
    for t in ((main.get("layout") or {}).get("tabs") or {}).values() if isinstance((main.get("layout") or {}).get("tabs"), dict) else []:
        tabids.append(str(t.get("title") or "").lower().replace(" ", "-").replace("/", "-"))
    print(f"== {sport}: {e['name']} | tabs: {tabids}")
    types = {}
    for tab in [""] + tabids:
        p = {"eventId": e["eventId"], "_ak": AK}
        if tab: p["tab"] = tab
        try:
            r = requests.get(f"{BASE}/event-page", params=p, headers=UA, timeout=30)
            for m in ((r.json().get("attachments") or {}).get("markets") or {}).values():
                t = re.sub(r"PLAYER_[A-Z]_", "PLAYER_*_", str(m.get("marketType")))
                types[t] = types.get(t, 0) + 1
        except Exception as exc:
            print("  tab", tab, "failed", str(exc)[:60])
        time.sleep(0.2)
    print("  " + ", ".join(sorted(t for t in types if not any(x in t for x in ("PERIOD", "QUARTER", "HALF", "DRIVE", "1ST_", "2ND_", "3RD_", "4TH_")))))
key = os.environ.get("ODDS_API_KEY")
if key:
    r = requests.get("https://api.the-odds-api.com/v4/sports", params={"apiKey": key}, timeout=30)
    print("ODDS API:", r.status_code, "used", r.headers.get("x-requests-used"), "remaining", r.headers.get("x-requests-remaining"))
else:
    print("ODDS API: no key in this job")
