"""Sample markets on specific FanDuel tabs (NHL goalies/goals, NBA combos/defense)."""
import re, requests, time
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36", "Accept": "application/json"}
AK = "FhMFpcPWXMeyZxOx"; BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
for sport, tabs in (("nhl", ("goalies", "goals", "popular")), ("nba", ("player-combos", "player-defense", "player-threes", "popular"))):
    js = requests.get(f"{BASE}/content-managed-page", params={"page": "CUSTOM", "customPageId": sport, "_ak": AK}, headers=UA, timeout=30).json()
    ev = sorted([e for e in (js.get("attachments") or {}).get("events", {}).values() if " @ " in str(e.get("name")) and "(" not in str(e.get("name"))], key=lambda e: str(e.get("openDate")))
    e = ev[0]; print("==", sport, e["name"])
    seen = set()
    for tab in tabs:
        r = requests.get(f"{BASE}/event-page", params={"eventId": e["eventId"], "tab": tab, "_ak": AK}, headers=UA, timeout=30)
        for m in ((r.json().get("attachments") or {}).get("markets") or {}).values():
            t = re.sub(r"PLAYER_[A-Z]_", "PLAYER_*_", str(m.get("marketType")))
            if t in seen or any(x in t for x in ("PERIOD", "QUARTER", "HALF")): continue
            seen.add(t)
            rs = []
            for rr in (m.get("runners") or [])[:3]:
                o = ((rr.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOdds")
                rs.append(f"{rr.get('runnerName')} hc={rr.get('handicap')} {o}")
            print(f"  [{tab}] {t} :: {m.get('marketName')} :: " + " || ".join(rs))
        time.sleep(0.2)
