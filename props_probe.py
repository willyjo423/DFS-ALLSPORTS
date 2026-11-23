"""FanDuel sportsbook (public, no login): sample player-prop markets per sport
- market type, name, and a few runners with handicap and American odds."""
import json, requests
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
      "Accept": "application/json"}
AK = "FhMFpcPWXMeyZxOx"
BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
TABS = {"nfl": ["", "passing-props", "receiving-props", "rushing-props", "td-scorer-props"],
        "nhl": ["", "points-assists", "shots", "goalie-props", "goal-scorer"],
        "nba": ["", "player-points", "player-rebounds", "player-assists", "player-threes", "player-combos", "player-defense"]}
KEEP = ("PLAYER", "TOUCHDOWN", "GOAL_SCORER", "SAVES", "SHOTS", "POINTS", "REBOUND", "ASSIST", "THREE", "BLOCK", "STEAL", "RECEPTION", "YARDS")
for sport, tabs in TABS.items():
    js = requests.get(f"{BASE}/content-managed-page", params={"page": "CUSTOM", "customPageId": sport, "_ak": AK}, headers=UA, timeout=30).json()
    ev = sorted([x for x in (js.get("attachments") or {}).get("events", {}).values() if " @ " in str(x.get("name"))],
                key=lambda x: str(x.get("openDate")))
    ev = [x for x in ev if "(" not in str(x.get("name"))]
    if not ev:
        print("==", sport, "no games"); continue
    e = ev[0]
    print(f"== {sport}: {e['name']} {e['openDate']} eventId {e['eventId']}")
    seen = {}
    for tab in tabs:
        p = {"eventId": e["eventId"], "_ak": AK}
        if tab: p["tab"] = tab
        r = requests.get(f"{BASE}/event-page", params=p, headers=UA, timeout=30)
        if not r.ok:
            print("  tab", tab, r.status_code); continue
        for m in ((r.json().get("attachments") or {}).get("markets") or {}).values():
            t = str(m.get("marketType"))
            if t in seen or not any(k in t for k in KEEP) or "PERIOD" in t or "QUARTER" in t or "HALF" in t or "DRIVE" in t:
                continue
            seen[t] = 1
            runs = []
            for rr in (m.get("runners") or [])[:4]:
                odds = (((rr.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOdds"))
                runs.append(f"{rr.get('runnerName')} | hc={rr.get('handicap')} | {odds}")
            print(f"  [{tab}] {t} :: {m.get('marketName')} :: " + " || ".join(runs))
