"""FanDuel sportsbook (public endpoints, no login): which player-prop markets
does one NBA / NHL / NFL game carry? Prints market names and counts."""
import requests
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
      "Accept": "application/json"}
AK = "FhMFpcPWXMeyZxOx"
BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
for sport in ("nba", "nhl", "nfl"):
    try:
        js = requests.get(f"{BASE}/content-managed-page", params={"page": "CUSTOM", "customPageId": sport, "_ak": AK},
                          headers=UA, timeout=30).json()
        ev = list((js.get("attachments") or {}).get("events", {}).values())
        print(f"== {sport}: {len(ev)} events on the page", flush=True)
        if not ev:
            continue
        e = ev[0]
        print("   sample event:", e.get("name"), e.get("openDate"))
        names = {}
        for tab in ("", "player-points", "player-props", "popular", "player-rebounds", "player-assists",
                    "player-threes", "shots", "points-assists", "passing-props", "receiving-props", "rushing-props", "td-scorer-props"):
            p = {"eventId": e["eventId"], "_ak": AK}
            if tab:
                p["tab"] = tab
            r = requests.get(f"{BASE}/event-page", params=p, headers=UA, timeout=30)
            if not r.ok:
                print(f"   tab {tab or '(main)'}: HTTP {r.status_code}"); continue
            mk = (r.json().get("attachments") or {}).get("markets") or {}
            for m in mk.values():
                n = m.get("marketType") or m.get("marketName")
                names[n] = names.get(n, 0) + 1
        print(f"   {len(names)} market types; with player in name/type:")
        print("   " + ", ".join(f"{k}" for k in sorted(names) if any(w in str(k).upper() for w in
              ("PLAYER", "POINTS", "REBOUND", "ASSIST", "THREE", "SHOTS", "BLOCK", "STEAL", "RECEPTION", "YARDS", "TD", "TOUCHDOWN", "SAVES", "GOAL")))[:1800])
    except Exception as exc:  # noqa: BLE001
        print(sport, "failed:", str(exc)[:200])
