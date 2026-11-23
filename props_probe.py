"""Can the publish jobs read public prop lines from DFS pick'em sites and
sportsbooks? No logins: public endpoints only. Prints status, size and a
sample per source; never prints or needs credentials.

    python props_probe.py
"""
import json
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
      "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"}
SOURCES = {
    "PrizePicks NBA": "https://api.prizepicks.com/projections?league_id=7&per_page=1000&single_stat=true&game_mode=pickem",
    "PrizePicks NFL": "https://api.prizepicks.com/projections?league_id=9&per_page=1000&single_stat=true&game_mode=pickem",
    "PrizePicks NHL": "https://api.prizepicks.com/projections?league_id=8&per_page=1000&single_stat=true&game_mode=pickem",
    "Underdog (all)": "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "Underdog v6": "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "DraftKings SB NBA": "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42648",
    "FanDuel SB NBA": "https://sbapi.nj.sportsbook.fanduel.com/api/content-managed-page?page=CUSTOM&customPageId=nba&_ak=FhMFpcPWXMeyZxOx",
    "Pick6 (DK)": "https://pick6.draftkings.com/api/v1/sports",
}


def stats_sample(name, js):
    try:
        if name.startswith("PrizePicks"):
            data = js.get("data") or []
            types = {}
            for d in data:
                t = (d.get("attributes") or {}).get("stat_type")
                types[t] = types.get(t, 0) + 1
            return f"{len(data)} lines; stat types: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items(), key=lambda x: -x[1])[:15])
        if name.startswith("Underdog"):
            lines = js.get("over_under_lines") or []
            types = {}
            for l in lines:
                t = ((l.get("over_under") or {}).get("appearance_stat") or {}).get("display_stat")
                types[t] = types.get(t, 0) + 1
            return f"{len(lines)} lines; stat types: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items(), key=lambda x: -x[1])[:15])
    except Exception as exc:                                   # noqa: BLE001
        return f"parse failed: {exc}"
    return "keys: " + ", ".join(list(js.keys())[:12]) if isinstance(js, dict) else type(js).__name__


for name, url in SOURCES.items():
    try:
        r = requests.get(url, headers=UA, timeout=30)
        body = r.text
        line = f"{name}: HTTP {r.status_code}, {len(body):,} bytes"
        if r.ok:
            try:
                line += " | " + stats_sample(name, r.json())
            except ValueError:
                line += " | not JSON: " + body[:120].replace("\n", " ")
        else:
            line += " | " + body[:160].replace("\n", " ")
    except Exception as exc:                                   # noqa: BLE001
        line = f"{name}: request failed ({str(exc)[:120]})"
    print(line, flush=True)
