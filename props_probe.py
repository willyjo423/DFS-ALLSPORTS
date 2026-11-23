import requests
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36","Accept":"application/json"}
for pid in ("ncaaf","college-football","ncaa-football","cfb"):
    try:
        js=requests.get("https://sbapi.nj.sportsbook.fanduel.com/api/content-managed-page",params={"page":"CUSTOM","customPageId":pid,"_ak":"FhMFpcPWXMeyZxOx"},headers=UA,timeout=30)
        ev=[e for e in (js.json().get("attachments") or {}).get("events",{}).values() if " @ " in str(e.get("name"))]
        print(pid, js.status_code, len(ev), [e["name"] for e in ev[:3]])
    except Exception as ex: print(pid,"fail",str(ex)[:80])
