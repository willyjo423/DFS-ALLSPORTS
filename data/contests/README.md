# Contest exports for the ownership fit

Drop DraftKings contest standings CSVs here, one folder per sport
(`nba/`, `nhl/`, `cfb/`, `mlb/`). Any classic contest works; bigger fields
give truer ownership. The "Fit ownership" workflow runs on every upload:
it finds the board we published for that slate, refits the formula, and
writes `data/ownership_fit_<sport>.json`, which every publish reads.

How to get one: on draftkings.com (desktop browser), open a contest you
entered after it locks -> "Export Lineups to CSV". The file has every
player's %Drafted.
