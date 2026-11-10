import json, subprocess, sys, glob, os
nt = sys.argv[1]
for f in sorted(glob.glob(os.path.join(sys.argv[2], "c*.py"))):
    p = subprocess.run([nt, "run", "--json", os.path.basename(f)], cwd=os.path.dirname(f), capture_output=True, text=True)
    name = os.path.basename(f)[:-3]
    try:
        r = json.loads(p.stdout[p.stdout.index("{"):])
    except Exception:
        print(f"## {name}: NO JSON (exit {p.returncode})\n{p.stderr[-600:]}\n"); continue
    if not r.get("found"):
        print(f"## {name}: found=False\n"); continue
    print(f"## {name}: exit {p.returncode}, chain {len(r['chain'])}")
    for e in r["chain"]:
        loc = e["location"] or {}
        print(f"  {e['phase']:8} {e['op']:40} {os.path.basename(loc.get('file','?'))}:{loc.get('line')} | {loc.get('code','')[:60]}")
        print(f"           why: {e['cause'][:230]}")
    print()
