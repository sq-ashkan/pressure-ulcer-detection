"""Deterministic verification of a BibTeX file against Crossref (and URL reachability).

For every entry with a DOI the Crossref record is fetched and compared field by field:
title (normalised similarity >= 0.90), family name of the first author, year, volume,
issue, first page, and container title (journal/booktitle, similarity >= 0.6).
Entries without DOI must have a reachable URL (HTTP 200 after redirects) or be a
book/report/thesis with publisher/institution/school. Every finding is reported;
nothing is changed automatically.

Usage: python scripts/verify_references.py docs/thesis/references.bib [--json out.json]
"""
import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

UA = "thesis-reference-check/1.0 (mailto:noreply@example.org)"


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start, depth, i = m.end(), 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0); i += 1
        body = text[start:i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j = fm.end()
            if j < len(body) and body[j] == "{":
                d, k = 1, j + 1
                while k < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[k], 0); k += 1
                val = body[j + 1:k - 1]
            elif j < len(body) and body[j] == '"':
                k = body.index('"', j + 1); val = body[j + 1:k]
            else:
                k = j
                while k < len(body) and body[k] not in ",\n":
                    k += 1
                val = body[j:k]
            fields[fm.group(1).lower()] = val.strip()
        entries.append({"type": m.group(1).lower(), "key": m.group(2), **fields})
    return entries


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = re.sub(r"[{}\\$^_~`'\"]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def get(url, accept="application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


def first_family(author_field):
    first = re.split(r"\s+and\s+", author_field or "")[0]
    return (first.split(",")[0] if "," in first else first.split()[-1] if first.split() else "")


def check_doi(e):
    doi = e["doi"].strip().removeprefix("https://doi.org/")
    try:
        _, raw = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    except urllib.error.HTTPError as ex:
        if ex.code == 404:   # e.g. arXiv DOIs (10.48550) are registered at DataCite
            try:
                _, raw = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi))
                d = json.loads(raw)["data"]["attributes"]
                probs = []
                if sim(e.get("title"), d["titles"][0]["title"]) < 0.90:
                    probs.append(f"title differs from DataCite: '{d['titles'][0]['title']}'")
                if e.get("year") and str(d.get("publicationYear")) != e["year"]:
                    probs.append(f"year: bib {e['year']} vs DataCite {d.get('publicationYear')}")
                return probs
            except Exception:
                pass
        return [f"DOI {doi}: Crossref HTTP {ex.code} (DOI may be wrong or not Crossref-registered)"]
    except Exception as ex:  # network
        return [f"DOI {doi}: Crossref request failed ({ex})"]
    m = json.loads(raw)["message"]
    probs = []
    ct = (m.get("title") or [""])[0]
    if sim(e.get("title"), ct) < 0.90 and not norm(e.get("title")).startswith(norm(ct)):
        probs.append(f"title differs: bib '{e.get('title')}' vs Crossref '{ct}'")
    auth = m.get("author") or m.get("editor") or []
    if auth and e.get("author"):
        if norm(first_family(e["author"])) != norm(auth[0].get("family", "")):
            probs.append(f"first author: bib '{first_family(e['author'])}' vs Crossref '{auth[0].get('family')}'")
    y = None
    for k in ("published-print", "published-online", "issued"):
        if m.get(k, {}).get("date-parts"):
            y = m[k]["date-parts"][0][0]; break
    years = {str(m[k]["date-parts"][0][0]) for k in ("published-print", "published-online", "issued")
             if m.get(k, {}).get("date-parts") and m[k]["date-parts"][0][0]}
    if e.get("year") and years and e["year"] not in years:
        probs.append(f"year: bib {e['year']} vs Crossref {sorted(years)}")
    for bf, cf in (("volume", "volume"), ("number", "issue")):
        if e.get(bf) and m.get(cf) and norm(e[bf]).lstrip("0") != norm(m[cf]).lstrip("0"):
            probs.append(f"{bf}: bib {e[bf]} vs Crossref {m[cf]}")
    if e.get("pages") and m.get("page"):
        if norm(e["pages"].split("-")[0]) != norm(m["page"].split("-")[0]):
            probs.append(f"pages: bib {e['pages']} vs Crossref {m['page']}")
    cont = (m.get("container-title") or [""])[0]
    venue = e.get("journal") or e.get("booktitle") or ""
    series = {"lecture notes in computer science", "spie proceedings", "proceedings of spie"}
    if venue and cont and norm(cont) not in series and sim(venue, cont) < 0.6 and norm(cont) not in norm(venue) and norm(venue) not in norm(cont):
        probs.append(f"venue: bib '{venue}' vs Crossref '{cont}'")
    return probs


def check_url(e):
    try:
        st, _ = get(e["url"], accept="text/html,*/*")
        return [] if st == 200 else [f"URL {e['url']}: HTTP {st}"]
    except urllib.error.HTTPError as ex:
        return [f"URL {e['url']}: HTTP {ex.code}"]
    except Exception as ex:
        return [f"URL {e['url']}: {ex}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--json")
    a = ap.parse_args()
    entries = parse_bib(open(a.bib, encoding="utf-8").read())
    report, bad = [], 0
    for e in entries:
        if e.get("doi"):
            probs, how = check_doi(e), "crossref"
        elif e.get("url"):
            probs, how = check_url(e), "url"
        elif e["type"] in ("book", "techreport", "phdthesis", "mastersthesis") and (
                e.get("publisher") or e.get("institution") or e.get("school")):
            probs, how = [], "book/report (manual)"
        else:
            probs, how = ["no DOI and no URL: cannot be verified automatically"], "none"
        report.append({"key": e["key"], "method": how, "ok": not probs, "problems": probs})
        bad += bool(probs)
        print(f"{'OK ' if not probs else 'BAD'} {e['key']:<34} {how:<20} {'; '.join(probs)[:200]}")
        time.sleep(0.2)
    print(f"\n{len(entries)} entries, {bad} with problems")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
