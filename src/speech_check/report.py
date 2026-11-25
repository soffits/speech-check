"""The short human-readable summary of a result."""
from __future__ import annotations

_LEVEL = {"high": "high", "medium": "medium", "low": "low", "very_low": "very low"}


def summary(result: dict) -> str:
    inp, rel = result["input"], result["reliability"]
    lines = [f"speech-check {result['tool']['version']}: {inp['file']}, {inp['duration_s']} s, "
             f"{inp['language']}, {inp['mode'].replace('_', ' ')}"]
    for w in result["recording"]["warnings"]:
        lines.append(f"  ! {w}")
    lines += ["", "Transcript (verbatim)", "  " + result["transcript"]["text"]]
    lines += _fluency(result["fluency"], rel["fluency"])
    if "pronunciation" in result:
        lines += _pronunciation(result["pronunciation"], rel["pronunciation"])
    if "intonation" in result:
        lines += _intonation(result["intonation"], rel["intonation"])
    return "\n".join(lines)


def _head(title: str, rel: dict) -> list[str]:
    return ["", f"{title} [reliability: {_LEVEL[rel['level']]}]"]


def _fluency(f: dict, rel: dict) -> list[str]:
    out = _head("Fluency", rel)
    if not f.get("available"):
        return out + [f"  not available: {f.get('reason')}"]
    p = f["pauses"]
    out.append(f"  speech rate {f['speech_rate_wpm']} words/min ({f['speech_rate_syll_per_s']} syll/s), articulation rate "
               f"{f['articulation_rate_syll_per_s']} syll/s, speaking {round(100 * f['phonation_time_ratio'])}% of the time")
    if f["mean_length_of_run_words"] is not None:
        out.append(f"  mean run between pauses: {f['mean_length_of_run_words']} words ({f['mean_length_of_run_syllables']} syllables)")
    pos = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in p["long_by_position"].items()) or "none"
    out.append(f"  pauses >= {p['threshold_s']} s: {p['count']} ({p['per_minute']}/min, mean {p['mean_duration_s']} s); "
               f"long (>= {p['long_threshold_s']} s): {p['long_count']} ({pos}); longest {p['longest_s']} s")
    for lp in p["long"][:6]:
        out.append(f"    {lp['duration']} s at {lp['start']} s, {lp['position'].replace('_', ' ')}: {lp['context']}")
    fl = f["fillers"]
    listed = ", ".join(f"{i['text']} at {i['start']} s" for i in fl["items"][:8])
    out.append(f"  fillers: {fl['count']} ({fl['per_100_words']} per 100 words)" + (f": {listed}" if listed else ""))
    r = f["repetitions"]
    if r["count"]:
        out.append("  repetitions: " + "; ".join(f"'{i['text']}' x{i['times']} at {i['start']} s" for i in r["items"][:6]))
    s = f["self_corrections"]
    if s["count"]:
        out.append("  self-corrections / cut-offs: " + "; ".join(f"{i['type'].replace('_', ' ')} '{i['text']}' at {i['start']} s" for i in s["items"][:6]))
    out.append("  (counts of fillers, repetitions and cut-offs are lower bounds)")
    return out


def _pronunciation(p: dict, rel: dict) -> list[str]:
    out = _head("Pronunciation (phone level)", rel)
    if not p.get("available"):
        return out + [f"  not available: {p.get('reason')}"]
    o = p["overall"]
    out.append(f"  {o['words_assessed']} words assessed; phone error rate {o['phone_error_rate']} "
               f"(includes recogniser noise; clean text-to-speech measured 0.06-0.09, so differences of a few hundredths mean nothing)")
    if "reference" in p:
        r = p["reference"]
        out.append(f"  read-aloud: {r['matched']}/{r['words_in_reference']} reference words matched"
                   + (f"; omitted: {', '.join(r['omitted'])}" if r["omitted"] else "")
                   + (f"; extra: {', '.join(r['inserted'])}" if r["inserted"] else "")
                   + (f"; said instead: {', '.join(s['said'] + ' for ' + s['expected'] for s in r['substituted'])}" if r["substituted"] else ""))
    rep = [x for x in p["patterns"] if x["repeated"]]
    if rep:
        out.append("  repeated sound patterns (most trustworthy):")
        for x in rep[:5]:
            out.append(f"    /{x['expected']}/ heard as /{x['heard']}/ {x['occurrences']} times: {', '.join(x['words'][:6])}")
    likely = [w for w in p["flagged"] if w["flag"] == "likely"]
    possible = [w for w in p["flagged"] if w["flag"] == "possible"]
    for label, group in (("likely problem words", likely), ("possible (needs other evidence)", possible)):
        if group:
            out.append(f"  {label}:")
            for w in group[:8]:
                changes = ", ".join(f"/{p['expected']}/ -> /{p['heard'] or 'nothing'}/ ({p['error_confidence']}"
                                    f"{', out of context' if p.get('source') == 'context_free' else ''})"
                                    for p in w.get("phones", []) if p["error_confidence"] >= 0.3)
                out.append(f"    {w['word']} at {w['start']} s: {changes or 'weak evidence'}")
    if not rep and not likely and not possible:
        out.append("  nothing flagged")
    return out


def _intonation(i: dict, rel: dict) -> list[str]:
    out = _head("Intonation (pitch)", rel)
    if not i.get("available"):
        return out + [f"  not available: {i.get('reason')}"]
    s = i["summary"]
    out.append(f"  median pitch {i['speaker_median_hz']} Hz; per sentence the median range is {s['median_sentence_range_st']} "
               f"semitones (sd {s['median_sentence_sd_st']}) over {s['sentences_with_stable_pitch']}/{s['sentences_total']} usable sentences; "
               f"flat sentences: {s['flat_sentences']}")
    for sent in i["sentences"]:
        if not sent.get("reliable"):
            continue
        fin = sent["final_contour"] or "n/a"
        out.append(f"    {sent['start']}-{sent['end']} s: range {sent['range_st']} st, sd {sent['sd_st']}, ends {fin}"
                   f"{' (a question)' if sent['ends_with'] == 'question' else ''}" + (" FLAT" if sent["flat"] else ""))
    return out
