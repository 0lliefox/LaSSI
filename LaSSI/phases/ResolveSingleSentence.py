import logging
import re
import time as ti

from LaSSI.similarities.levenshtein import lev
from LaSSI.structures.meuDB.meuDB import MeuDBEntry, MeuDB

services = None
stanza_service = None

# ISO 8601 datetime: 2026-04-14T14:00Z, 2026-04-14T14:00:00, 2026-04-14T14:00+02:00, etc.
# CoreNLP tokenises this into multiple chunks at `T` and the dashes, so add an
# explicit DATE MEU spanning the whole thing — TypeResolver.mergeMeuNodes folds
# the chunks back into a single node when a MEU contains them.
_ISO_8601_DATETIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
# Date-only ISO 8601: 2026-04-27 (no time component).
_ISO_8601_DATE_ONLY = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?!T\d)\b"
)



# A plural weekday ("on Saturdays", "Mondays") names a recurring day, not one date. SUTime reads the lowercase form
# that way (a SET, "XX-XX-6") but resolves the capitalised form to a single date relative to the document date, and,
# with no reference date, to the day the pipeline runs ("Saturdays" -> 2026-11-14 on 19 Nov 2026), so the same
# sentence meant different things on different days. Both forms are now read as the recurring day, in SUTime's own
# SET notation (19 Nov 2026, Oliver).
_PLURAL_WEEKDAYS = {"mondays": 1, "tuesdays": 2, "wednesdays": 3, "thursdays": 4, "fridays": 5, "saturdays": 6,
                    "sundays": 7}


def _recurring_weekday(sentence, time_info):
    try:
        surface = sentence[int(time_info.get("start_char")):int(time_info.get("end_char"))]
    except (TypeError, ValueError):
        return time_info
    day = _PLURAL_WEEKDAYS.get(surface.strip().lower())
    if day is None or str(time_info.get("type", "")).upper() == "SET":
        return time_info
    return dict(time_info, text=f"XX-XX-{day}", type="SET")


# Stanza can stretch a proper name over a lowercase common noun that begins a lexicalised compound: in "Newcastle city
# centre" it tags "Newcastle city" as one GPE. LaSSI gives a Stanza entity confidence 1.0 (d8e59c4), so that span won
# the compound merge over HOnK's exact "city centre", and the sentence read as Newcastle city + "centre" instead of
# Newcastle + "city centre" (as the first LaSSI paper's own example has it). The span's boundary is the error, so it is
# repaired: when a name-type entity ends in lowercase words that, with the next one or two words, form an exact HOnK
# entry made of content words only, those words are cut from the entity. Names that end in a capitalised word
# ("Whitley Bay Metro" before "station") and every other span are unchanged. (19 Nov 2026, city-centre set.)
_NAME_TYPES = {"GPE", "LOC", "FAC", "ORG"}
_CLOSED_WORDS = {"a", "an", "the", "of", "in", "on", "at", "near", "to", "for", "from", "by", "with", "and", "or",
                 "but", "not", "no", "into", "onto", "over", "under", "off", "out", "up", "down", "as"}


class _TrimmedEntity:
    def __init__(self, text, type_, start_char, end_char):
        self.text, self.type, self.start_char, self.end_char = text, type_, start_char, end_char


def _honk_exact_lookup(services):
    """term -> True if the HOnK table has exactly that label (case-insensitive); None if no matcher is available."""
    try:
        matcher = services.getFuzzyHOnK().s
    except Exception:
        return None
    if not hasattr(matcher, "typedFuzzyMatch"):
        return None
    cache = {}

    def exact(term):
        if term not in cache:
            try:
                found = matcher.typedFuzzyMatch(1.0, term)
            except Exception:
                found = {}
            cache[term] = any(str(c).lower() == term for cands in found.values() for c, _ in cands)
        return cache[term]
    return exact


def _trim_overreaching_name(ent, sentence, tokens, exact):
    if exact is None or ent.type not in _NAME_TYPES or " " not in ent.text.strip():
        return None
    inside = [t for t in tokens if t[1] >= ent.start_char and t[2] <= ent.end_char]
    after = [t for t in tokens if t[1] >= ent.end_char]
    j = len(inside)
    while j > 1 and inside[j - 1][0][:1].islower():
        j -= 1
    if j == len(inside) or j == 0:
        return None
    tail = [t[0] for t in inside[j:]]
    for n in (1, 2):
        if len(after) < n:
            break
        words = tail + [t[0] for t in after[:n]]
        if not all(w.isalpha() and w.lower() not in _CLOSED_WORDS for w in words):
            continue
        if exact(" ".join(words).lower()):
            end = inside[j - 1][2]
            return _TrimmedEntity(sentence[ent.start_char:end], ent.type, ent.start_char, end)
    return None


def logger_func(x):
    return

def process_sentence_worker(args):
    from LaSSI.external_services.Services import Services
    
    # Check if we have 5 or 6 arguments
    if len(args) == 6:
        idx, sentence, with_time, recall_threshold, precision_threshold, disable_fuzzy_honk = args
    else:
        idx, sentence, with_time, recall_threshold, precision_threshold = args
        disable_fuzzy_honk = False

    global services, stanza_service
    if services is None:
        services = Services.getInstance(logger=logger_func)
        stanza_service = services.getStanzaNLP()

    print(f"Processing sentence: {sentence}")
    start_time = ti.time()

    multi_entity_unit = []

    _iso_spans = set()
    for m in _ISO_8601_DATETIME.finditer(sentence):
        text = m.group(0)
        _iso_spans.add((m.start(), m.end()))
        multi_entity_unit.append(MeuDBEntry(
            text, "DATE", m.start(), m.end(),
            text, 1.0, text, "ISO8601"))
    for m in _ISO_8601_DATE_ONLY.finditer(sentence):
        text = m.group(0)
        span = (m.start(), m.end())
        # Skip if a longer datetime span starts at the same position
        if any(s <= m.start() and e >= m.end() for (s, e) in _iso_spans):
            continue
        multi_entity_unit.append(MeuDBEntry(
            text, "DATE", m.start(), m.end(),
            text, 1.0, text, "ISO8601"))

    if not disable_fuzzy_honk:
        multi_entity_unit.extend(services.getFuzzyHOnK().resolve_u(
            recall_threshold, precision_threshold, sentence))

    # multi_entity_unit.extend(services.getGeoNames().resolve_u(
    #     recall_threshold, precision_threshold, sentence, "GPE"))
    #
    # multi_entity_unit.extend(services.getConcepts().resolve_u(
    #     recall_threshold, precision_threshold, sentence, "ENTITY"))

    for time_info in with_time:
        time_info = _recurring_weekday(sentence, time_info)
        time_entry = MeuDBEntry.from_dict_with_src(time_info, "SUTime")
        multi_entity_unit.append(time_entry)

    results = stanza_service(sentence)
    org_entities_for_replacement = []
    _tokens = [(t.text, t.start_char, t.end_char) for snt in results.sentences for t in snt.tokens
               if t.start_char is not None]
    _exact_lookup = None if disable_fuzzy_honk else _honk_exact_lookup(services)

    for ent in results.ents:
        trimmed = _trim_overreaching_name(ent, sentence, _tokens, _exact_lookup)
        if trimmed is not None:
            ent = trimmed
        monad = ent.text #.replace(" ", "") # TODO: This is affecting sentences from NEET, what was this used for before?
        if ent.type == "ORG":
            org_entities_for_replacement.append((ent.text, monad))

        # TODO: Might not need this hardcode but leaving for now...
        ent_type = "LOC" if ent.type == "FAC" else ent.type

        # A place span that runs across a comma ("Briggate, Leeds") is a place and the larger place containing it,
        # written as in an address; each part is its own entity, as when the tagger returns them separately.
        if ent_type in ("GPE", "LOC") and ", " in ent.text:
            offset = ent.start_char
            for part in ent.text.split(", "):
                if part.strip():
                    multi_entity_unit.append(MeuDBEntry(
                        part, ent_type, offset, offset + len(part),
                        part, lev(part.lower(), part.lower()), part, "Stanza"))
                offset += len(part) + 2
            continue

        multi_entity_unit.append(MeuDBEntry(
            ent.text, ent_type, ent.start_char, ent.end_char,
            monad, lev(monad.lower(), ent.text.lower()), monad, "Stanza"))

    for sent in results.sentences:
        for word in sent.words:
            if word.pos.lower() == 'verb':
                multi_entity_unit.append(MeuDBEntry(
                    word.text, word.pos.lower(), word.start_char,
                    word.end_char, word.lemma, 1.0, word.lemma, "Stanza"))

    modified_sentence = sentence
    for original, replacement in org_entities_for_replacement:
        modified_sentence = modified_sentence.replace(original, replacement)

    end_time = ti.time()
    benchmark_data = (idx, "Generating meuDB", end_time - start_time)

    print(f"Finished processing: {sentence}")
    return idx, MeuDB(modified_sentence, multi_entity_unit), benchmark_data