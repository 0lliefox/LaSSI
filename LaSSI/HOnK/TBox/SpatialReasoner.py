from LaSSI.structures.extended_fol.Formulae import FVariable, FOr, FAnd, FUnaryPredicate, FBinaryPredicate, FNot
from LaSSI.HOnK.HOnK import HOnKSingleton, CasusHappening
from LaSSI.HOnK.TBox.ComparatorUtils import isImplication
from LaSSI.utils.logical_analysis_reader import spatial_relation_labels

_GEO_TYPES = {"LOC", "GPE", "FAC"}

# Names (lowercase) that are geo-typed (LOC/GPE/FAC) somewhere in the current
# comparison set. Populated per matrix build by TabularCWASemantics so that a
# place mis-typed as a bare "noun" in one sentence is still recognised as a
# location (see _space_named_geo_entities). Empty by default, so the per-pair
# compare path (ConstituentComparator / ModelSearch) keeps its strict
# geo-typed-only behaviour unless a caller opts in.
_DOC_GEO_NAMES: frozenset = frozenset()


def set_doc_geo_names(names) -> None:
    global _DOC_GEO_NAMES
    _DOC_GEO_NAMES = frozenset(
        n.strip().lower() for n in (names or ()) if isinstance(n, str) and n.strip()
    )


# Names (lowercase) that the current comparison set types GPE somewhere. Set by TabularCWASemantics before the
# constituent comparisons, which read it through _container_of: "On or near Haymarket, Newcastle" reaches eFOL as
# Haymarket[specification: "Newcastle"], with the extra's type dropped, and "Newcastle" typed GPE in another
# sentence tells that the extra is a town.
_DOC_GPE_NAMES: frozenset = frozenset()


def set_doc_gpe_names(names) -> None:
    global _DOC_GPE_NAMES
    _DOC_GPE_NAMES = frozenset(
        n.strip().lower() for n in (names or ()) if isinstance(n, str) and n.strip()
    )


# The evidence's source and location label, from the dataset's sidecar (set by LaSSI per dataset).
_DOC_SOURCE = None
_DOC_LOCATION = None
_MAP_POINT_TYPES_CACHE = None


def set_doc_source(source, location) -> None:
    global _DOC_SOURCE, _DOC_LOCATION
    _DOC_SOURCE = source.strip() if isinstance(source, str) and source.strip() else None
    _DOC_LOCATION = location.strip() if isinstance(location, str) and location.strip() else None


def _police_uk_map_point_types():
    """The point-of-interest types Police.uk names its non-street map points by (raw_data/police_uk_map_point_types.txt)."""
    global _MAP_POINT_TYPES_CACHE
    if _MAP_POINT_TYPES_CACHE is None:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "raw_data", "police_uk_map_point_types.txt")
        try:
            with open(path, encoding="utf-8") as fh:
                _MAP_POINT_TYPES_CACHE = frozenset(l.strip().lower() for l in fh if l.strip() and not l.startswith("#"))
        except OSError:
            _MAP_POINT_TYPES_CACHE = frozenset()
    return _MAP_POINT_TYPES_CACHE


# --- Police.uk "On or near X" (dev-set change after archived v4.4, agreed with the supervisor 12-14 Nov 2026) --------
#
# Police.uk publishes an approximate location only: each crime is moved to its nearest map point, the centre of a
# street or a public place, and the record names that point as "On or near X" (data.police.uk, About, "Location
# Anonymisation"). So, for a Police.uk record, "On or near X" is read as the vicinity of X, in both arms:
#   - it fully supports "near X" (and "on or near X"), but not "on X" or "at X" (_space_relation_narrowing);
#   - a point named only by its type ("Parking Area", "Bus/Coach Station") is never the same place as a named place
#     (_spaces_corefer);
#   - a different street or point never contradicts it, because LaSSI measures no distances; a different town does,
#     as for any source (policy v3, below: _places_derivably_disjoint).
# This reading is specific to the source. It is not a general rule that "on X" implies "near X".


def _police_uk_record_label():
    """The record's map-point name ("Prince Albert Terrace" for "On or near Prince Albert Terrace") when the evidence
    is a Police.uk record, else None."""
    import re
    if (_DOC_SOURCE or "").lower() != "police.uk" or not _DOC_LOCATION:
        return None
    label = re.sub(r"^\s*on\s+or\s+near\s+", "", _DOC_LOCATION, flags=re.IGNORECASE).strip()
    return label or None


def _is_police_uk_type_label(name) -> bool:
    """True when `name` is the record's map-point name and that name is a type of place ("Parking Area")."""
    label = _police_uk_record_label()
    return (label is not None and isinstance(name, str) and _name_tokens(name) == _name_tokens(label)
            and label.lower() in _police_uk_map_point_types())


def _is_police_uk_record_point(name) -> bool:
    """True when `name` is the record's map-point name, whether a street or a type of place."""
    label = _police_uk_record_label()
    return label is not None and isinstance(name, str) and _name_tokens(name) == _name_tokens(label)


# --- Forecasts keep their place (labelling policy v3.1's Forecasts rule; Oliver, 24 Nov 2026) -------------------------
#
# A forecast applies only to its own place and time, so a forecast for another town is not a contradiction: "Gateshead
# is forecast to be cloudy at 4pm" and the Met Office forecast for Newcastle can both be true. A statement is a weather
# forecast when it holds a term of HOnK's PredictionVerb class (forecast, predict, expect ...) and a term of its
# WeatherConditionNoun or WeatherConditionAdjective class (forecast, rain, precipitation, chance, cloud ...). That is
# the class rule ChunkProfiler already applies to weather report headers, read here from the statement's own words.
# "Work expected to end ..." has a prediction term and no weather term, so roadworks keep the town test. Only the place
# test is off for forecasts: the time test and the comparison of the weather itself are unchanged.

def _forecast_classes():
    """(prediction terms, weather-condition terms), lowercased, from HOnK's classes; empty when HOnK is not ready."""
    if not HOnKSingleton.isReady():
        return frozenset(), frozenset()
    kb = HOnKSingleton.get()

    def lower(ws):
        return {w.lower() for w in (ws or ()) if isinstance(w, str)}

    return (lower(kb.getPredictionVerbs()),
            lower(kb.getWeatherConditionNouns()) | lower(kb.getWeatherConditionAdjectives()))


def _statement_terms(formula, out=None, depth=0):
    """The words of a statement: its relations, arguments, their specifications and their properties, except the SPACE
    and TIME values (a place or a time is not what the statement says)."""
    out = set() if out is None else out
    if depth > 8 or formula is None:
        return out

    def words(text):
        if isinstance(text, str) and text and not text.startswith("?"):
            t = text.strip().lower()
            out.add(t)
            out.update(t.split())

    def props(properties):
        for k, vs in (properties or ()):
            if k in ("SPACE", "TIME"):
                continue
            for x in (vs if isinstance(vs, tuple) else (vs,)):
                _statement_terms(x, out, depth + 1)

    if isinstance(formula, str):
        words(formula)
    elif isinstance(formula, FVariable):
        words(formula.name)
        words(formula.specification if isinstance(formula.specification, str) else None)
        props(formula.properties)
    elif isinstance(formula, FUnaryPredicate):
        words(formula.rel)
        _statement_terms(formula.arg, out, depth + 1)
        props(formula.properties)
    elif isinstance(formula, FBinaryPredicate):
        words(formula.rel)
        _statement_terms(formula.src, out, depth + 1)
        _statement_terms(formula.dst, out, depth + 1)
        props(formula.properties)
    elif isinstance(formula, (FAnd, FOr)):
        for x in formula.args:
            _statement_terms(x, out, depth + 1)
    elif isinstance(formula, FNot):
        _statement_terms(formula.arg, out, depth + 1)
    elif isinstance(formula, (tuple, list, frozenset)):
        for x in formula:
            _statement_terms(x, out, depth + 1)
    return out


def _is_weather_forecast(formula) -> bool:
    prediction, weather = _forecast_classes()
    if not prediction or not weather:
        return False
    terms = _statement_terms(formula)
    return not terms.isdisjoint(prediction) and not terms.isdisjoint(weather)


# Spatial-relation type labels, declared once in logical_analysis.json's
# `spatial_relations` block (where they originate as the `type: space`
# derivation-rule properties). `movement` = displacement-or-stay labels;
# `relation` adds the proximity label "near place" — the full set that can
# appear under a SPACE value's `type` property, used to compute the
# directional implication between e.g. a plain "near place" and a
# disjunctive "on or near" = OR(stay in place, near place).
_SPATIAL_MOVEMENT_LABELS = spatial_relation_labels("movement")
_SPATIAL_RELATION_LABELS = spatial_relation_labels("relation")
_PROPER_NOUN_TYPES = {"LOC", "GPE", "ORG", "PERSON", "FAC"}

def _strip_spatial_type_properties(props):
    if not props:
        return props
    new_props = []
    for key, val in props:
        if key == "type":
            vals = val if isinstance(val, tuple) else (val,)
            filtered = tuple(
                v for v in vals
                if not (isinstance(v, str) and v.strip().lower() in _SPATIAL_MOVEMENT_LABELS)
            )
            if not filtered:
                continue
            if len(filtered) == 1:
                new_props.append((key, filtered[0]))
            else:
                new_props.append((key, filtered))
        else:
            new_props.append((key, val))
    return frozenset(new_props)

def _canonicalize_geo_fvar(fvar):
    if not isinstance(fvar, FVariable):
        return fvar
    if (fvar.type or "").upper() not in _GEO_TYPES and (fvar.type or "").upper() != "SPACE":
        return fvar
    new_props = _strip_spatial_type_properties(fvar.properties)
    spec = fvar.specification
    if not isinstance(spec, str) or not spec or not fvar.name:
        if new_props != fvar.properties:
            return FVariable(fvar.name, fvar.type, fvar.specification, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
        return fvar
    spec_first_word = spec.strip().split()[0].lower() if spec.strip() else ""
    if not spec_first_word:
        if new_props != fvar.properties:
            return FVariable(fvar.name, fvar.type, fvar.specification, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
        return fvar
    facility_nouns = HOnKSingleton.get().getFacilityNouns()
    if spec_first_word not in facility_nouns:
        if new_props != fvar.properties:
            return FVariable(fvar.name, fvar.type, fvar.specification, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
        return fvar
    if spec_first_word in fvar.name.lower().split():
        # Already folded, avoid duplication.
        return FVariable(fvar.name, fvar.type, None, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
    return FVariable(f"{fvar.name} {spec.strip()}", fvar.type, None, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)

def _canonicalize_proper_noun_modifier(fvar):
    if not isinstance(fvar, FVariable):
        return fvar
    if (fvar.type or "").upper() not in _PROPER_NOUN_TYPES:
        return fvar
    spec = fvar.specification
    if not isinstance(spec, str) or not spec.strip():
        return fvar
    spec_first = spec.strip().split()[0].lower()
    if spec_first in HOnKSingleton.get().getFacilityNouns():
        return fvar
    return FVariable(spec, "noun", None, None, fvar.id, fvar.properties,
                     fvar.spec_negation, fvar.meta, fvar.asAll)

def _strip_geo_generic_suffix(name: str | None) -> str | None:
    if name is None:
        return None
    words = name.strip().split()
    if len(words) > 1 and words[-1].lower() in HOnKSingleton.get().getGeoSuffixNouns():
        return " ".join(words[:-1])
    return name

def _has_near_place(fvar) -> bool:
    if not isinstance(fvar, FVariable) or not fvar.properties:
        return False
    for key, val in fvar.properties:
        if key != "type":
            continue
        vals = val if isinstance(val, tuple) else (val,)
        for v in vals:
            if isinstance(v, FOr):
                for arg in getattr(v, "args", ()) or ():
                    arg_name = getattr(arg, "name", None)
                    if arg_name and "near place" in arg_name.lower():
                        return True
                continue
            v_str = v if isinstance(v, str) else getattr(v, "name", None)
            if v_str and "near place" in v_str.lower():
                return True
    return False

def _space_near_place_names(formula):
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return None
    if not formula.properties:
        return None
    space_vals = None
    for k, vs in formula.properties:
        if k == "SPACE":
            space_vals = vs if isinstance(vs, tuple) else (vs,)
            break
    if not space_vals:
        return None
    names = set()
    for v in space_vals:
        if not isinstance(v, FVariable):
            continue
        if v.name is None or (isinstance(v.name, str) and v.name.startswith("?")):
            continue
        if not v.properties:
            continue
        type_vals = None
        for kk, vv in v.properties:
            if kk == "type":
                type_vals = vv if isinstance(vv, tuple) else (vv,)
                break
        if not type_vals:
            continue
        is_near = False
        for t in type_vals:
            if isinstance(t, FOr):
                for arg in getattr(t, "args", ()) or ():
                    arg_name = getattr(arg, "name", None)
                    if arg_name and "near place" in arg_name.lower():
                        is_near = True
                        break
                if is_near:
                    break
                continue
            t_str = t if isinstance(t, str) else (t.name if isinstance(t, FVariable) else None)
            if t_str and "near place" in str(t_str).lower():
                is_near = True
                break
        if is_near:
            stripped = _strip_geo_generic_suffix(v.name)
            names.add(stripped.strip())
    return names if names else None

def _space_mismatch_contradiction(lhs, rhs) -> bool:
    # If the two SPACE descriptors corefer (e.g. one names a place as a single
    # "<Locality> <Facility>" compound and the other decomposes it into a
    # [<Facility>, <Locality>] pair), they denote the same place and cannot be
    # a contradiction — regardless of the spatial-relation direction.
    if _space_coreference_verdict(lhs, rhs) is not None:
        return False
    # A forecast applies only to its own place (policy v3.1): see _is_weather_forecast.
    if _is_weather_forecast(lhs) or _is_weather_forecast(rhs):
        return False
    lhs_places = _space_place_values(lhs, near=True)
    rhs_places = _space_place_values(rhs, near=True)
    if not lhs_places or not rhs_places:
        return False
    return _places_derivably_disjoint(lhs_places, rhs_places, require_unique_on_both=False)

def _space_named_geo_entities(formula):
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return None
    if not formula.properties:
        return None
    space_vals = None
    for k, vs in formula.properties:
        if k == "SPACE":
            space_vals = vs if isinstance(vs, tuple) else (vs,)
            break
    if not space_vals:
        return None
    out = set()
    for v in space_vals:
        if not isinstance(v, FVariable):
            continue
        if v.name is None or (isinstance(v.name, str) and v.name.startswith("?")):
            continue
        # SPACE-slot values are locations by construction, but the NER stage
        # types named places inconsistently across rows (LOC in one sentence,
        # bare "noun"/None in another — a known gazetteer/NodeMerger
        # nondeterminism, see reference_concerns_gpe_min_overlap_bug). Accept a
        # mis-typed value only when this exact name is geo-typed (LOC/GPE/FAC)
        # somewhere in the comparison set (_DOC_GEO_NAMES), so a place like
        # "South Gosforth"/"Benton" still counts, while a generic facility/
        # common noun ("station", "premises", "section") — never geo-typed
        # anywhere — is not treated as a distinguishing location. Geo-typed
        # values are taken as-is.
        _vt = (v.type or "").upper()
        if _vt not in _GEO_TYPES:
            _nm = v.name.strip().lower() if isinstance(v.name, str) else ""
            if not _nm or _nm not in _DOC_GEO_NAMES:
                continue
        if v.properties:
            is_near = False
            for kk, vv in v.properties:
                if kk != "type":
                    continue
                type_vals = vv if isinstance(vv, tuple) else (vv,)
                for t in type_vals:
                    t_str = t if isinstance(t, str) else (t.name if isinstance(t, FVariable) else None)
                    if t_str and "near place" in str(t_str).lower():
                        is_near = True
                        break
                if is_near:
                    break
            if is_near:
                continue
        out.add(v.name.strip())
    return out if out else None

def _space_city_mismatch_contradiction(lhs, rhs) -> bool:
    # "between A and B" vs "between C and B": the two SPACE descriptors share an
    # endpoint (B) but describe different route segments. A pure subset/superset
    # ({A} vs {A,B}) is specificity rather than contradiction, so a contradiction
    # needs a place on EACH side with no counterpart on the other, and that pair
    # must be derivably distinct (see _places_derivably_disjoint).
    if _space_coreference_verdict(lhs, rhs) is not None:
        return False
    # A forecast applies only to its own place (policy v3.1): see _is_weather_forecast.
    if _is_weather_forecast(lhs) or _is_weather_forecast(rhs):
        return False
    lhs_places = _space_place_values(lhs, near=False)
    rhs_places = _space_place_values(rhs, near=False)
    if not lhs_places or not rhs_places:
        return False
    return _places_derivably_disjoint(lhs_places, rhs_places, require_unique_on_both=True)


# --- Deriving a spatial contradiction (v4.4) ---------------------------------
#
# The confidence score is 0 only when a contradiction is derived, never when
# compatibility merely cannot be proven (thesis Ch.5, Definition of confidence;
# Ch.6, Refuted needs an explicit contradiction). Earlier versions declared two
# SPACE descriptors contradictory whenever name_eq could not prove their places
# compatible, so missing knowledge ("Newcastle upon Tyne" vs "Newcastle", or a
# generic "station" vs "Monkseaton station") became a contradiction. A spatial
# contradiction is now derived from four stated assumptions:
#   1. same event: it is only tested between atoms the comparator already treats
#      as describing the same event;
#   2. one place per event: a different place for that event conflicts;
#   3. unique names: distinct *specific* place names denote distinct places,
#      where a generic class noun ("station", "metro station") is not a name and
#      spelling variants ("Newcastle upon Tyne" / "Newcastle", "Haymarket
#      Station" / "Haymarket") are one name;
#   4. containment: a place inside the other (HOnK isA/partOf reachability via
#      name_eq) is specificity, not contradiction.
# Two distinct names conflict only when both are settlements (towns, cities),
# for every source, Police.uk included (labelling policy v3; Ellis and Oliver,
# 23-24 Nov 2026, a declared change). A different street, station or other
# facility on its own is not a contradiction: within one town, LaSSI measures no
# distances, so it cannot show the event was not there (v4.4 also read street vs
# street and station vs station as disjoint). Different or unknown kinds cannot
# be shown disjoint either, so the pair stays indifferent.

_STREET_TYPE_NOUNS_CACHE = None


def _street_type_nouns():
    global _STREET_TYPE_NOUNS_CACHE
    if _STREET_TYPE_NOUNS_CACHE is None:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "raw_data", "nouns", "street_type_nouns.txt")
        try:
            with open(path, encoding="utf-8") as fh:
                _STREET_TYPE_NOUNS_CACHE = frozenset(l.strip().lower() for l in fh if l.strip())
        except OSError:
            _STREET_TYPE_NOUNS_CACHE = frozenset()
    return _STREET_TYPE_NOUNS_CACHE


_PLACE_CLASS_NOUNS_CACHE = None


def _place_class_nouns():
    """Generic place-class nouns ("station", "car park", "entrance", "road", "area"), read from the
    noun lists in raw_data/nouns. HOnK's LocationNoun class is not used: it also holds named places
    ("Newcastle", "Haymarket"), which must never count as generic."""
    global _PLACE_CLASS_NOUNS_CACHE
    if _PLACE_CLASS_NOUNS_CACHE is None:
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "raw_data", "nouns")
        out = set()
        for f in ("facility_nouns.txt", "access_point_nouns.txt", "route_nouns.txt", "geo_suffix_nouns.txt"):
            try:
                with open(os.path.join(base, f), encoding="utf-8") as fh:
                    out.update(l.strip().lower() for l in fh if l.strip())
            except OSError:
                continue
        _PLACE_CLASS_NOUNS_CACHE = frozenset(out)
    return _PLACE_CLASS_NOUNS_CACHE


def _class_nouns():
    return _place_class_nouns() | _street_type_nouns()


def _name_tokens(name):
    return [t for t in str(name).lower().replace("-", " ").replace("/", " ").split() if t and t != "the"]


def _is_generic_place_name(name) -> bool:
    """A name made only of class nouns ("station", "metro station", "car park")
    names a kind of place, not a place."""
    toks = _name_tokens(name)
    classes = _class_nouns()
    if not toks:
        return True
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and f"{toks[i]} {toks[i + 1]}" in classes:
            i += 2
        elif toks[i] in classes:
            i += 1
        else:
            return False
    return True


def _normalised_place_name(name) -> str:
    """One spelling for the variants of a name: lower-cased, "X upon Y" read as
    X, and trailing class nouns dropped ("Haymarket Metro station" -> "haymarket")."""
    toks = _name_tokens(name)
    if "upon" in toks[1:]:
        toks = toks[:toks.index("upon")]
    classes = _class_nouns()
    while len(toks) > 1:
        if len(toks) > 2 and f"{toks[-2]} {toks[-1]}" in classes:
            toks = toks[:-2]
        elif toks[-1] in classes:
            toks = toks[:-1]
        else:
            break
    return " ".join(toks)


_CLASS_TYPED_MEMBERS_CACHE = None
_BROAD_PLACE_NOUNS = frozenset({"premises", "building", "property", "site", "location", "venue", "establishment",
                                "hub", "metro", "area", "region", "district", "zone", "sector", "vicinity",
                                "neighbourhood", "neighborhood", "part", "access", "line", "service", "network",
                                "route", "direction", "endpoint", "corridor", "track"})


def _class_typed_members():
    """name -> the HOnK isA classes it belongs to whose head is a place class noun
    (e.g. "Haymarket" -> {"metro station"}). Built once from the isA adjacency."""
    global _CLASS_TYPED_MEMBERS_CACHE
    if _CLASS_TYPED_MEMBERS_CACHE is None:
        out = {}
        if HOnKSingleton.isReady():
            # Specific facility classes only: broad heads ("place", "site", "area",
            # "location", "building") would type almost anything.
            broad = _BROAD_PLACE_NOUNS | _street_type_nouns()
            heads = _place_class_nouns() - broad
            isA = getattr(HOnKSingleton.get(), "_isA_supers", {}) or {}
            for cls, members in isA.items():
                toks = _name_tokens(cls) if isinstance(cls, str) else []
                if toks and (toks[-1] in heads or " ".join(toks[-2:]) in heads):
                    for m in members:
                        out.setdefault(m, set()).add(cls)
        _CLASS_TYPED_MEMBERS_CACHE = out
    return _CLASS_TYPED_MEMBERS_CACHE


def _container_of(value) -> str | None:
    """The town a SPACE value is written inside: T for "X, T", which eFOL carries as X[specification: T]
    ("On or near Haymarket, Newcastle"), when the comparison set types T GPE somewhere (_DOC_GPE_NAMES). None
    otherwise, so a class noun ("Whitley Bay[extra: station]") or another extra ("Met Office[extra: forecast]") is
    not a town. Policy v3.1 (Oliver, 24 Nov): "X, T" is a place inside T."""
    spec = getattr(value, "specification", None)
    if not isinstance(spec, str) or not spec.strip():
        return None
    t = spec.strip()
    if t.lower() not in _DOC_GPE_NAMES or _is_generic_place_name(t):
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and _normalised_place_name(name) == _normalised_place_name(t):
        return None
    return t


def _place_kind(name, value) -> str | None:
    """The kind of place a name denotes: from its own class noun; else from its specification (e.g.
    Haymarket[extra: Metro station]); else "settlement" when the sentence types it GPE; else from a HOnK class it is
    listed in (Parmenides' "Haymarket isA metro station"). None when it cannot be told.
    The sentence's GPE comes before HOnK's class (Oliver, 24 Nov: "GPE first"): Parmenides lists the towns
    Sunderland, Whitley Bay and South Shields as Metro stations, and "in Sunderland" is a town. "GPE first" is for a
    bare GPE: one written inside a town ("Haymarket, Newcastle", see _container_of) lies in that town, so it is not
    read as a town itself (policy v3.1; Oliver, 24 Nov)."""
    streets = _street_type_nouns()
    classes = _place_class_nouns()

    def kind_of(text):
        toks = _name_tokens(text) if text else []
        for k in range(len(toks) - 1, -1, -1):
            if k > 0 and f"{toks[k - 1]} {toks[k]}" in classes:
                return f"{toks[k - 1]} {toks[k]}"
            if toks[k] in streets:
                return "street"
            if toks[k] in classes:
                return "station" if toks[k] in ("station", "stop", "interchange", "metro") else toks[k]
        return None

    kind = kind_of(name)
    if kind is None and value is not None:
        spec = getattr(value, "specification", None)
        if isinstance(spec, str):
            kind = kind_of(spec)
    if (kind is None and value is not None and (getattr(value, "type", "") or "").upper() == "GPE"
            and _container_of(value) is None):
        kind = "settlement"
    if kind is None and HOnKSingleton.isReady():
        # A bare name typed by HOnK, e.g. Parmenides' "Haymarket isA metro station".
        for c in sorted(_class_typed_members().get(name, ())):
            kind = kind_of(c)
            if kind is not None:
                break
    return kind


def _same_place(kb, a, b) -> bool:
    if a == b or a.lower() == b.lower():
        return True
    if _normalised_place_name(a) == _normalised_place_name(b):
        return True
    return kb is not None and _geo_names_compatible(kb, a, b)


def _places_derivably_disjoint(lhs_places: dict, rhs_places: dict, require_unique_on_both: bool) -> bool:
    """`lhs_places`/`rhs_places` map a place name to its FVariable (or None).
    True only when some pair of specific names, one per side, has no
    counterpart on the other side and both are settlements (policy v3: a
    different street, station or facility alone is not a contradiction)."""
    lhs, rhs, lhs_unique, rhs_unique = _unique_places(lhs_places, rhs_places)
    if not lhs or not rhs:
        return False
    if require_unique_on_both:
        if not lhs_unique or not rhs_unique:
            return False
    elif len(lhs_unique) < len(lhs) or len(rhs_unique) < len(rhs):
        # A shared place: the descriptors overlap, which is not a contradiction.
        return False
    for ln, lv in lhs_unique.items():
        if _place_kind(ln, lv) != "settlement":
            continue
        for rn, rv in rhs_unique.items():
            if _place_kind(rn, rv) == "settlement":
                return True
    return False


def _unique_places(lhs_places: dict, rhs_places: dict):
    """(lhs, rhs, lhs_unique, rhs_unique): each side's specific places (generic class nouns dropped), and those with
    no counterpart on the other side (same name, spelling variant, or a HOnK-compatible name)."""
    kb = HOnKSingleton.get() if HOnKSingleton.isReady() else None
    lhs = {n: v for n, v in lhs_places.items() if not _is_generic_place_name(n)}
    rhs = {n: v for n, v in rhs_places.items() if not _is_generic_place_name(n)}
    lhs_unique = {n: v for n, v in lhs.items() if not any(_same_place(kb, n, o) for o in rhs)}
    rhs_unique = {n: v for n, v in rhs.items() if not any(_same_place(kb, n, o) for o in lhs)}
    return lhs, rhs, lhs_unique, rhs_unique


def _space_places_differ(lhs, rhs) -> bool:
    """True when each side names a specific place that the other doesn't, whether or not the two can be shown
    disjoint: another street, station or facility in the same town, or a forecast's other town. That is not a
    contradiction (policy v3), but the evidence doesn't support the claim's place either, so the score is capped at
    NEE (TabularCWASemantics.get_straightforward_id_similarity; policy v3.1, Oliver, 24 Nov: "a different street gets
    NEE in the score"). One side naming a place the other leaves out ("on Fenkle Street in Newcastle" against "in
    Newcastle") is not a different place, and descriptors that corefer are one place. Nor is a Police.uk map point
    named by its type ("Bus/Coach Station"): it names a kind of place, not which one, like a class noun."""
    if _space_coreference_verdict(lhs, rhs) is not None:
        return False

    def specific(places):
        return {n: v for n, v in places.items()
                if not (_is_police_uk_type_label(n) or _is_police_uk_type_label(getattr(v, "name", None)))}

    for near in (True, False):
        lhs_places = _space_place_values(lhs, near=near)
        rhs_places = _space_place_values(rhs, near=near)
        if lhs_places and rhs_places:
            _, _, lhs_unique, rhs_unique = _unique_places(lhs_places, rhs_places)
            if specific(lhs_unique) and specific(rhs_unique):
                return True
    return False


def _space_place_values(formula, near: bool):
    """Name -> FVariable for the SPACE places that the near-place rule
    (`near=True`) or the named-entity rule (`near=False`) compares; the same
    selection as `_space_near_place_names` / `_space_named_geo_entities`.
    A place written inside a town ("Haymarket, Newcastle": see _container_of)
    also names that town, as a GPE (policy v3.1)."""
    names = _space_near_place_names(formula) if near else _space_named_geo_entities(formula)
    if not names:
        return None
    values = {}
    for k, vs in (getattr(formula, "properties", None) or ()):
        if k != "SPACE":
            continue
        for v in (vs if isinstance(vs, tuple) else (vs,)):
            if isinstance(v, FVariable) and isinstance(v.name, str):
                values[v.name.strip()] = v
                stripped = _strip_geo_generic_suffix(v.name)
                if stripped:
                    values.setdefault(stripped.strip(), v)
    out = {n: values.get(n) for n in names}
    for n in names:
        town = _container_of(values.get(n))
        if town is not None:
            out.setdefault(town, FVariable(town, "GPE"))
    return out

def _geo_names_compatible(kb, lhs_name: str, rhs_name: str) -> bool:
    """Compare location names exactly first, then case-fold as a fallback.

    Some ontology entries are case-sensitive enough that lowercasing first can
    lose a good match ("Newcastle upon Tyne" -> "Newcastle").  Other inputs
    differ only by parser casing ("Parking Area" vs "parking area").  Preserve
    the explicit form as authoritative, and use lowercase only when the exact
    comparison is inconclusive.
    """
    eq = kb.name_eq(lhs_name, rhs_name)
    if eq == CasusHappening.EQUIVALENT or isImplication(eq):
        return True
    lhs_lower = lhs_name.lower()
    rhs_lower = rhs_name.lower()
    if lhs_lower == rhs_lower:
        return True
    if lhs_lower != lhs_name or rhs_lower != rhs_name:
        eq_lower = kb.name_eq(lhs_lower, rhs_lower)
        if eq_lower == CasusHappening.EQUIVALENT or isImplication(eq_lower):
            return True
    return False


# --- Compound-vs-decomposed location coreference -------------------------
#
# A location is sometimes parsed as a single "<Locality> <Facility>" compound
# Singleton ("Newcastle Bus Station") and elsewhere as a [<Facility>, <Locality>]
# pair of Singletons ("Bus/Coach Station" + "Newcastle"). These denote the same
# real-world place. Recognising the coreference needs the city and facility
# parts *jointly*, so it can't be expressed as a per-FVariable comparison; the
# helpers below operate on the whole SPACE value-list of a predicate.
#
# Facility-head compatibility is data-driven from HOnK (synonymy + a single isA
# hop), and never a hardcoded word list. The directional verdict reflects the
# spatial-relation type (near vs on-or-near), so a plain "near" entails a
# disjunctive "on or near" but not vice versa.


def _space_geo_values(formula):
    """Return the list of FVariable values under the SPACE property key of a
    predicate (unwrapping FNot)."""
    if isinstance(formula, FNot):
        formula = formula.arg
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return []
    if not formula.properties:
        return []
    for k, vs in formula.properties:
        if k == "SPACE":
            vals = vs if isinstance(vs, tuple) else (vs,)
            return [v for v in vals if isinstance(v, FVariable)]
    return []


def _contains_class_noun(name) -> bool:
    """True iff `name` contains a HOnK-known facility / access-point / route /
    location class noun as a whitespace-delimited token (so "Newcastle Bus
    Station" matches via "station", but bare "Newcastle" does not). Sourced
    from the ontology noun lists, never hardcoded."""
    from LaSSI.ner.string_functions import _ontology_class_suffix_terms
    if not isinstance(name, str) or not name:
        return False
    terms = _ontology_class_suffix_terms()
    if not terms:
        return False
    lowered = name.lower()
    tokens = set(lowered.replace("/", " ").split())
    for term in terms:
        if " " in term:
            if term in lowered:
                return True
        elif term in tokens:
            return True
    return False


def _facility_head_compatible(kb, fa, fb) -> 'CasusHappening':
    """Compare two facility/location head phrases ("bus station" vs "bus/coach
    station", "coach station" vs "bus station"), bypassing name_eq's
    supertype-intersection gate (which short-circuits to INDIFFERENT when one
    side has an empty supertype closure). Returns EQUIVALENT when they denote
    the same facility type via case-insensitive equality, HOnK synonymy, or a
    single isA hop in either direction; otherwise INDIFFERENT.

    A sub/super-type facility within the *same locality* is taken to be the
    same physical place, so the verdict is symmetric EQUIVALENT — the only
    coreference asymmetry comes from the spatial-relation type, computed
    separately in `_space_coreference_verdict`."""
    if not fa or not fb:
        return CasusHappening.INDIFFERENT
    a = fa.strip().lower()
    b = fb.strip().lower()
    if not a or not b:
        return CasusHappening.INDIFFERENT
    if a == b:
        return CasusHappening.EQUIVALENT
    if kb.are_synonyms(a, b):
        return CasusHappening.EQUIVALENT
    isA = getattr(kb, "_isA_supers", {})
    if b in isA.get(a, ()) or a in isA.get(b, ()):
        return CasusHappening.EQUIVALENT
    return CasusHappening.INDIFFERENT


def _decompose_compound(compound, locality):
    """If `locality`'s tokens are a contiguous prefix or suffix of `compound`'s
    tokens, return the remaining (facility) tokens as a string; else None.
    Handles both "Newcastle Bus Station" (city prefix) and "Bus/Coach Station
    Newcastle" (city suffix)."""
    if not compound or not locality:
        return None
    ct = compound.split()
    lt = locality.split()
    if len(ct) <= len(lt):
        return None
    lc = [t.lower() for t in ct]
    ll = [t.lower() for t in lt]
    if lc[:len(ll)] == ll:
        return " ".join(ct[len(ll):])
    if lc[-len(ll):] == ll:
        return " ".join(ct[:-len(ll)])
    return None


def _spaces_corefer(lhs_vals, rhs_vals, kb) -> bool:
    """True iff one SPACE list is a single "<Locality> <Facility>" compound and
    the other decomposes it into a separate locality + a facility whose head is
    ontologically compatible (the compound-vs-decomposed case)."""
    for comp_vals, other_vals in ((lhs_vals, rhs_vals), (rhs_vals, lhs_vals)):
        comp_names = [v.name for v in comp_vals
                      if v.name and _contains_class_noun(v.name) and not _is_police_uk_type_label(v.name)]
        other_names = [v.name for v in other_vals if v.name]
        # A Police.uk map point named by its type ("Bus/Coach Station") does not say which one it is.
        other_facilities = [n for n in other_names if _contains_class_noun(n) and not _is_police_uk_type_label(n)]
        for cf in comp_names:
            for loc in other_names:
                remainder = _decompose_compound(cf, loc)
                if not remainder or not _contains_class_noun(remainder):
                    continue
                for fac in other_facilities:
                    if _facility_head_compatible(kb, remainder, fac) == CasusHappening.EQUIVALENT:
                        return True
    return False


def _collect_label_strings(t):
    """Lowercased label strings carried by a `type` value, expanding FOr arms
    (each arm is typically an FVariable named after the spatial relation)."""
    out = set()
    if isinstance(t, str):
        if t.strip():
            out.add(t.strip().lower())
    elif isinstance(t, FOr):
        for arg in getattr(t, "args", ()) or ():
            out |= _collect_label_strings(arg)
    elif isinstance(t, FVariable):
        if t.name and t.name.strip():
            out.add(t.name.strip().lower())
    return out


def _spatial_relation_labels(vals):
    """Aggregate the spatial-relation type labels (stay/near/motion) of the
    facility-bearing SPACE entities, expanding any FOr disjunction."""
    labels = set()
    for v in vals:
        if not isinstance(v, FVariable) or not v.properties or not v.name:
            continue
        if not _contains_class_noun(v.name):
            continue
        for k, val in v.properties:
            if k != "type":
                continue
            for t in (val if isinstance(val, tuple) else (val,)):
                labels |= _collect_label_strings(t)
    return labels & _SPATIAL_RELATION_LABELS


def _space_coreference_verdict(lhs_formula, rhs_formula):
    """Directional verdict for two predicates whose SPACE descriptors corefer
    via the compound-vs-decomposed pattern, or None when inapplicable (caller
    falls back to the default pairwise comparison — so unrelated pairs are
    unaffected).

    When the places corefer, the verdict reflects the spatial-relation type
    implication: a plain "near place" entails a disjunctive "on or near" =
    OR(stay in place, near place) (GENERAL_IMPLICATION), but not vice versa
    (INDIFFERENT); identical relation sets give EQUIVALENT."""
    if not HOnKSingleton.isReady():
        return None
    lhs_vals = _space_geo_values(lhs_formula)
    rhs_vals = _space_geo_values(rhs_formula)
    if not lhs_vals or not rhs_vals:
        return None
    kb = HOnKSingleton.get()
    if not _spaces_corefer(lhs_vals, rhs_vals, kb):
        return None
    lhs_rel = _spatial_relation_labels(lhs_vals)
    rhs_rel = _spatial_relation_labels(rhs_vals)
    if not lhs_rel or not rhs_rel or lhs_rel == rhs_rel:
        return CasusHappening.EQUIVALENT
    if lhs_rel <= rhs_rel:
        return CasusHappening.GENERAL_IMPLICATION
    return CasusHappening.INDIFFERENT


def _space_values_with_negation(formula):
    """Every value under the SPACE key of a predicate, negated values included, across all SPACE entries. A negated
    predicate is not unwrapped: its SPACE is inside the negation's scope."""
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)) or not formula.properties:
        return []
    out = []
    for k, vs in formula.properties:
        if k == "SPACE":
            out.extend(vs if isinstance(vs, tuple) else (vs,))
    return out


def _space_negated_place_contradiction(lhs_formula, rhs_formula, compare_value, compare_formula) -> bool:
    """True when one predicate puts the event in a place and the other puts the same event explicitly not in that
    place: "There is traffic in the Newcastle city centre" against "Newcastle has traffic but not in the city centre",
    be(traffic)[SPACE: city centre, SPACE: Newcastle] against be(traffic)[SPACE: NOT(city centre), SPACE: Newcastle].

    Both tests are the comparator's own. `compare_value(p, NOT(q))` is EXCLUSIVES only when p and q are the same place
    (a different place, "not at Haymarket" against "at Monument", stays indifferent), and `compare_formula` on the two
    originals is EXCLUSIVES only when relation and arguments match, so the two describe one event. The check is needed
    on the originals because the expansion search compares one have(place, traffic) per place, where the shared
    Newcastle matches, and drops the exclusion between same-relation originals as WordNet antonym noise."""
    for a, b in ((lhs_formula, rhs_formula), (rhs_formula, lhs_formula)):
        negated = [v for v in _space_values_with_negation(b) if isinstance(v, FNot) and isinstance(v.arg, FVariable)]
        if not negated:
            continue
        plain = [v for v in _space_values_with_negation(a) if isinstance(v, FVariable)]
        if any(compare_value(p, n) == CasusHappening.EXCLUSIVES for p in plain for n in negated):
            return compare_formula(lhs_formula, rhs_formula) == CasusHappening.EXCLUSIVES
    return False


def _space_relation_narrowing(lhs_formula, rhs_formula) -> bool:
    """True when LHS only commits to a broader spatial relation than RHS.

    Example: a predicate located "on or near" a place carries
    OR(stay in place, near place), while another says simply "near place".
    The values are related enough to be partial support, but LHS should not
    fully entail RHS because "on" is also allowed.

    A Police.uk record's own "On or near X" is read as the vicinity of X: it
    fully supports "near X" and "on or near X" for the same name X, and stays
    partial for any other relation that includes being on or at X. A different
    name, even one HOnK relates to X, gets no such reading.
    """
    for lhs_pred in _iter_predicate_atoms(lhs_formula):
        for rhs_pred in _iter_predicate_atoms(rhs_formula):
            if getattr(lhs_pred, "rel", None) != getattr(rhs_pred, "rel", None):
                continue
            for lhs_val in _space_geo_values(lhs_pred):
                lhs_rel = _spatial_relation_labels_for_value(lhs_val)
                if not lhs_rel:
                    continue
                record_point = _is_police_uk_record_point(lhs_val.name) and lhs_rel == _ON_OR_NEAR_LABELS
                for rhs_val in _space_geo_values(rhs_pred):
                    if not _space_values_name_compatible(lhs_val, rhs_val):
                        continue
                    rhs_rel = _spatial_relation_labels_for_value(rhs_val)
                    if record_point and _name_tokens(rhs_val.name) == _name_tokens(lhs_val.name):
                        if rhs_rel and rhs_rel != lhs_rel and "stay in place" in rhs_rel:
                            return True
                        continue
                    if rhs_rel and rhs_rel < lhs_rel:
                        return True
    return False


_ON_OR_NEAR_LABELS = frozenset({"stay in place", "near place"})


def _iter_predicate_atoms(formula):
    if isinstance(formula, FNot):
        formula = formula.arg
    if isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        yield formula
        return
    if isinstance(formula, (FAnd, FOr)):
        for arg in getattr(formula, "args", ()) or ():
            yield from _iter_predicate_atoms(arg)


def _spatial_relation_labels_for_value(value):
    if not isinstance(value, FVariable) or not value.properties:
        return set()
    labels = set()
    for key, val in value.properties:
        if key != "type":
            continue
        for item in (val if isinstance(val, tuple) else (val,)):
            labels |= _collect_label_strings(item)
    return labels & _SPATIAL_RELATION_LABELS


def _space_values_name_compatible(lhs_val, rhs_val):
    lhs_name = getattr(lhs_val, "name", None)
    rhs_name = getattr(rhs_val, "name", None)
    if not lhs_name or not rhs_name:
        return False
    if lhs_name == rhs_name or lhs_name.lower() == rhs_name.lower():
        return True
    if not HOnKSingleton.isReady():
        return False
    return _geo_names_compatible(HOnKSingleton.get(), lhs_name, rhs_name)
