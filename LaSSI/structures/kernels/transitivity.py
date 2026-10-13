"""Whether a verb keeps its dependency target as a direct object.

v4.3 dropped the target of any verb missing from HOnK's TransitiveVerb class. Under HOnK v1.5.0's sense-aware
identity, genuinely transitive verbs such as "forecast" are no longer in that class, so their objects were
silently lost. v4.4 drops the target only for a verb that WordNet knows and lists with no direct-object frame
(e.g. "arrive"); a verb in TransitiveVerb, or one WordNet does not know, keeps it.
"""
from functools import lru_cache

# WordNet verb frames with a direct-object slot ("Somebody ----s something", ditransitives, object + PP/infinitive).
_OBJECT_FRAMES = frozenset({8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 30, 31})


@lru_cache(maxsize=None)
def _wordnet_has_object_frame(lemma: str):
    """True or False if WordNet knows `lemma` as a verb, None if it does not."""
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(lemma, pos=wn.VERB)
    if not synsets:
        return None
    return any(set(l.frame_ids()) & _OBJECT_FRAMES
               for s in synsets for l in s.lemmas() if l.name().lower() == lemma.lower())


def drops_object(lemmas, transitive_verbs) -> bool:
    """True if the kernel should drop the verb's target, i.e. treat the verb as having no direct object."""
    lemmas = {x for x in lemmas if x}
    if transitive_verbs.intersection(lemmas):
        return False
    frames = [_wordnet_has_object_frame(x) for x in sorted(lemmas)]
    return any(f is False for f in frames) and not any(frames)
