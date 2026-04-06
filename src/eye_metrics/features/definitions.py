"""Feature set definitions mapping names to regex column patterns."""

FEATURE_SETS: dict[str, list[str]] = {
    "colet": [r"pupil_diameter", r"saccades_", r"fixations_", r"blinks_"],
    "all": [
        r"pupil_.*wv",
        r"pupil_.*ipa",
        r"saccades_.*_(m|(std))",
        r"blinks_",
        r"fixations_.*_(m|(std))",
    ],
    "no_fixations": [r"pupil_.*wv", r"pupil_.*ipa", r"saccades_", r"blinks_"],
    "no_blinks": [r"pupil_.*wv", r"pupil_.*ipa", r"saccades_", r"fixations_"],
    "no_fix_no_blinks": [r"pupil_.*wv", r"pupil_.*ipa", r"saccades_"],
    "no_wavelets": [r"pupil_.*ipa", r"saccades_", r"fixations_"],
    "ipa_wavelets": [r"pupil_.*wv", r"pupil_.*ipa"],
    "pupil": [r"pupil_"],
}
