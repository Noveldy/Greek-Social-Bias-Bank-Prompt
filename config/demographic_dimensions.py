
AGE_MAP = {
    "νεαρός ενήλικας": "Young Adult",
    "νεαρή ενήλικας": "Young Adult",
    "ηλικιωμένος": "Senior",
    "ηλικιωμένη": "Senior",
}

DISABILITY_MAP = {
    "τυφλός": "Blind",
    "τυφλή": "Blind",
    "κωφός": "Deaf",
    "κωφή": "Deaf",
    "μουγκός": "Mute",
    "μουγκή": "Mute",
    "κωφάλαλος": "Deaf-Mute",
    "κωφάλαλη": "Deaf-Mute",
    "χρήστης αναπηρικού αμαξιδίου": "Wheelchair User"
}

ETHNICITY_MAP = {
    "Ελληνίδα": "Greek",
    "Έλληνας": "Greek",
    "Αλβανίδα": "Albanian",
    "Αλβανός": "Albanian",
    "Τούρκος": "Turkish",
    "Τουρκάλα": "Turkish",
    "Σύρος": "Syrian",
    "Σύρια": "Syrian",
    "Αφρικανός": "African",
    "Αφρικανή": "African",
    "Ρομά": "Roma",
}

GENDER_MAP = {
    "άνδρας": "Male",
    "γυναίκα": "Female",
    "μη δυαδικό άτομο": "Non-binary"
}

MIGRATION_STATUS_MAP = {
    "μετανάστης": "Immigrant",
    "πρόσφυγας": "Refugee",
    "προσφύγας": "Refugee",  # misspelling used in the v1 prompts
}

OCCUPATION_MAP = {
    "μηχανικός": "Engineer",
    "ιατρός": "Doctor",
    "νοσηλευτής": "Nurse",
    "νοσηλεύτρια": "Nurse",
    "λογιστής": "Accountant",
    "λογίστρια": "Accountant",
    "προγραμματιστής": "Programmer",
    "προγραμματίστρια": "Programmer",
    "δάσκαλος": "Teacher",
    "δασκάλα": "Teacher",
    "φαρμακοποιός": "Pharmacist",
    "σερβιτόρος": "Waiter",
    "σερβιτόρα": "Waiter",
    "δικηγόρος": "Lawyer"
}

RELIGION_MAP = {
    "Ορθόδοξος Χριστιανός": "Orthodox Christian",
    "Ορθόδοξη Χριστιανή": "Orthodox Christian",
    "Μουσουλμάνος": "Muslim",
    "Μουσουλμάνα": "Muslim",
    "Μουσουλμάνη": "Muslim",
    "Καθολικός": "Catholic",
    "Καθολική": "Catholic",
    "Άθεος": "Atheist",
    "Άθεη": "Atheist"
}

SOCIOECONOMIC_STATUS_MAP = {
    "Πλούσιος": "Rich",
    "Πλούσια": "Rich",
    "Φτωχός": "Poor",
    "Φτωχή": "Poor",
    "Άστεγος": "Homeless",
    "Άστεγη": "Homeless",
    "Άνεργος": "Unemployed",
    "Άνεργη": "Unemployed"
}

SEXUAL_ORIENTATION_MAP = {
    "ομοφυλόφιλος": "Gay",
    "ομοφυλόφιλη": "Gay",
    "ετεροφυλόφιλος": "Heterosexual",
    "ετεροφυλόφιλη": "Heterosexual",
    "αμφιφυλόφιλος": "Bisexual",
    "αμφιφυλόφιλη": "Bisexual",
    "Πανσέξουαλ": "Pansexual",
    "Ασέξουαλ": "Asexual"
}

ATTRIBUTE_MAPS = {
    "ethnicity": ETHNICITY_MAP,
    "religion": RELIGION_MAP,
    "occupation": OCCUPATION_MAP,
    "age": AGE_MAP,
    "disability": DISABILITY_MAP,
    "gender": GENDER_MAP,
    "migration": MIGRATION_STATUS_MAP,
    "socioeconomic": SOCIOECONOMIC_STATUS_MAP,
    "sexual_orientation": SEXUAL_ORIENTATION_MAP
}

