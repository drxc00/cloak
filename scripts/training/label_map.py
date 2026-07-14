"""
ai4privacy fine-grained labels → Cloak Core-3 mapping.

This is the single source of truth for which PII types the NER classifier is
trained to detect.  Anything not explicitly mapped here is treated as "O"
(not redacted by the NER stage).

The label set observed in the bundled ai4privacy snapshot
(scripts/training/datasets/{train,validation}/):
  ACCOUNTNUM, BUILDINGNUM, CITY, CREDITCARDNUMBER, DATEOFBIRTH,
  DRIVERLICENSENUM, EMAIL, GIVENNAME, IDCARDNUM, PASSWORD, SOCIALNUM,
  STREET, SURNAME, TAXNUM, TELEPHONENUM, USERNAME, ZIPCODE

Labels owned by Cloak's regex + secrets stages (run before NER):
  EMAIL, CREDITCARDNUMBER, TELEPHONENUM, SOCIALNUM, DATEOFBIRTH, PASSWORD,
  ACCOUNTNUM, DRIVERLICENSENUM, IDCARDNUM, TAXNUM
These are mapped to O because NER only sees leftover gaps — training on them
wastes model capacity and invites double-detection.
"""

AI4_TO_CLOAK: dict[str, str | None] = {
    # ai4privacy uses GIVENNAME / SURNAME.
    # Include the plan's FIRSTNAME / MIDDLENAME / LASTNAME aliases in case a
    # dataset revision reintroduces them.
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "FIRSTNAME": "NAME",
    "MIDDLENAME": "NAME",
    "LASTNAME": "NAME",
    "USERNAME": "USERNAME",
    "NAME": "NAME",
    "ADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",  # alias seen in older versions
    "SECONDARYADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "COUNTY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "EMAIL": None,
    "CREDITCARDNUMBER": None,
    "TELEPHONENUM": None,
    "SOCIALNUM": None,
    "DATEOFBIRTH": None,
    "PASSWORD": None,
    "ACCOUNTNUM": None,
    "DRIVERLICENSENUM": None,
    "IDCARDNUM": None,
    "TAXNUM": None,
    "COMPANYNAME": None,
    "JOBTITLE": None,
    "JOBAREA": None,
    "NEARBYGPSCOORDINATE": None,
}

CLOAK_TYPES: list[str] = ["NAME", "ADDRESS", "USERNAME"]

# BIO encoding: for N entity types → O + B-<TYPE> + I-<TYPE> = 2N+1 labels.
_BIO_PREFIXES: list[str] = ["B", "I"]
BIO_LABELS: list[str] = ["O"] + [f"{p}-{t}" for t in CLOAK_TYPES for p in _BIO_PREFIXES]
# → ["O","B-NAME","I-NAME","B-ADDRESS","I-ADDRESS","B-USERNAME","I-USERNAME"]

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

# Quick lookup: is this an entity label (not O)?
_ENTITY_LABELS: set[str] = set(BIO_LABELS) - {"O"}


def is_entity(label: str) -> bool:
    """True if *label* is a BIO entity tag (B-… or I-…)."""
    return label in _ENTITY_LABELS


def entity_type_from_bio(bio_label: str) -> str | None:
    """Extract the Cloak type from a BIO label, e.g. "B-NAME" → "NAME"."""
    if bio_label == "O" or "-" not in bio_label:
        return None
    return bio_label.split("-", 1)[1]
