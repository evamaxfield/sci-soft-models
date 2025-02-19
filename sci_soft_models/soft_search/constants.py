#!/usr/bin/env python

TRAINED_UPLOADED_MODEL_NAME = "evamxb/nsf-soft-search-v2"
MODEL_STR_INPUT_TEMPLATE = """
<award-details>
    <title>{award_title}</title>
    <abstract>{award_abstract}</abstract>
    <outcomes>{award_outcomes}</outcomes>
</award-details>
""".strip()
