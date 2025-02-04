#!/usr/bin/env python

# TRAINED_UPLOADED_MODEL_NAME = "evamxb/dev-author-em-clf"
MODEL_STR_INPUT_TEMPLATE = """
<award-details>
    <title>{award_title}</title>
    <abstract>{award_abstract}</abstract>
</award-details>
""".strip()
