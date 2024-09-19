#!/usr/bin/env python

TRAINED_UPLOADED_MODEL_NAME = "evamxb/dev-author-em-clf"
MODEL_STR_INPUT_TEMPLATE = """
<developer-details>
    <username>{dev_username}</username>
    <name>{dev_name}</name>
    <email>{dev_email}</email>
</developer-details>

---

<author-details>
    <name>{author_name}</name>
</author-details>
""".strip()
