# Scientific Software (Predictive) Models

Computational predictive models to assist in the identification, classification, and study of scientific software.

## Models

### Developer-Author Entity Matching

This model is a binary classifier that predicts whether a developer and an author are the same person. It is trained on a dataset of 3000 developer-author pairs that have been annotated as either matching or not matching.

#### Usage

Given a set of developers and authors, we use the model on each possible pair of developer and author to predict whether they are the same person. The model returns a list of only the found matches in `MatchedDevAuthor` objects, each containing the developer, author, and the confidence of the prediction.

```python
from sci_soft_models import dev_author_em

devs = [
    dev_author_em.DeveloperDetails(
        username="evamaxfield",
        name="Eva Maxfield Brown",
    ),
    dev_author_em.DeveloperDetails(
        username="nniiicc",
    ),
]

authors = [
    "Eva Brown",
    "Nicholas Weber",
]

matches = dev_author_em.match_devs_and_authors(devs=devs, authors=authors)
print(matches)
# [
#   MatchedDevAuthor(
#       dev=DeveloperDetails(
#           username='evamaxfield',
#           name='Eva Maxfield Brown',
#           email=None,
#       ),
#       author='Eva Brown',
#       confidence=0.9851127862930298
#   )
# ]
```

<summary><h2>Extra Notes</h2></summary>
<details>

### Developer-Author-EM Dataset

This model was originally created and managed as a part of [rs-graph](https://github.com/evamaxfield/rs-graph) and as such, to regenerate the dataset for annotation, the following steps can be taken:

```bash
git clone https://github.com/evamaxfield/rs-graph.git
cd rs-graph
git checkout c1d8ec89
pip install -e .
rs-graph-modeling create-developer-author-em-dataset-for-annotation
```

[Link to annotation set creation function](https://github.com/evamaxfield/rs-graph/blob/c1d8ec8999a7a26e5d1669e9531adaad13245393/rs_graph/bin/modeling.py#L168).

</details>