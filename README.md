# Scientific Software (Predictive) Models

Computational predictive models to assist in the identification, classification, and study of scientific software.

## Notes

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