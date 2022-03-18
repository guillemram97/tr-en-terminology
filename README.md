# Incorporating a terminology list to a MT model
Our method is based on [Dinu's method](https://aclanthology.org/P19-1294.pdf) for incorporating a terminology list. The idea is that the model learns
when a lexical constraint is applied and doesn't learn the terminology words.
The original method used factors. However, we are applying this method at word-level:

**Input**: `All <0> alternates <1> Stellvertreter <2> shall be elected for one term`

**Output**: `Alle Stellvertreter werden fur eine Amtszeit gewahlt` 
  
This was the most used method in [terminology shared task of WMT2021](https://aclanthology.org/2021.wmt-1.69/).

---

## Finetunning the original model
The original model is a MT model for English to Turkish and vice versa. We used the same training data as the original model to avoid a change in performance. However, the training data was modified to include terminology tags: we identified word pairs from the [MUSE dictionary](https://github.com/facebookresearch/MUSE) that appeared in both the source and target sentence:

`In our Solar System there are perhaps more than billion huge, <0> stray <1> başıboş <2> meteors that would cause serious consequences should they colli de .`

However, we only include the tags in the source sentence and not in the target sentence. 
