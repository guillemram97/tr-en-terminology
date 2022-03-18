# Incorporating a terminology list to a MT model
Our method is based on [Dinu's method](https://aclanthology.org/P19-1294.pdf) for incorporating a terminology list. The idea is that the model learns
when a lexical constraint is applied and doesn't learn the terminology words.
The original method used factors. However, we are applying this method at word-level:

**Input**: `All <0> alternates <1> Stellvertreter <2> shall be elected for one term`

**Output**: `Alle Stellvertreter werden fur eine Amtszeit gewahlt` 
  
This was the most used method in [terminology shared task of WMT2021](https://aclanthology.org/2021.wmt-1.69/).

---


