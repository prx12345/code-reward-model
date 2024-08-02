# todo

- run generation. waiting on colab quota. try N=4 first to halve it
- 03_label is slow, 4 workers isn't enough on a 2-core colab box
- humaneval prompts have doctests in them, mbpp don't. this probably matters
  more than i wrote in the readme
- try pass-fraction instead of binary labels. 2/3 tests passing is not the
  same as 0/3
- features.py: add n_lambdas, n_nested_loops. maybe cyclomatic complexity
- the fingerprint probe in stage 8 is the number i most want to see
- check whether the 0.95 abort threshold is too low once i have real numbers
