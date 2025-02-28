# factual-recall-iclr

Code for [Understanding Factual Recall in Transformers via Associative Memories](https://arxiv.org/abs/2412.06538). Eshaan Nichani, Jason D. Lee, Alberto Bietti. ICLR 2025.

- linear_AM.ipynb: generates scaling of linear associative memory (Figure 1, left)
- mlp_sweep.py: training sweep for MLP associative memory (Figure 1, right; Figure 6).
- mlp_sweep_plots.ipynb: plotting code for MLP associative memory

- run.py, model.py, task.py: helper code for factual recall task
- fact_recall_sweep.py: training sweep for factual recall task (Figure 4 (left); Figures 7-9)
- param_scaling_sweep.py: training sweep for Figure 4 (right)
- plotting.ipynb: plotting code for Figures 4, 7, 8, 9

- sequential-learning.ipynb: code for generating Figure 5
