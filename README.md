# seasonal-influenza-xyonix-blog
Source used to create XYONIX blog for time series forecasting of seasonal influenza via Neural Basis Expansion Analysis for interpretable Time Series forecasting (N-BEATS).

![sample state ILI series](images/ili_state_samples.png)

# resources

* XYONIX blog: https://www.xyonix.com/blog/using-ai-to-improve-sports-performance-amp-achieve-better-running-efficiency
* XYONIX blog source: https://github.com/xyonix/seasonal-influenza-xyonix-blog
* N-BEATS paper: https://arxiv.org/abs/1905.10437
* N-BEATS source: https://github.com/philipperemy/n-beats


# installation

```
make clean install
make run-jupyter
```

After launching the notebook, select the `xyonix-flu` kernel as illustrated below:

![select xyonix-flu kernel](images/kernel_selection.png)

# model training and evaluation

Run the cells in the `flu-foreacasting.ipynb` Jupyter notebook to train and evaluate selected N-BEATS state models.

![California error heatmap](images/california_error_heatmap.png)
