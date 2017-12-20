## Crowds_and_Networks-WIKI

### A simple realization of the algorithm proposed in "Mining Missing Hyperlinks from Human Navigation Traces: A Case Study of Wikipedia".

### Method:
### step1. Collect paths from the paths_finished.tsv .
### step2. Generate pairs.
### step3. Filter.
       * For each path p=<p_0, p_1, ..., p_n=t>, only the second half of paths is considered.
       * The filter process according to anchor is not implemented because of the limit of the dataset (no reference).
### step4. Rank the candidates.
       * Three methods are implemented separately: MW, SVD, Frequency.

## Results
      * The result is stored in the file "results.json" with the form {"Paths:[[path1], [path2], ...], "Target":..., "MW":[candidates], "SVD":[candidates], "Frequency":[candidates]}
      * May use json.loads() to load it.
