# CIKM PAM
Compact multi-relational network representation using prime numbers

Accompanying code for CIKM2023 submission, please do not redistribute.

The requirements for this project can be installed by running:

```python
pip -r requirements.txt
````

Python version used: Python 3.9.16

Data are expected to be downloaded in a folder name data, here on the top-level folder.

The datasets used can be downloaded all in one  from [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/CjweAbP5iMSlc3m).


The *pam_creation.py*,*utils.py*, *grakel_utils.py* files, contain functionality code to support the proposed framework and facilitate the experiments.

The rest of the files are used to reproduce the results present in the article:
- The *scalability_test.py* reproduces the usability results presented in Section 3.1.
- The *relation_prediction.py* reproduces the results presented in Section 3.2.
- The *graph_classification_with_gridsearch.py* reproduces the graph-kernel comparison results presented in Section 3.3. In order to calculate the Deltas for the performance on time and accuracy you have to run the *graph_classification_calculate_deltas.py*. You can either run it as is, with the already calculated results (saved in *gc_results.csv*) or re-run the *graph_classification_with_gridsearch.py* script to generate the new results.


This will be updated regularly.
