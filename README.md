## This repo includes the codes for the submission: Breaking Sparsity Barrier between Students and Concepts for Cognitive Diagnosis Models in Intelligent Education 

Our environment:
```
Python 3.9.7 
torch 2.0.1
pandas 1.3.4
scikit-learn 0.24.2
networkx 2.6.3
```


### 1. Run codes on ASSIST dataset
```
cd assist/codes
python main_our.py
```

We also include two representative baseline models. Please refer to main_NCDM.py and main_KaNCD.py for more details.  



### 2. Run codes on MOOC-Radar dataset
This dataset is too large to upload in a repo. Please refer to https://github.com/THU-KEG/MOOC-Radar and download the [coarse version](https://cloud.tsinghua.edu.cn/d/5443ee05152344c79419/). 

Please first run "cd mooc/codes" and "python divide_data.py" to preprocess data. Then, you can run the codes by "python main_our.py". 
