
# CIKM 2024 full paper Practical and Robust Safety Guarantees for Advanced Counterfactual Learning to Rank

==============================

Source code for running the experiments for the CIKM 2024 full paper titled, "Practical and Robust Safety Guarantees for Advanced Counterfactual Learning to Rank". 

Steps to run the code: 
-----------------------

1) Create a new conda envioronment and install the dependencies for the project via the requirements.txt file

```
conda create --name safe_ultr
pip3 install -r requirements.txt
```

2) Download the LTR datasets ([MSLR30K](https://www.microsoft.com/en-us/research/project/mslr/), [Yahoo LTR](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c), [ISTELLA-S LETOR](https://istella.ai/datasets/letor-dataset/)). 


3) Put them under a common path, under the folder names: MSLR30K, Yahoo, ISTELLA and add the folder path to the config/config.yaml file under the "root_dir" tag under the "dataset" header.

### Data pre-processing: 

1. Some feature values in the datasets is unusually high, so it's a good idea to remove feature values > say 98 percentile in the dataset.


#### Next steps:

4) Create a directory to store the log data and add the path to config/config.yaml under dataset:predict_dir yaml tag


5) Train the logging policy on 3% of the relevance (full-information) dataset via the following command:

```
python logging_policy.py --noise 0.1 --job_id 1 --dataset Yahoo --num_sessions 100 --fraction 0.030000 --T 1.0 --deterministic False
```

This command will train a logging policy and store the model in the LTR_datasets folder. It will generate 100 sessions (queries) along with a displayed rankings and clicks simulated via the position based model and store the log dataset in the predict_dir folder. 


5) To train the baseline IPS model on the generated logged data, run the following command
```
python cltr_pytorch.py --risk 0 --noise 0.5 --job_id 1 --dataset Yahoo --num_sessions 100 --T 1.0
```


6) To train the baseline Doubly-robust model on the generated logged data, run the following command
```
python cltr_pytorch_dr.py --risk 0 --noise 0.5 --job_id 1 --dataset Yahoo --num_sessions 100 --T 1.0
```

7) To train the proposed safe Doubly Robust model on the generated logged data, run the following command
```
python cltr_pytorch_dr.py --risk 1 --noise 0.5 --job_id 1 --dataset Yahoo --num_sessions 100 --T 1.0
```

8) To train the proposed PRPO model on the generated logged data, run the following command
```
python cltr_pytorch_ppo.py --risk 1 --noise 0.5 --job_id 1 --dataset Yahoo --num_sessions 100 --T 1.0

```

## Paper
If you use our code in your research, please remember to cite our work:

```BibTeX
    @inproceedings{gupta-2024-practical,
      author = {Gupta, Shashank and Oosterhuis, Harrie and de Rijke, Maarten},
      booktitle = {CIKM 2024: 33rd ACM International Conference on Information and Knowledge Management},
      date-added = {2024-07-16 08:06:13 -0400},
      date-modified = {2024-07-16 08:08:20 -0400},
      month = {October},
      publisher = {ACM},
      title = {Practical and Robust Safety Guarantees for Advanced Counterfactual Learning to Rank},
      year = {2024}}

    @inproceedings{gupta-2023-safe,
      author = {Gupta, Shashank and Oosterhuis, Harrie and de Rijke, Maarten},
      booktitle = {SIGIR 2023: 46th international ACM SIGIR Conference on Research and Development in Information Retrieval},
      date-added = {2023-04-05 14:39:36 +0100},
      date-modified = {2023-07-20 06:44:36 +0200},
      month = {July},
      pages = {249--258},
      publisher = {ACM},
      title = {Safe Deployment for Counterfactual Learning to Rank with Exposure-Based Risk Minimization},
      year = {2023}},
    }
```
