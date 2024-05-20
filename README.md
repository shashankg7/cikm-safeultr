
# CIKM 2024 submission number 65 Practical and Robust Safety Guarantees for Advanced Counterfactual Learning to Rank

==============================

Anonymous source code for running the experiments for the CIKM 2024 submission number 65 titled, "Practical and Robust Safety Guarantees for Advanced Counterfactual Learning to Rank". 

Steps to run the code: 
-----------------------

1) Create a new conda envioronment and install the dependencies for the project via the requirements.txt file

```
conda create --name safe_ultr
pip3 install -r requirements.txt
```

2) Download the LTR datasets ([MSLR30K](https://www.microsoft.com/en-us/research/project/mslr/), [Yahoo LTR](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c), [ISTELLA-S LETOR](https://istella.ai/datasets/letor-dataset/)). 


3) Put them under a common path, under the folder names: MSLR30K, Yahoo, ISTELLA and add the folder path to the config/config.yaml file under the "root_dir" tag under the "dataset" header. 


4) Create a directory to store the log data and add the path to config/config.yaml under dataset:predict_dir yaml tag


4) Train the logging policy on 3% of the relevance (full-information) dataset via the following command:

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
