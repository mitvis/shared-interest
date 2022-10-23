# Shared Interest
This repository contains code for:

[Shared Interest: Measuring Human-AI Alignment to Identify Recurring Patterns in Model Behavior](https://arxiv.org/abs/2107.09234)  
Authors: [Angie Boggust](http://angieboggust.com/), [Benjamin Hoover](https://www.bhoov.com/), [Arvind Satyanarayan](https://arvindsatya.com/), and [Hendrik Strobelt](http://hendrik.strobelt.com/)

Shared Interest is a method to quantify model behavior by comparing human and model decision making. In Shared Interest, human decision is approximated via ground truth annotations and model decision making is approximated via saliency. By quantifying each instance in a dataset, Shared Interest can enable large-scale analysis of model behavior.

## Using Shared Interest
### Step 0: Clone this repo.

### Step 1: Install Shared Interest.
Install the method locally for use in other development projects. It can be referenced as `shared_interest` within this package and in other locations.  
```
cd shared-interest
pip install -e git+https://github.com/mitvis/shared-interest.git#egg=shared_interest
```

### Step 2: Install interpretability methods (optional).
Shared Interest relies on saliency methods to compute model behavior. The examples within this repo rely on the repo [`interpretability_methods`](https://github.mit.edu/aboggust/interpretability_methods). If you are planning to run the example notebook as is, then install the `interpretability_methods`. Otherwise, you can skip this step.  
```pip install git+https://github.com/aboggust/interpretability-methods.git```

### Step 3: Install the requirements.
Requirements are listed in `requirements.txt`. Install via:  
```pip install -r requirements.txt```

### Step 4: Use Shared Interest to analyze model behavior.
See [notebook](https://github.com/mitvis/shared-interest/blob/main/shared_interest/examples/shared_interest_example.ipynb) for example usage!

