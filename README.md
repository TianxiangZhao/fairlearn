# fair learning without sensitive attributes
Official code for WSDM 2022 paper: [Towards Fair Classifiers Without Sensitive Attributes: Exploring Biases in Related Features](https://arxiv.org/abs/2104.14537)


## Dataset
Two datasets, Law_school and Compas, are provided in this project. 

## Algorithm
We provide several algorithms for fair learning:
-corre: constrain correlation with sensitive attributes
-groupTPR: regularize group-wise true positive rate for fairness
-remove: remove related attributes
-learnCorre: learn to constrain correlation with related attributes

## Example
An example on adult dataset is provided here:
<code>python main.py --method="learnCorre" --dataset=adult --related age --r_weight 0.1 --weightSum=0.1  --beta=0.4 --seed=42</code>
