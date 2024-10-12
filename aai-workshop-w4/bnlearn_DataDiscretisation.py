import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# choices of scoring functions: bic, k2, bdeu, bds, aic
TRAINING_DATA = '.\data\cardiovascular_data-original-train.csv'
SCORING_FUNCTION = 'aic'
MAX_ITERATIONS=20000000
VISUALISE_STRUCTURE=False

# data loading using pandas, but only the first 1K rows due to memory issues
data = pd.read_csv(TRAINING_DATA, encoding='latin', nrows=2000)
print("DATA:\n", data)

# definition of directed acyclic graph (predefined Naive Bayes structure -- only for discretising data)
edges = [('target', 'ï»¿age'),('target', 'gender'),('target', 'height'),('target', 'weight'),
         ('target', 'ap_hi'),('target', 'ap_lo'),('target', 'cholesterol'),('target', 'gluc'),
		 ('target', 'smoke'),('target', 'alco'),('target', 'active')]

# performs discretisation of continuous data for the columns specified and structure provided
# the output of this steps is later used for training a Bayesian network -- no longer the original dataset
continuous_columns = ["ï»¿age", "height", "weight", "ap_hi", "ap_lo"]
discrete_data = bn.discretize(data, edges, continuous_columns, max_iterations=1, verbose=3)
for randvar in discrete_data:
    print("VARIABLE:",randvar)
    print(discrete_data[randvar])

# structure learning using a chosen scoring function as per SCORING_FUNCTION
model = bn.structure_learning.fit(discrete_data, methodtype='hillclimbsearch', scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
num_model_edges = len(model['model_edges'])
print("model=",model)
print("num_model_edges="+str(num_model_edges))

# visualise the learnt structure
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()

# parameter learning using Maximum Likelihood Estimation (MLE) and discretised data -- not original data
DAG = bn.make_DAG(model['model_edges'])
model = bn.parameter_learning.fit(DAG, discrete_data, methodtype="maximumlikelihood")
print("model=",model)

# probabilistic inference for a test example -- not part of the training data
discretised_evidence = { 
'ï»¿age':bn.discretize_value(discrete_data["ï»¿age"], 17623), 
'height':bn.discretize_value(discrete_data["height"], 169), 
'weight':bn.discretize_value(discrete_data["weight"], 82), 
'ap_hi':bn.discretize_value(discrete_data["ap_hi"], 150), 
'ap_lo':bn.discretize_value(discrete_data["ap_lo"], 100), 
'gender':2, 'cholesterol':1, 'gluc':1, 'smoke':0, 'alco':0, 'active':1}
inference_result = bn.inference.fit(model, variables=['target'], evidence=discretised_evidence)
print(inference_result)

