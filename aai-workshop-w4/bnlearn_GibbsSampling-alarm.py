import bnlearn as bn
from collections import Counter
from pgmpy.factors.discrete import TabularCPD


QUERY_RANDVAR = 'Burglary'
EVIDENCE = {'JohnCalls':'T', 'MaryCalls':'T'}
NUM_SAMPLES = 100
STRUCTURE=[('Burglary', 'Alarm'), ('Earthquake', 'Alarm'), ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')]

# creates the CPTs of the Burglary Bayes net using the probabilities specified in Russell & Norvig's book.
cpt_B = TabularCPD(variable='Burglary', variable_card=2, 
				   state_names={'Burglary': ['T', 'F']}, 
                   values=[[0.001], [0.999]])
				   
cpt_E = TabularCPD(variable='Earthquake', variable_card=2, 
				   state_names={'Earthquake': ['T', 'F']}, 
                   values=[[0.002], [0.998]]) 
				   
cpt_A = TabularCPD(variable='Alarm', variable_card=2,
                   values=[[0.95, 0.94, 0.29, 0.001],
                           [0.05, 0.06, 0.71, 0.999]],
                   state_names={'Alarm': ['T', 'F'], 'Burglary': ['T', 'F'], 'Earthquake': ['T', 'F']}, 
                   evidence=['Burglary','Earthquake'], evidence_card=[2,2])
				   
cpt_J = TabularCPD(variable='JohnCalls', variable_card=2,
                   values=[[0.9, 0.05],
                           [0.1, 0.95]],
                   state_names={'JohnCalls': ['T', 'F'], 'Alarm': ['T', 'F']}, 
                   evidence=['Alarm'], evidence_card=[2])  	 
				   
cpt_M = TabularCPD(variable='MaryCalls', variable_card=2,
                   values=[[0.7, 0.01],
                           [0.3, 0.99]],
                   state_names={'MaryCalls': ['T', 'F'], 'Alarm': ['T', 'F']}, 
                   evidence=['Alarm'], evidence_card=[2])

print("cpt_B.values=",cpt_B.values)
print("cpt_E.values=",cpt_E.values)
print("cpt_A.values=",cpt_A.values)
print("cpt_J.values=",cpt_J.values)
print("cpt_M.values=",cpt_M.values)

# creates a Bayesian network using the structure and CPTs defined above
model = bn.make_DAG(STRUCTURE, CPD=[cpt_B, cpt_E, cpt_A, cpt_J, cpt_M])

# samples N states using the Gibbs sampling algorithm
samples = bn.sampling(model, n=NUM_SAMPLES, methodtype='gibbs')
samples = samples.replace({0: 'T', 1: 'F'})
print("\nAPPROXIMATE INFERENCE results:")
print("samples:\n%s" % (samples))

# filters out samples that are compatible with the evidence 
compatible_samples = samples[(samples['JohnCalls'] == EVIDENCE['JohnCalls'])][(samples['MaryCalls'] == EVIDENCE['MaryCalls'])][QUERY_RANDVAR]
print("p_burglary:\n%s" % (compatible_samples))

# computes and prints a probability distribution of the query random variable -- using sampled data
frequencies = Counter(compatible_samples)
total_count = sum(frequencies.values())
for value, count in frequencies.items():
    print("P(%s=%s): %s" % (QUERY_RANDVAR, value, {count/total_count}))

# computes and prints a probability distribution of the query random variable -- using exact inference
# these results are for the purposes of checking how far the approximation is to that of exact inference
print("\nEXACT INFERENCE results:")
inference_result = bn.inference.fit(model, variables=[QUERY_RANDVAR], evidence=EVIDENCE)
print(inference_result)
