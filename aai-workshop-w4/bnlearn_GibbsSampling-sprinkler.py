import bnlearn as bn
from collections import Counter

model = bn.import_DAG('sprinkler', CPD=True)

QUERY_RANDVAR = 'Wet_Grass'
EVIDENCE = {'Sprinkler':0, 'Rain':1} # meaning=> 1: 'True', 0: 'False'
NUM_SAMPLES = 100

# samples N states using the Gibbs sampling algorithm
samples = bn.sampling(model, n=NUM_SAMPLES, methodtype='gibbs')
print("\nAPPROXIMATE INFERENCE results:")
print("samples:\n%s" % (samples))

# filters out samples that are compatible with the evidence 
compatible_samples = samples[(samples['Sprinkler'] == EVIDENCE['Sprinkler'])][(samples['Rain'] == EVIDENCE['Rain'])][QUERY_RANDVAR]
print("compatible_samples:\n%s" % (compatible_samples))

# calculates the total count of events with relevant evidence and 
# prints a probability distribution of the query random variable
frequencies = Counter(compatible_samples)
total_count = sum(frequencies.values())
for value, count in frequencies.items():
    print("P(%s=%s): %s" % (QUERY_RANDVAR, value, {count/total_count}))

# these results are only displayed to check the expected probability distribution, 
# to see how far the approximate inference results are from exact inference.
print("\nEXACT INFERENCE results:")
inference_result = bn.inference.fit(model, variables=[QUERY_RANDVAR], evidence=EVIDENCE)
print(inference_result)
