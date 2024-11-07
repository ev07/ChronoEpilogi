python3 results.py NoisyVAR_8000 1.0 0.0001 fbe_m ARDLModel 
python3 results.py NoisyVAR_2500 0.4 0.0001 fbe_m ARDLModel 
python3 results.py NoisyVAR_500 2.0 0.0001 fbe_m ARDLModel 

python3 results.py NoisyVAR_8000 1.0 0.0001 fe ARDLModel 
python3 results.py NoisyVAR_2500 0.4 0.0001 fe ARDLModel 
python3 results.py NoisyVAR_500 2.0 0.0001 fe ARDLModel 

python3 results_others.py NoisyVAR_8000 1.0  GroupLasso ARDLModel 
python3 results_others.py NoisyVAR_2500 0.4  GroupLasso ARDLModel 
python3 results_others.py NoisyVAR_500 2.0  GroupLasso ARDLModel 



python3 results.py electricity_long 1.0 0.0001 fe all
python3 results.py solar_long 1.0 0.0001 fe all
python3 results.py traffic_long 1.0 0.0001 fe all

python3 results_others.py electricity_long 1.0 GroupLasso all
python3 results_others.py solar_long 1.0 GroupLasso all
python3 results_others.py traffic_long 1.0 GroupLasso all

python3 results_others.py electricity_long 1.0 NoSelection all
python3 results_others.py solar_long 1.0 NoSelection all
python3 results_others.py traffic_long 1.0 NoSelection all


