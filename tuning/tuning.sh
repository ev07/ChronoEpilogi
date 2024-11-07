python3 first_wave_main.py NoisyVAR_8000 ChronOMP ARDLModel 1
python3 first_wave_main.py NoisyVAR_2500 ChronOMP ARDLModel 1
python3 first_wave_main.py NoisyVAR_500 ChronOMP ARDLModel 1

python3 first_wave_main.py NoisyVAR_8000 BackwardChronOMP ARDLModel 1
python3 first_wave_main.py NoisyVAR_2500 BackwardChronOMP ARDLModel 1
python3 first_wave_main.py NoisyVAR_500 BackwardChronOMP ARDLModel 1

python3 first_wave_main.py NoisyVAR_8000 GroupLasso ARDLModel 1
python3 first_wave_main.py NoisyVAR_2500 GroupLasso ARDLModel 1
python3 first_wave_main.py NoisyVAR_500 GroupLasso ARDLModel 1


python3 first_wave_main.py electricity_long ChronOMP SVRModel 1
python3 first_wave_main.py solar_long ChronOMP SVRModel 1
python3 first_wave_main.py traffic_long ChronOMP SVRModel 1

python3 first_wave_main.py electricity_long NoSelection SVRModel 1
python3 first_wave_main.py NoisyVAR_2500 NoSelection SVRModel 1
python3 first_wave_main.py NoisyVAR_500 NoSelection SVRModel 1

python3 first_wave_main.py electricity_long GroupLasso SVRModel 1
python3 first_wave_main.py solar_long GroupLasso SVRModel 1
python3 first_wave_main.py traffic_long GroupLasso SVRModel 1


python3 second_wave_main.py electricity_long ChronOMP TFTModel SVRModel 1
python3 second_wave_main.py solar_long ChronOMP TFTModel SVRModel 1
python3 second_wave_main.py traffic_long ChronOMP TFTModel SVRModel 1

python3 second_wave_main.py electricity_long NoSelection TFTModel SVRModel 1
python3 second_wave_main.py NoisyVAR_2500 NoSelection TFTModel SVRModel 1
python3 second_wave_main.py NoisyVAR_500 NoSelection TFTModel SVRModel 1

python3 second_wave_main.py electricity_long GroupLasso TFTModel SVRModel 1
python3 second_wave_main.py solar_long GroupLasso TFTModel SVRModel 1
python3 second_wave_main.py traffic_long GroupLasso TFTModel SVRModel 1


python3 second_wave_main.py electricity_long ChronOMP DeepARModel SVRModel 1
python3 second_wave_main.py solar_long ChronOMP DeepARModel SVRModel 1
python3 second_wave_main.py traffic_long ChronOMP DeepARModel SVRModel 1

python3 second_wave_main.py electricity_long NoSelection DeepARModel SVRModel 1
python3 second_wave_main.py NoisyVAR_2500 NoSelection DeepARModel SVRModel 1
python3 second_wave_main.py NoisyVAR_500 NoSelection DeepARModel SVRModel 1

python3 second_wave_main.py electricity_long GroupLasso DeepARModel SVRModel 1
python3 second_wave_main.py solar_long GroupLasso DeepARModel SVRModel 1
python3 second_wave_main.py traffic_long GroupLasso DeepARModel SVRModel 1
