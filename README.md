# Spectral Graph Convolution for Infant Brain segmentation and age prediction
The code used for cortical surface parcellation and age regression (birth age, scan age)

### Training
The model can be trained using below command:  
```
python3 main_train_parcellation.py --config ./parcellation/config/config_train_parcellation.json --gpu 0
python3 main_train_birth_regression.py --config ./parcellation/config/config_train_birth_regression.json --gpu 0
python3 main_train_scan_regression.py --config ./parcellation/config/config_train_scan_regression.json --gpu 0
```
