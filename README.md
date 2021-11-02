# Spectral Graph Convolution for Infant Brain segmentation and age prediction
The code used for cortical surface parcellation and age regression (birth age, scan age)

### Spectral embedding
The aligned spectral embedding for brain surfaces can be computed using the following repository
  ```
  https://github.com/kharitz/aligned_spectral_embedding.git
  ```
  
### Dataset
A "dataset" folder with the aligned spectral embedding is taken as input for brain surface analyis 

### Training
The model can be trained using below command:  
To train parcellation/segmentation newtork
```
python3 main_train_parcellation.py --config ./parcellation/config/config_train_parcellation.json --gpu 0
```

To train birth age regression newtork
```
python3 main_train_birth_regression.py --config ./parcellation/config/config_train_birth_regression.json --gpu 0
```

To train scan age regression newtork
```
python3 main_train_scan_regression.py --config ./parcellation/config/config_train_scan_regression.json --gpu 0
```

