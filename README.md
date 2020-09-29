# wifi-sensor

## Installation

1. Install Anaconda
2. Create conda virtual env called py37
   > ```conda create -n py37 python=3.7```
3. Activate the virtual env:
   > ```conda activate py37```

4. Install package
   > ``` pip install tensorflow keras numpy scipy sklearn flask```
5. Clone the repo with the following command
   >```git clone https://github.com/wenjinzhang/wifi-sensor```

## Activities Recognition[real_time]
### Model training
- Preparing the training dataset as the following structure
  ``` 
  ~/wifi-sensor 
    dataset/
        training/
            raising hand/(directories of CSI mat files, it supports nesed directirse)
            squating/
            walking/
            [some other activities]/
        test/
            raising hand/
            squating/
            walking/
            [some other activities]/
    
    model/
        .../(saved trained model checkpoint)
  ```
-  Genereate the dataset from dataset folder:
    >```python dataset2.py```
-  Training model:
-   > ```python activity_model_construction.py```

### Real_time Inference
- Suitable config profile_construction2.py: host
- Start CSI data receiver service
  >```python profile_construction2.py```
- Start Real_time Inference service
  >```python main.py```
- Put ```http://127.0.0.1:5000``` to your brower
![image](https://github.com/wenjinzhang/wifi-sensor/blob/master/img/realtime.png)

