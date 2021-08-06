# ForestFireDetection
**Comprehensive Analysis of Forest Fire Detection using Deep Learning Models and Conventional Machine Learning Algorithms**
> Zeynep Hilal Kilimci and Süha Berk Kukuk

> Paper : [International Journal of Computational and Experimental Science and Engineering](https://doi.org/10.22399/ijcesen.950045)
 
 > Abstract : Forest fire detection is a very challenging problem in the field of object detection.
Fire detection-based image analysis have advantages such as usage on wide open
areas, the possibility for operator to visually confirm presence, intensity and the
size of the hazards, lower cost for installation and further exploitation. To
overcome the problem of fire detection in outdoors, deep learning and
conventional machine learning based computer vision techniques are employed to
determine the fire detection when indoor fire detection systems are not capable.
In this work, we propose a comprehensive analysis of forest fire detection using
conventional machine learning algorithms, object detection techniques, deep and
hybrid deep learning models. The contribution of this work to the literature is to
analyze different classification and object detection techniques in more details that
is not addressed before in order to detect forest fire. Experiment results
demonstrate that convolutional neural networks outperform other methods with
99.32% of accuracy result.

 
 # Dataset
 You can Download  Data On Google Drive. Link is 
[Google Drive Link](https://drive.google.com/drive/folders/1gHNe0AOk05E68hoG0qMTSPLXUvi6oSsf?usp=sharing)

# Folders
1. Models Folder 
* There are contain algorithms that we are used for fire detection
2. Prepare Data 
* Own data's are collected using Beatifulsoup and Web Scariping. In this Folder contains codes are  that we used 

# Result 
> The following abbreviations are used in the tables: AC: Accuracy, FM: F-measure, PR: Precision, RC: Recall, SVM: Support vector machine, RF: Random Forest, CNN: Convolutional neural network, CNN-GRU: Convolution Neural Network-Gated Recurrent Unit, CNN-LSTM: Convolutional neural network-long short-term memory, SSD: Single shot detector, Faster R-CNN: Faster recurrent-convolutional neural network, Avg: Average. The best results are obtained for each dataset in the Table 1 and Table 2 after experiments of hyperparameter tuning. The best performance results are also demonstrated in boldface in all tables. In Table 1, the performance results of all classification models according to evaluation metrics in the first dataset (DS1) are demonstrated.

<p align="center">
    <img src="https://github.com/shbkukuk/ForestFireDetection/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202021-08-06%20114109.png"> <br />
    <em> 
    </em>
</p>

>In Table 2, the performance results of all classification models according to evaluation metrics in the second dataset (DS2) are demonstrated.
<p align="center">
    <img src="https://github.com/shbkukuk/ForestFireDetection/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202021-08-06%20114143.png"> <br />
    <em> 
    </em>
</p>



# Packages Reqs:
* Keras
* TensorFlow
* Numpy
* Matplotlib
* Scipy
* ApiClient
* Imutils
* OpenCv
* Argparse
* MoviePy
* SimpleJson
* PyFcm
* Oauth2client
* Httplib2
