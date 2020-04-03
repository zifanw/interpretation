# interpretation
Implementation of the paper https://arxiv.org/abs/2002.07985


## Dependencies
- Python3
- Keras 
- Tensorflow (1.x)
- Scipy 
- Numpy
- Scikit-learn
- tqdm 

## Model
Use the pretrained model from keras-applicaiton https://keras.io/applications/#applications


## Evaluation Metric

The evalution of N-Ord, S-Ord, TPN and TPS are included in `evaluations.py`. 

## Implementation of Attribution methods

A quick guide to the implementations for attribution methods



### Saliency Map

Original Paper: https://arxiv.org/pdf/1312.6034.pdf

Implementation: `KerasAttr.saliency_map()`  in `\explainer\Attribution.py` 



### Integrated Gradient

Original Paper: https://arxiv.org/pdf/1703.01365.pdf

Implementation: `KerasAttr.integrated_grad()`  in `\explainer\Attribution.py` 



### Smooth Gradient

Original Paper: https://arxiv.org/pdf/1706.03825.pdf

Implementation: `KerasAttr.smooth_grad()`  in `\explainer\Attribution.py` 



### DeepLIFT

Original Paper: https://arxiv.org/pdf/1704.02685.pdf

Implementation: Use the RevealCancner version by  https://github.com/kundajelab/deeplift



### Layerwise Relevance Propagation

Original Paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140

Paper discussing the implementation of LRP-$\alpha 2 \beta 1$: https://arxiv.org/pdf/1706.07979.pdf

Implementaion: `KerasAttr.lrpa2b1()`  in `\explainer\Attribution.py` , 

​							 which is a wrapper of https://github.com/atulshanbhag/Layerwise-Relevance-Propagation. 



### Guided Backbropagation

Original Paper: https://arxiv.org/pdf/1412.6806.pdf

Implementaion: `GuidedModel()`  in `\explainer\Attribution.py`



### GradCAM

Original Paper: https://arxiv.org/pdf/1610.02391.pdf

Implementaion: `KerasAttr.gradcan()`  in `\explainer\Attribution.py`

​							which is a wrapper of the implementation by https://github.com/eclique/keras-gradcam


