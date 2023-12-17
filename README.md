# ner
## NER task on MultiNERD dataset

This repository contains codes for fine-tuning a pre-trained large language model on the MultiNERD dataset for the NER task. The goal of this implementation is to detect the effect of removing (masking) some of the named entity classes from the dataset (setting them to non-named entities) on the performance of the model on the other entities. The selected classes to be kept in both experiments are:
1. PER
2. ORG
3. LOC
4. DIS
5. ANIM


## run instructions:
Before running the codes, install requirements from the requirements.txt file.
* Training
    1. set training arguments in the arguments.py file. You can choose a pre-trained checkpoint from the HuggingFace hub there. 
    2. run train.py 

* inference
    1. set the inference arguments in the arguments.py file. 
    2. run inference.py

## evaluation metrics:
Analyzing the dataset shows that the classes are significantly imbalanced and the accuracy metric wouldn't be insightful. So, the combination of the metrics that are chosen to evaluate the models are:
1. accuracy
2. precision
3. F1 score
4. recall
These metrics are evaluated on each of the selected categories.
![alt text](readmefiles/default-dataset.png)
![alt text](readmefiles/masked-dataset.png)

## results:
Due to limited computational resources, 2 lightweight models are chosen to be fine-tuned. Although lightweight, their results are comparable to the larger models like BERT. The trained checkpoints are in [this](https://huggingface.co/pariakashani) HuggingFace repository. The performance of the fine-tuned models is measured on the test dataset. (click on each model to navigate to the hub's readme for checking the results)
1. DistilBERT
    * default dataset (experiment A):
        - [without augmenting](https://huggingface.co/pariakashani/en-multinerd-ner-more-training):
        - [with up-sampling](https://huggingface.co/pariakashani/en-multinerd-ner-unmasked-upsampled):
    * masked dataset (experiment B):
        - [without augmenting](https://huggingface.co/pariakashani/en-multinerd-masked-ner-more-training):
        - [with up-sampling](https://huggingface.co/pariakashani/en-multinerd-ner-upsampled):
2. roBERTa
    * [default dataset](https://huggingface.co/pariakashani/en-multinerd-ner-roberta):



## highlights:
The NER model had a harder time recognizing diseases ('DIS') and animals ('ANIM') because there weren't many examples of these in the dataset. I tried repeating sentences with these categories in the training data to help, but it didn't improve things.

I used a simpler model that did okay for most categories. Surprisingly, making the model bigger didn't change much. Switching between different fancy models, like RoBERTa and DistilBERT, also didn't make a big difference.

To make the model better, I think we need more examples from larger datasets. Adding more sentences about rare categories could help the model understand them better.

In conclusion, the imbalanced classes were tough to handle, fancier models didn't show much improvement, and getting more examples for rare categories might be the key to enhancing the model's performance.

## Acknowledge:
most of the codes in this repository are taken from HuggingFace [tutorial on Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification) and are modified for this task's dataset.





