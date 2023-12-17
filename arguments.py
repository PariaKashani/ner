MASKED_TRAINING = False
SELECTED_CATEGORIES = ['PER', 'ORG', 'LOC', 'DIS', 'ANIM']
# pretrained model to be use in training, options: 'distilbert-base-uncased', 'roberta-base'
MODEL_CKPT = 'distilbert-base-uncased'
OUTPUT_DIR = 'multinerd-ner'

# ------- inference arguments ------
# model used in inference, options:
# list of options: (add repository name : pariakashani to the begining of the options) 
# A models(DistilBERT): en-multinerd-ner, en-multinerd-ner-more-training, en-multinerd-ner-unmasked-upsampled
# B models(DistilBERT): en-multinerd-masked-ner, en-multinerd-masked-ner-more-training, en-multinerd-ner-upsampled (this model is masked)
# A models(roBERTa): en-multinerd-ner-roberta
MODEL_NAME = 'pariakashani/en-multinerd-ner'

# True: evaluates the model on the test set, 
# False: evaluates the model on the TEST_TEXT
EVALUATE_ON_TEST_SET = True
TEXT = "Dogs can get infected by rabies, a viral disease that affects the nervous system and can be fatal."
