"""
Acknowledge: most of the codes are taken and from Token Classification tutorial in the HuggingFace website 
and 'Natural Language Processing with Transformers' book and are modified for this task's dataset. 
"""
from datasets import load_dataset, Dataset
from transformers import ( DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                         TrainingArguments, Trainer, AutoTokenizer)
import evaluate
import numpy as np

from arguments import MASKED_TRAINING, SELECTED_CATEGORIES, MODEL_CKPT, OUTPUT_DIR



label_str2ind_dict = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
  }
label_ind2str_dict = {v: k for k,v in label_str2ind_dict.items()}

if MASKED_TRAINING:
    label2id = {l:i for l,i in label_str2ind_dict.items() if l != 'O' and l.split('-')[1] in SELECTED_CATEGORIES}
    label2id = {'O':0} | label2id
    label2id = {label: i for i, label in enumerate(label2id)}
else:
    label2id = label_str2ind_dict

id2label = {i:l for l,i in label2id.items()}
LABEL2ID, ID2LABEL = label2id, id2label

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")


def mask_cats_in_dataset(example):
  """
  example: an entry or a batch of dataset entries

  keeps the selected categories in the dataset labels and sets other labels to 'O' or 0
  
  """
  new_tags = []
  for tag in example['ner_tags']:
    if tag == 0 or label_ind2str_dict[tag].split('-')[1] in SELECTED_CATEGORIES:
      new_tag = LABEL2ID[label_ind2str_dict[tag]]
    else:
      new_tag = 0
    new_tags.append(new_tag)
  example['ner_tags'] = new_tags
  return example


def load_ner_dataset(masked_training: bool=False)-> Dataset:
    """
    Downloads the english part of MultiNERD dataset files and returns a dataset objecgt
    """
    data_files = {'train': 'https://huggingface.co/datasets/Babelscape/multinerd/resolve/main/train/train_en.jsonl', 
                  'test': 'https://huggingface.co/datasets/Babelscape/multinerd/resolve/main/test/test_en.jsonl', 
                  'val': 'https://huggingface.co/datasets/Babelscape/multinerd/resolve/main/val/val_en.jsonl'}
    multinerd_dataset = load_dataset('json', data_files=data_files)
    multinerd_dataset.reset_format()
    if masked_training:
        multinerd_dataset = multinerd_dataset.map(mask_cats_in_dataset)
    
    # remove lang column
    try:
        multinerd_dataset = multinerd_dataset.remove_columns('lang')
    except:
        pass
    
    return multinerd_dataset


def upsample_rare_classes(example):
  """
  repeats examples from rare classes 2 times in the dataset to decrease the effect of imbalance data for those classes
  """
  print(example)
  anim_tag , dis_tag = label2id['B-ANIM'], label2id['B-DIS']
  new_tokens = []
  new_labels = []
  new_tokens.extend(example['tokens'])
  new_labels.extend(example['ner_tags'])
  for i, tags in enumerate(example['ner_tags']):
    if anim_tag in tags or dis_tag in tags:
        new_tokens.append(example['tokens'][i])
        new_labels.append(example['ner_tags'][i])
  example['tokens'] = new_tokens
  example['ner_tags'] = new_labels
  return example


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100 (ignore index of cross entropy loss in pytorch)
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)
    out = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    for sc in SELECTED_CATEGORIES:
      try:
        out[sc+'-precision'] = results[sc]['precision']
        out[sc+'-recall'] = results[sc]['recall']
        out[sc+'-f1'] = results[sc]['f1']
      except:
        try:
          out[sc+'-precision'] = results['B-'+sc]['precision']
          out[sc+'-recall'] = results['B-'+sc]['recall']
          out[sc+'-f1'] = results['B-'+sc]['f1']
        except:
          pass

    return out


def main():
  dataset = load_ner_dataset(MASKED_TRAINING)
  tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
  num_labels = len(LABEL2ID)
  model = AutoModelForTokenClassification.from_pretrained(MODEL_CKPT, 
                                                          num_labels=num_labels, 
                                                          id2label=ID2LABEL, 
                                                          label2id=LABEL2ID)
  training_args = TrainingArguments(
     output_dir=OUTPUT_DIR,
     learning_rate=2e-5,
     per_device_train_batch_size=16,
     per_device_eval_batch_size=16,
     num_train_epochs=4,
     weight_decay=0.01,
     evaluation_strategy="epoch",
     save_strategy="epoch",
     load_best_model_at_end=True,
     push_to_hub=False
    )
  
  trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_dataset["train"],
     eval_dataset=tokenized_dataset["val"],
     tokenizer=tokenizer,
     data_collator=data_collator,
     compute_metrics=compute_metrics
     )
  
  trainer.train()

  trainer.evaluate(tokenized_dataset["test"])


if __name__=='__main__':
   main()