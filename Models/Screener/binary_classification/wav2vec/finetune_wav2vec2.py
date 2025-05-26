from datasets import load_dataset, load_from_disk, Audio, Dataset, concatenate_datasets
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments
import torch
import argparse
import os

def balance_dataset(dataset):
    # Separate the dataset into two classes based on the label
    class_0 = dataset.filter(lambda example: example["label"] == 0)
    class_1 = dataset.filter(lambda example: example["label"] == 1)

    # Get the sizes of both classes
    class_0_size = len(class_0)
    class_1_size = len(class_1)

    # If class_0 is larger, we might undersample it to match the size of class_1
    if class_0_size > class_1_size:
        # Undersample class_0 (majority class) to match the size of class_1
        undersampled_class_0 = class_0.shuffle(seed=42).select([i for i in range(class_1_size)])
        balanced_dataset = concatenate_datasets([undersampled_class_0, class_1])
    elif class_1_size > class_0_size:
        # Undersample class_1 (majority class) to match the size of class_0
        undersampled_class_1 = class_1.shuffle(seed=42).select([i for i in range(class_0_size)])
        balanced_dataset = concatenate_datasets([class_0, undersampled_class_1])
    else:
        # If both classes are already balanced, no changes needed
        balanced_dataset = dataset

    return balanced_dataset


class MyTrainer(Trainer):
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        batch = next(iter(dataloader))
        print("First batch shape:", batch['input_values'].shape)
        return dataloader

def main(args):

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True, padding=True 
        )
        # Ensure labels are cast to int64
        examples["label"] = torch.tensor(examples["label"], dtype=torch.int64)
        return inputs

    # Load dataset
    data = load_from_disk(args.data_id)
    
    # Encode labels
    data = data.class_encode_column("label")

    # Resample audio (wav2vec2 needs sampling_rate of 16000 Hz)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    # Preprocess data (create batched inputs + change name of label column)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.pretrained_model_id)
    encoded_data = data.map(preprocess_function, remove_columns="audio", batched=True)

    # Ensure that the label column is of type int64 (Long)
    encoded_data = encoded_data.map(lambda example: {"label": torch.tensor(example["label"], dtype=torch.int64)}, batched=True)

    # Split the dataset using stratified sampling (train_test_split on labels)
    split = encoded_data.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

    split['train'] = balance_dataset(split['train'])
    split['test'] = balance_dataset(split['test'])

    # Verify the distribution in the splits
    print(f"Train set distribution: {split['train']['label'].count(0)}")
    print(f"Train set distribution: {split['train']['label'].count(1)}")
    print(f"Test set distribution: {split['test']['label'].count(0)}")
    print(f"Test set distribution: {split['test']['label'].count(1)}")

    # Ensure labels are cast to int64 in both train and test splits
    split['train'] = split['train'].map(lambda example: {"label": torch.tensor(example["label"], dtype=torch.int64)}, batched=True)
    split['test'] = split['test'].map(lambda example: {"label": torch.tensor(example["label"], dtype=torch.int64)}, batched=True)

    # Verify the label type (it should be int64/Long)
    print(f"Label type (train): {split['train'].features['label'].dtype}")
    print(f"Label type (test): {split['test'].features['label'].dtype}")

    # Load model
    id2label = {0: 'DLD', 1: 'TD'}
    label2id = {'DLD': 0, 'TD': 1}
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        args.pretrained_model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
    )

    # Define trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
    )

    # Train model
    trainer.train()

    # Check if the model output directory exists, if not, create it
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    # Once ready, save to disk
    trainer.save_model(args.model_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, default="finetuning_data/4yo_childes_auris_denoised")
    parser.add_argument("--pretrained_model_id", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--model_output_dir", type=str, default="finetuned_model/4yo_childes_auris_denoised")
    args = parser.parse_args()
    main(args)