from peft import get_peft_model, LoraConfig, TaskType
from transformers import MarianMTModel, MarianTokenizer, TrainingArguments, Trainer


class LoRATrainer:
    def __init__(self, model_name='feVeRin/enko-transliteration'):
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.peft_model = None

    def set_lora(self, r=16, alpha=32, dropout=0.1):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2', 'lm_head'],
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def data_split(self, data_path, test_size=0.2):
        import pandas as pd

        from data.textdataset import TextDataset
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(data_path, sep='\t')
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=123)

        train_dataset = TextDataset(train_df, self.tokenizer)
        val_dataset = TextDataset(val_df, self.tokenizer)

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, output_dir='./LoRA'):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5000,
            eval_strategy='steps',
            eval_steps=5000,
            save_strategy='steps',
            save_steps=5000,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            learning_rate=3e-4,
            gradient_accumulation_steps=2,
            fp16=True,
            report_to='wandb',
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            run_name='Transliteration'
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
