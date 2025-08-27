from peft import get_peft_model, LoraConfig, TaskType
from transformers import MarianMTModel, MarianTokenizer, TrainingArguments, Trainer


class LoRATrainer:
    '''
    LoRA Trainer class for fine-tuning MarianMTModel
    
    Args:
        model_path: huggingface model path
    '''

    def __init__(self, model_path='feVeRin/enko-transliteration'):
        self.model_path = model_path
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.base_model = MarianMTModel.from_pretrained(model_path)
        self.peft_model = None

    def set_lora(self, r=16, alpha=32, dropout=0.1):
        '''
        Set up LoRA configuration and apply it to the base model
        
        Args:
            r: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout rate
        '''

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2', 'lm_head'],
        )

        self.peft_model = get_peft_model(self.base_model, lora_config)
        self.peft_model.print_trainable_parameters()

    def data_split(self, data_path, test_size=0.2):
        '''
        Load transliteration dataset and split into train/validation sets
        
        Args:
            data_path: dataset file path
            test_size: split ratio
        
        Returns:
            train_dataset: training dataset
            val_dataset: validation dataset
        '''

        import pandas as pd

        from data.textdataset import TextDataset
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(data_path, sep='\t')
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=123)

        train_dataset = TextDataset(train_df, self.tokenizer)
        val_dataset = TextDataset(val_df, self.tokenizer)

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, report_to='wandb', output_dir='./LoRA'):
        '''
        Fine-tune the base MarianMT model with LoRA
        
        Args:
            train_dataset: training dataset
            val_dataset: validation dataset
            report_to: wandb logging (if you don't need, set it to 'None')
            output_dir: saving directory
        '''

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
            report_to=report_to,
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
