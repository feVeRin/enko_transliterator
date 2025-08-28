# English-to-Korean Transliterator | 영한 음역기

▶ README: [**ENGLISH**](README.md) | [KOREAN](docs/README.ko.md)

This project focuses on the English-to-Korean transliteration task (e.g. `english` -> `잉글리시`). Specifically, this project accelerates inference speed and reduces the large model size (~1.2GB) of the previous [MT5-based Transliterator](https://github.com/eunsour/engtokor-transliterator/tree/main).

- **Highlights**
    1. Providing lightweight model size (~400MB) with faster, accurate transliteration results.
        - Please check performance comparisons below.
    2. LoRA is applied to the MarianMT translation model.
        - The corresponding fine-tuned model is available in [HuggingFace](https://huggingface.co/feVeRin/enko-transliteration).

- **Performance Comparisons**
    ![image](./output.png)

## How to Start

1. Install dependencies (Assuming PyTorch is already installed):

    ```bash
    pip install -r requirements.txt
    ```

2. Clone Repository:

    ```bash
    git clone https://github.com/feVeRin/enko_transliterator.git
    ```

## How to Use

1. Transliteration (w/ pre-trained model)

    ```python
    from transliteration import Transliterator

    model = Transliterator.from_pretrained('feVeRin/enko-transliteration')
    result = model.transliterate('LORA IS ALL YOU NEED')
    print(result)  # 로라 이즈 올 유 니드
    ```

2. Model Training (from scratch)
    - Training is linked to `wandb`. If not necessary, add `report_to=None` to the `train()` function.

    ```python
    from train import LoRATrainer
    from data.textdataset import TextDataset

    trainer = LoRATrainer()
    trainer.set_lora(r=16, alpha=32, dropout=0.1)
    train_dataset, val_dataset = trainer.data_split('./data/data.txt', 0.2)
    trainer.train(train_dataset, val_dataset)
    ```

## References

- This project utilized a dataset of the [EngtoKor-Transliterator](https://github.com/eunsour/engtokor-transliterator/tree/main).
- [Opus-hplt-EN-KO](https://huggingface.co/Neurora/opus-hplt-en-ko-v2.0) was used as the base model for applying LoRA.
