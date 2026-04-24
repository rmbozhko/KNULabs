1) User inputs PDF page

↓

2) Script converts PDF to PNG

↓

3) Tesseract OCR produces tokens + bboxes

↓

4) LayoutLMv3
(assign label per token)

↓

5) Merge tokens into blocks

↓

6) Convert blocks to Excalidraw JSON

↓

7) Render board


Мій проєкт - це конвертація сторінки підручника формату PDF з англійської мови у борд Excalidraw, який можна редагувати.
1. На вхід я отримую сторінку у форматі PDF, яку завантажує користувач.
2. Окремий скрипт конвертує її у PNG.
3. Далі функція витягує з сторінки окремі токени та їх bounding boxes за допомогою Tesseract OCR.
4. Далі є декілька варіантів розвитку:
    4.1 Дати дані моделі класифікації (INSTRUCTION, CONTENT) у форматі токенів
        4.1.1 Токени дати на класифікацію натренованій моделі класифікації (INSTRUCTION, CONTENT) а потім об'єднати в блоки за ідентичним лейблами, які знаходяться поруч.
    4.2 Дати дані моделі класифікації (INSTRUCTION, CONTENT) у форматі блоків
        4.2.1 Токени потрібно об'єднати в блоки, за близкістю один до одного, і дати на класифікацію натренованій моделі класифікації (INSTRUCTION, CONTENT).
        4.2.2 Токени потрібно об'єднати в блоки за допомогою Tesseract OCR і дати на класифікацію натренованій моделі класифікації (INSTRUCTION, CONTENT).
        4.2.3 Токени потрібно об'єднати в блоки за допомогою окремою моделі і дати на класифікацію натренованій моделі класифікації (INSTRUCTION, CONTENT). 

5. Класифіковані блоки я конвертую у формат:
[{
    'block-instruction': ['block-content', ...]  -- якщо за блоком інструкції відразу йдуть блоки контенту
    'block-instruction': [] -- якщо за блоком інструкції йде ще один блок інструкції
}]
6. Отримані дані я передаю до функції, яка відповідає за генерацію борда Excalidraw (https://docs.excalidraw.com/docs/@excalidraw/excalidraw/api/props). Блоки інструкції мають #afd0d6 background color, а блоки контенту мають #BFB6BB background color.

If you implement token classification with a standard model (e.g., a fine-tuned BERT or a Random Forest), you are feeding it:Tokens (text): "Question", "1", "."Raw Coordinates: $(x, y, w, h)$The model tries to decide if "Question" is an INSTRUCTION or CONTENT based primarily on the word itself and its location. It has no "visual sense." It cannot see that a block is indented, bolded, or sits in a specific column relative to an image. It relies entirely on your manual feature engineering to tell it that "indentation implies instruction." It will fail when your textbook layout varies.The LayoutLMv3 Difference
(Multimodal Fusion) LayoutLMv3 doesn't just "see" the text and coordinates. It processes three modalities simultaneously through its attention mechanism:
Textual: The embedding of the word "Question."
Spatial: The 2D positional embedding (where it is on the page).
Visual: The actual pixel patches of the document image.

LayoutLMv3 is a classification model that takes into account not only the word aka token itself but also spatial and visual components



# TODO
Check the dataset distribution of INSTRUCTION vs CONTENT and adjust the weights accordingly in the loss function to handle class imbalance.
You do not modify the model architecture for this; you modify the Loss Function in your training loop.

Since you are likely using the Hugging Face Trainer API or a standard PyTorch loop, you apply the weights in the initialization of your CrossEntropyLoss.

Where: In your custom Trainer subclass (if overriding compute_loss) or where you define your criterion.

The Math: If "Content" is 10x more frequent than "Instruction," your weights should be roughly [1.0, 10.0] (assuming label 0 is Content, 1 is Instruction).

Implementation snippet:

Python
import torch
import torch.nn as nn

# Define weights based on your dataset distribution
# Weights: [weight_for_class_0, weight_for_class_1]
weights = torch.tensor([1.0, 10.0]) 

# Pass this to your loss function
loss_fct = nn.CrossEntropyLoss(weight=weights)

Pro Tip: Don't just set it and forget it—monitor your F1-score for the "Instruction" class specifically. If Recall for "Instruction" is still low, increase the weight further.

# Phase 1: The "Bridge" (Data Conversion)
1. Make sure build_dataset.py works correctly:
- adjust build_dataset to work like build_block_dataset, e.g. train/val/test split
- find a way to determine factor for weighted loss as "Content" blocks usually outnumber "Instruction"
- image: path to the PNG image file. Should it be resized in the script or left as is for the model? Can model read image from Google Drive?
- Coordinate Alignment: Double-check that your bboxes in the training JSON are truly normalized between $0$ and $1000$. If they are pixel values (e.g., $0$ to $2481$), your model will not learn spatial relationships.
- Tag Consistency: Ensure every token has a label.
- Validation Check: Take one page and check its token-label pairs manually to confirm the integrity of your dataset.

# Phase 2: The Fine-Tuning Loop
1. Make sure token training loop is set up correctly:
- Make sure that model can work with **images** and tokens.
- Config: Set your id2label and label2id mapping (e.g., {0: "INSTRUCTION", 1: "CONTENT", 2: "OTHER"}).
- Weighted Loss: Apply that CrossEntropyLoss with weights we discussed earlier to handle your CONTENT vs. INSTRUCTION imbalance.
```import json, torch, torch.nn as nn
  cw = json.load(open('dataset\Focus_3_-_Student_book_-_2020\hf_dataset/class_weights.json'))
  n  = 3  # num_labels
  w  = torch.tensor([cw['class_weights_by_id'][str(i)] for i in range(n)])
  loss_fn = nn.CrossEntropyLoss(weight=w)```

## Testing:
1. The model will predict labels for every token. Write a script that iterates through the predictions and groups tokens with the same label if their bounding boxes are adjacent (the "Proximity Threshold" logic). -> derive block-level predictions from token-level predictions. Should I have a separate script for that or do it in the baseline notebook?
2. Write a function to debug the output of your model by visualizing the predicted labels on the original image. This will help you see if the model is correctly identifying instructions vs. content.

# Phase 3: The Inference & Integration Script
1. Convert separate scripts into a unified pipeline:

```single page PDF -> OCR -> Model Inference -> Label Aggregation -> Excalidraw JSON```
   - Functions should be called instead of separate scripts.
2. Prepare module to convert predicted blocks into Excalidraw JSON format, example is provided in `sample_json.excalidraw`.
3. Test the entire pipeline and import resulting Excalidraw json to Excalidraw platform to verify the correct integration.