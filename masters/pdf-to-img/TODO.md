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

### Dataset Preparation:
- Review the weights calculation.

### Training
- I need to add some metrics and graphs for training stage (CPU/GPU usage, loss curves, F1-score for each class) to monitor the training process better.
- Learning Rate Schedulers: Use a Linear Warmup scheduler. This prevents the model from "jolting" its weights at the start of training and preserves the pre-trained visual knowledge. -> https://gemini.google.com/app/0762b47769ee40ed

## Testing:
- Plot, charts, confusion matrix, as many as possible

### Post-processing
1. Majority Voting (The "Block-Level" Fix)
- Group your tokens into physical lines or paragraphs first (using your OCR coordinates).
- For each group, take the mode (most frequent label) of the tokens within that group.
- If a block is 80% "INSTRUCTION" and 20% "CONTENT", force the entire block to "INSTRUCTION."
2. (Alternative to Majority Voting) Viterbi Decoding / CRF (Conditional Random Fields)
- Standard token classification models (like your current one) assume every token is independent. That's why you get [INSTRUCTION, CONTENT, INSTRUCTION].
A CRF layer on top of your model forces it to learn transition probabilities. It learns that an INSTRUCTION is highly likely to be followed by another INSTRUCTION, and highly unlikely to be followed by a CONTENT block without a transition.
- Action: If you aren't using a CRF head on your model, consider adding one. It is the industry standard for fixing "jittery" NER outputs.
3. Heuristic Filtering (The "Domain Knowledge" Layer)
- You are an English teacher. You know the textbook better than the model. Use your knowledge to enforce rules that the model is too "dumb" to learn:
- The "Keyword" Rule: If a block is predicted as "CONTENT" but contains the word "Listen," "Look," or "Exercise," the system must override it to "INSTRUCTION."
- The "Short Block" Rule: If a block is only 2-3 words long, it is statistically more likely to be a header or instruction than a paragraph of content.
4. Write a function to debug the output of your model by visualizing the predicted labels on the original image. This will help you see if the model is correctly identifying instructions vs. content.

# Phase 3: The Inference & Integration Script
1. Convert separate scripts into a unified pipeline:
```single page PDF -> OCR -> Model Inference -> Label Aggregation -> Excalidraw JSON```
   - Functions should be called instead of separate scripts.
2. Prepare module to convert predicted blocks into Excalidraw JSON format, example is provided in `sample_json.excalidraw`.
3. Test the entire pipeline and import resulting Excalidraw json to Excalidraw platform to verify the correct integration.

# Current TODO
✅ - Drop "OTHER" class and re-train model and run test with Majority Voting 
✅ - Learning Rate Schedulers: https://gemini.google.com/app/0762b47769ee40ed
✅ - Reduce Variance with Probability Threshold (baseline notebook history) + Majority Voting
- Viterbi Decoding / CRF
    - chat about lower f1-score than without CRF
- Merge tokens into blocks (check if blocks are organized correctly
    - maybe add some vertical and horizontal padding)
- Heuristic Filtering (The "Domain Knowledge" Layer)
    - it would be great to include ACTUAL label from baseline notebook to compare with model output and see how many blocks are corrected by heuristics
- Debugging function to visualize predicted labels on the original image
✅- Figure out how to convert PNG for model inference
✅- Figure out how to map model output to dict the Excalidraw module expects

- perform a "Salience Map" check. Use a tool like Captum to see which pixels the model is actually looking at when it predicts INSTRUCTION. If it's looking at random white space or the middle of a paragraph instead of the bold headers or icons in the Focus 3 book, your problem is 100% spatial normalization or data quality.
- make weights more aggressive: INSTRUCTION: [3.0, 4.0], CONTENT: [0.5, 1.0]
- increase size of dataset
- Data Augmentation: Since you only have 50 pages, you need to "hallucinate" more. Apply random rotations (1-2 degrees), slight scaling, and brightness shifts to your training images. This forces the model to rely on structural features rather than fixed pixel locations.
- in baseline comparison notebook tokens disappear in `eval_split` although in `ds` they are still present. Can LayoutLMv3's image processor be the culprit? Does it drop tokens that are outside of a certain area of the image? Check if tokens are still present after the image processor step and before being fed into the model.

✅- it makes sense to combine multiple CONTENT blocks into a single one if they are not separated by an INSTRUCTION block. The multiple INSTRUCTION blocks can be merged only if their proximity is less than a certain threshold. This will help to reduce the number of blocks and make the Excalidraw output cleaner.
✅- replace labels with tokens

✅- add boolean variable to work as a toggle to enable or disable usage of Gemini LLM for post-processing of model output. Instead of locating blocks with instructions and content in order as on image, we can feed the model output to Gemini and ask it to rearrange blocks in the correct linear order based on the content of the blocks (omitting blocks overlaying on each other) as well as complete the sentences with missing tokens. This will be especially useful for cases when the layout of the page is complex and blocks are not arranged in a linear way. I have been using the following LLM:
model = "gemini-3-flash-preview"

response = client.models.generate_content(
    model=model,
    config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko."),
    contents="Explain how AI works in a few words"
)
print(response.text)
I suppose we should change the system instruction to something like "You are an assistant that helps to rearrange blocks of text extracted from a textbook page. Your task is to take the output of a token classification model, which consists of blocks labeled as INSTRUCTION or CONTENT, and rearrange them in the correct linear order as they would appear on the page. You should also complete any sentences that are missing tokens based on the context of the blocks. The input will be a list of blocks with their labels and content, and you should output a rearranged list of blocks in the correct order." and also provide an example of the input and expected output in the system instruction to make it clearer for the model.



