1. PDF to PNG: python .\pdf_to_png.py '.\pdf\674_1- Focus 3. Student's Book_2020, 2nd, 159p.pdf'

2. PNG OCR: python png_ocr.py --images dataset/Focus_3_-_Student_book_-_2020/images --debug dataset/Focus_3_-_Student_book_-_2020/debug --output dataset/Focus_3_-_Student_book_-_2020/ocr --limit 5 --verify

3. OCR Postprocess: python ocr_postprocess.py --input dataset/Focus_3_-_Student_book_-_2020/ocr --output dataset/Focus_3_-_Student_book_-_2020/ocr_clean

4. Serve images: python serve.py

5. Go to label-studio. Enable virtual environment and start Label Studio with `label-studio`.
Creds: ```admin@admin.com/qwerty12345```

6. Export annotations from Label Studio as JSON.

7. Map token labels to blocks: `python .\post_labelling_mapping.py --input .\project-6-at-2026-03-31-15-30-3adfa9dd.json --output dataset/Focus_3_-_Student_book_-_2020/mapped`

8. Build HF dataset: `python build_dataset.py --input dataset/Focus_3_-_Student_book_-_2020/mapped --output dataset/Focus_3_-_Student_book_-_2020/hf_dataset`

