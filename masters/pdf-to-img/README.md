1. PDF to PNG: python pdf_to_png.py ".\pdf\674_1- Focus 3. Student's Book_2020, 2nd, 159p.pdf" --output-dir dataset/Focus_3_-_Student_book_-_2020/images

2. PNG OCR: python png_ocr.py --images dataset/Focus_3_-_Student_book_-_2020/images --debug dataset/Focus_3_-_Student_book_-_2020/debug --output dataset/Focus_3_-_Student_book_-_2020/ocr --verify

3. OCR Postprocess: python ocr_postprocess.py --input dataset/Focus_3_-_Student_book_-_2020/ocr --output dataset/Focus_3_-_Student_book_-_2020/ocr_clean

4. Serve images: python serve.py

5. Go to label-studio. Enable virtual environment and start Label Studio with `label-studio`. Creds: ```admin@admin.com/qwerty12345```. Upload cleaned OCR output and start annotating.

6. Export annotations from Label Studio as JSON.

7. Map token to labels: `python .\post_labelling_mapping.py --input .\annotations\project-6-at-2026-04-15-15-19-70bfb200.json --output dataset/Focus_3_-_Student_book_-_2020/labeled`

8. Build HF dataset: `python build_dataset.py --input dataset/Focus_3_-_Student_book_-_2020/labeled --output dataset/Focus_3_-_Student_book_-_2020/hf_dataset`

9. Copy the `hf_dataset` folder to `/content/drive/MyDrive/layoutlmv3/dataset/hf_dataset`.

10. Make sure that dataset images are also uploaded to `/content/drive/MyDrive/layoutlmv3/dataset/Focus_3_-_Student_book_-_2020/images`. Make sure that the image paths in the dataset point to the correct location in Colab workspace. We copy images and HF dataset to Colab workspace and point to them from the dataset because LayoutLMv3's image processor expects local paths, not Google Drive paths.
