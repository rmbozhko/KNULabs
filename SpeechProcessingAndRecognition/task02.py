import re
import difflib
import unicodedata
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

INSERTION_PLACEHOLDER = '""'
CORRECT_MATCH_SYMBOL = '*'
SHEET_CONFIGS = [
    {
        "metric_key": "wer",
        "sheet_title": "WER analysis",
        "col_a_width": 25,
        "col_b_width": 80,
    },
    {
        "metric_key": "cer",
        "sheet_title": "CER analysis",
        "col_a_width": 15,
        "col_b_width": 50,
    },
]


def normalize_apostrophes(text: str) -> str:
    return re.sub(r"['''`]", "ʼ", text)


def replace_punctuation_with_spaces(text: str) -> str:
    return ''.join(
        ' ' if unicodedata.category(ch).startswith('P') else ch
        for ch in text
    )


def collapse_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def preprocess(text: str) -> str:
    if not text:
        return ''
    text = text.lower()
    text = normalize_apostrophes(text)
    text = replace_punctuation_with_spaces(text)
    text = collapse_whitespace(text)
    return text


def tokenize_words(text: str) -> list[str]:
    return word_tokenize(text) if text else []


def tokenize_chars(text: str) -> list[str]:
    return list(text.replace(' ', '_')) if text else []


def compute_alignment(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple:
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    substitutions = insertions = deletions = 0
    alignment_pairs = []

    for operation, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if operation == 'equal':
            for offset in range(ref_end - ref_start):
                alignment_pairs.append(
                    (ref_tokens[ref_start + offset], CORRECT_MATCH_SYMBOL)
                )

        elif operation == 'delete':
            for offset in range(ref_end - ref_start):
                alignment_pairs.append((ref_tokens[ref_start + offset], '-'))
                deletions += 1

        elif operation == 'insert':
            for offset in range(hyp_end - hyp_start):
                alignment_pairs.append(
                    (INSERTION_PLACEHOLDER, hyp_tokens[hyp_start + offset])
                )
                insertions += 1

        elif operation == 'replace':
            ref_chunk = ref_tokens[ref_start:ref_end]
            hyp_chunk = hyp_tokens[hyp_start:hyp_end]
            max_chunk_len = max(len(ref_chunk), len(hyp_chunk))

            for offset in range(max_chunk_len):
                ref_word = ref_chunk[offset] if offset < len(ref_chunk) else INSERTION_PLACEHOLDER
                hyp_word = hyp_chunk[offset] if offset < len(hyp_chunk) else INSERTION_PLACEHOLDER

                if ref_word == INSERTION_PLACEHOLDER:
                    insertions += 1
                elif hyp_word == INSERTION_PLACEHOLDER:
                    deletions += 1
                else:
                    substitutions += 1

                alignment_pairs.append((ref_word, hyp_word))

    ref_length = len(ref_tokens)
    return substitutions, insertions, deletions, alignment_pairs, ref_length


def compute_error_rate(substitutions: int, insertions: int, deletions: int, ref_length: int) -> float:
    if ref_length == 0:
        return 0.0
    return (substitutions + insertions + deletions) / ref_length


def evaluate_wer(ref: str, hyp: str) -> dict:
    ref_words = tokenize_words(ref)
    hyp_words = tokenize_words(hyp)
    s, i, d, alignment, n = compute_alignment(ref_words, hyp_words)
    return {
        "wer": compute_error_rate(s, i, d, n),
        "s": s, "i": i, "d": d,
        "alignment": alignment
    }


def evaluate_cer(ref: str, hyp: str) -> dict:
    ref_chars = tokenize_chars(ref)
    hyp_chars = tokenize_chars(hyp)
    s, i, d, alignment, n = compute_alignment(ref_chars, hyp_chars)
    return {
        "cer": compute_error_rate(s, i, d, n),
        "s": s, "i": i, "d": d,
        "alignment": alignment
    }


def build_hypothesis_column_label(metric_key: str, score: float, s: int, i: int, d: int) -> str:
    return f"STT Hypothesis ({metric_key.upper()}={score:.2%}, S={s}, I={i}, D={d})"


def write_alignment_sheet(writer, sheet_title: str, alignment_pairs: list, hypothesis_col_label: str,
                           col_a_width: int, col_b_width: int) -> None:
    rows = [[ref, hyp] for ref, hyp in alignment_pairs]
    df = pd.DataFrame(rows, columns=["Reference", hypothesis_col_label])
    df.to_excel(writer, index=False, sheet_name=sheet_title)

    worksheet = writer.sheets[sheet_title]
    worksheet.column_dimensions['A'].width = col_a_width
    worksheet.column_dimensions['B'].width = col_b_width


def export_to_excel(ref: str, hyp: str, output_path: str) -> None:
    evaluators = {
        "wer": evaluate_wer(ref, hyp),
        "cer": evaluate_cer(ref, hyp),
    }

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for config in SHEET_CONFIGS:
            metric_key = config["metric_key"]
            results = evaluators[metric_key]
            score = results[metric_key]
            s, i, d = results["s"], results["i"], results["d"]

            hypothesis_label = build_hypothesis_column_label(metric_key, score, s, i, d)
            write_alignment_sheet(
                writer=writer,
                sheet_title=config["sheet_title"],
                alignment_pairs=results["alignment"],
                hypothesis_col_label=hypothesis_label,
                col_a_width=config["col_a_width"],
                col_b_width=config["col_b_width"],
            )


if __name__ == '__main__':
    reference_text = """
    Покликавши Мойсея, сказав Господь до нього з намету зборів таке: «Промов до синів Ізраїля та скажи їм: Коли хтось із-поміж вас приноситиме Господеві жертвопринос, то з товару або з отар, маєте приносити вашу жертву. Коли його жертва на всепалення - з товару, то нехай принесе самця без вади; коло входу в намет зборів принесе він його, щоб знайти ласку в Господа. Він покладе свою руку на голову жертви всепалення, й вона буде прийнята йому на користь - як покута за нього. І заріже бичка перед Господом, а сини Аронові, священики, принесуть його кров і покроплять нею з усіх боків жертовник, що при вході в намет зборів.
    """

    wav2vec2_hypothesis = """
    покликавши мойсея господь сказав до нього з намету зборів такепромов до синів ізраїля та скажи їмколи хтось із поміж вас приноситиме господові жертво принос то з худоби великої або з отар маєте приносити вашу жертвуколи його жертва на всепалення з худоби великої то нехай принесе самця безвади коло входу в намед зборів принесе він його щоб знайти ласку в господавін покладе свою руку на голову жертви всепалення і вона буде прийнята йому на користь як покута за ньогоі заріже бичка перед господом а сини аронові священники принесуть його кров і покроплять нею з усіх боків жертовник що привході в намед збо
    """

    deep_speech_hypothesis = """
    покликавши мойсея господь сказав до нього з намету зборів таке промов до синів ізраїля та скажи їм коли хтось із поміж вас приноситиме господеві жертвувати вашу жертву його жертва на спалення худоби великої невикористовувані останнього і кроплять нею з усіх боків повноприводна
    """

    preprocessed_ref = preprocess(reference_text)
    preprocessed_wav2vec2 = preprocess(wav2vec2_hypothesis)
    preprocessed_deep_speech = preprocess(deep_speech_hypothesis)

    export_to_excel(preprocessed_ref, preprocessed_wav2vec2, output_path='wav2vec2.xlsx')
    export_to_excel(preprocessed_ref, preprocessed_deep_speech, output_path='deep_speech_with_lm.xlsx')