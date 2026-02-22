from sklearn.metrics import classification_report
import re

def extract_tokens_and_labels(text):
    pattern = r"(\w+)([.,!?])?"
    matches = re.findall(pattern, text)
    
    tokens = [m[0].lower() for m in matches]
    labels = [m[1] if m[1] else 'O' for m in matches]
    
    return tokens, labels

reference_text = "покликавши мойсея, сказав господь до нього з намету зборів таке, промов до синів ізраїля та скажи їм, коли хтось і споміж вас приноситиме господеві жертво принос, то з товару або зотар, майте приносити вашу жертву. коли його жертва на все палення з товару, то нехай принесе самця безвади, коло входув на мед зборів принесе він його, щоб знайти ласку вгоспода. він покладе свою руку на голову жертві всепалення, й вона буде прийнята йому на користь, як покута за нього. і заріже бичка перед господом, а синиаронові, священики, принесуть його кров і покроплять нею з усіх боків жертовник, що привходів на мед зборів."
hypothesis_text = "Покликавши Мойсея, сказав Господь до нього з намету зборів, таке промов до синів Ізраїля та скажи їм Коли хтось і споміж вас, приноситиме Господеві жертво- принос, то з товару або зотар майте приносити вашу жертву. Коли його жертва на все палення з товару, то нехай принесе самця безвади коло входув на мед зборів. Принесе він його, щоб знайти ласку вгоспода, він покладе свою руку на голову жертві всепалення, й вона буде прийнята йому на користь як покута за нього і заріже бичка перед Господом, а синиаронові священики принесуть його кров і покроплять нею з усіх боків Жертовник, що привходів на мед зборів"

ref_tokens, ref_labels = extract_tokens_and_labels(reference_text)
hyp_tokens, hyp_labels = extract_tokens_and_labels(hypothesis_text)

if len(ref_labels) != len(hyp_labels):
    min_len = min(len(ref_labels), len(hyp_labels))
    ref_labels, hyp_labels = ref_labels[:min_len], hyp_labels[:min_len]

punctuation_marks = [',', '.', '!', '?']
existing_marks = list(set(ref_labels) & set(punctuation_marks))

print(classification_report(ref_labels, hyp_labels, labels=existing_marks, zero_division=0, digits=4))