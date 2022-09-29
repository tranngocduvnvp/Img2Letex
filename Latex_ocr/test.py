from transformers import PreTrainedTokenizerFast
import re
import numpy as np
def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.
    Args:
        s (str): Input string
    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s  
def token2str(tokens, tokenizer) -> list:
    dec = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
    tem = ""
    for i in dec:
      tem+=i
    print("tem:",tem)
    s = ''.join(tem.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip()
    return s

tokenizer = PreTrainedTokenizerFast(tokenizer_file="D:\LaTeX-OCR-1\Latex_ocr\\tokenizer\\tokenizer.json")
str = "D.\   5,31.10^{- 23}"
token_ids = tokenizer(str)
print("str input:", str)
print("token_id:", token_ids["input_ids"])
token = tokenizer.convert_ids_to_tokens(token_ids["input_ids"])
print("Token:", token)
s = token2str(token_ids, tokenizer)
print("token2str:", s)
print("Post process:", post_process(s))


