from transformers import PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast(tokenizer_file="D:\LaTeX-OCR-1\Latex_ocr\\tokenizer\\tokenizer.json")
print(len(tokenizer))