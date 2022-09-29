import pandas as pd
from transformers import PreTrainedTokenizerFast

test_frame = pd.read_csv("D:\Img2Latex\datasets\private_test_norm\private_test.csv", sep='delimiter')
tokenizer = PreTrainedTokenizerFast(tokenizer_file="D:\Img2Latex\Latex_ocr\\tokenizer\\tokenizer.json")
images = []
equations = []

def make_new_token(token):
    sen = ""
    # B.\HOOC-(CH_{2})_{4}-COOH
    '''
    B .\ 
    '''
    for i, k in enumerate(token):
        if i+1 == len(token):
            sen += (k+" ") if (token[i])!=".\\" else k
        else:
            try:
                sen+=(k+" ") if (token[i] + token[i+1])!=".\\" else k
            except:
                print(i)
    return sen[:-1]

for  equation in test_frame["formula"]:
    tok = tokenizer([equation.replace(" ","")], return_token_type_ids=False)
    dec = tokenizer.convert_ids_to_tokens(tok["input_ids"][0])
    new_eq = make_new_token(dec)
    equations.append(new_eq)

new_frame = pd.DataFrame({"formula":equations})
new_frame.to_csv("D:\Img2Latex\datasets\private_test_norm\private_test_tem.csv", index=False)

# print(test_frame.head())