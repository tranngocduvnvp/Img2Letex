from ast import arg
import enum
from lib2to3.pgen2 import token
from operator import eq
from pickletools import optimize
import cv2
import re
import numpy as np
from dataset import Im2LatexDataset
import torch
from torchtext.data import metrics
from torchmetrics.functional import char_error_rate
from model import OCR_model
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transform import test_transform
from transformers import PreTrainedTokenizerFast
import warnings
warnings.filterwarnings('ignore')

class ExperimentModel:
    def __init__(self, args) -> None:
        self.trainloader = Im2LatexDataset().load(args["data_train"])
        self.valloader = Im2LatexDataset().load(args["data_val"])
        self.testloader = Im2LatexDataset().load(args["data_test"])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = OCR_model(args["model"][0], args["model"][1]).to(self.device)
        if args["checkpoint"] != None:
            checkpoint = torch.load(args["checkpoint"], map_location=self.device)
            self.model.load_state_dict(checkpoint)
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = optim.SGD(self.model.parameters(),lr = 0.01, momentum=0.9, weight_decay=1e-4)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=args["tokenizer"])
        self.args = args
    
    def exp(self):
        for epoch in range(1, self.args["num_epoch"]):
            train_loss = self.train_step(iter(self.trainloader))
            eval_loss = self.eval_step(iter(self.valloader))
            print("Train loss {:.4f} | Eval loss {:.4f}".format(train_loss, eval_loss))
            if epoch%5 == 0:
                bleu_score, cer_loss = self.metric(iter(self.testloader))
                print("Bleu score {:.4f} | Cer loss {:.4f}".format(bleu_score, cer_loss))
            print("-"*20)
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(),f"./checkpoints/checkpoint_epoch{epoch}")

    def train_step(self, datatrain):
        self.model.train()
        total_loss = 0
        count = 0
        # datatrain = iter(self.trainloader)
        with tqdm(total=len(datatrain)) as pbar:
            for i, (tok, img) in enumerate(datatrain):
                bn = img.shape[0]
                img = img.to(self.device)
                tok = tok.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(img, tok["input_ids"][:,:-1])
                loss = self.criterion(logits, tok["input_ids"][:,1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()    
                total_loss +=loss.item()*bn
                count +=bn
                pbar.set_description("Loss train: {:.4f}".format(loss.item()))
                pbar.update(1)
        return total_loss/count

    def eval_step(self, dataval):
        self.model.eval()
        total_loss = 0
        count = 0
        # dataval = iter(self.valloader)
        with tqdm(total=len(dataval)) as pbar:
            for i, (tok, img) in enumerate(dataval):
                bn = img.shape[0]
                img = img.to(self.device)
                tok = tok.to(self.device)
                logits = self.model(img, tok["input_ids"][:,:-1])
                loss = self.criterion(logits, tok["input_ids"][:,1:])
                total_loss +=loss.item()*bn
                count +=bn
                pbar.set_description("Loss eval: {:.4f}".format(loss.item()))
                pbar.update(1)
        return total_loss/count
    
    def metric(self, datatest):
        self.model.eval()
        bleus_score = []
        cer_loss = []
        with tqdm(total=len(datatest)) as pbar:
            for i, (tok, img) in enumerate(datatest):
                bn = img.shape[0]
                img = img.to(self.device)
                tok = tok.to(self.device)
                pred = self.predict(img)
                #========================= Tính bleu score =======================================
                truth_detoc = self.detokenize(tok['input_ids'], self.tokenizer)
                pred_detoc = self.detokenize(pred, self.tokenizer)
                # print("pred:", truth_detoc)
                # print("trult:", pred_detoc)
                bleus = metrics.bleu_score(pred_detoc, [[x] for x in truth_detoc])
                # break
                #=============================== End ==============================================
                #============================ Tính CER metric =====================================
                truth_post_pro = self.token_post_process(tok["input_ids"], self.tokenizer)
                pred_post_pro = self.token_post_process(pred, self.tokenizer)
                cer = char_error_rate(pred_post_pro, truth_post_pro)
                #==================================================================================
                bleus_score.append(bleus)
                cer_loss.append(cer)
                pbar.set_description("Bleus score: {:.4f}| Cer loss: {:.4f}".format(bleus, cer))
                pbar.update(1)
        return np.mean(bleus_score), np.mean(cer_loss)
        
    def predict(self, img):
        self.model.eval()
        pred = self.model.predict(img)
        return pred
    def inference(self, img):
        self.model.eval()
        pred = self.model.predict(img)
        pred = self.token_post_process(pred, self.tokenizer)
        return pred

    def detokenize(self, tokens, tokenizer):
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                    del toks[b][i]
        return toks

    def token_post_process(self, tokens, tokenizer):
        """convert token ở dạng 

        Args:
            tokens (list): token ở dạng id
            tokenizer (_type_): _description_

        Returns:
            list: list các string predict đã được hậu xử lý
        """
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens] #convert token ids to token 
        toks = [self.token2str(tok) for tok in toks] #convert list token to string token
        toks = [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in toks]
        toks = [self.post_process(tok) for tok in toks]
        return toks
    
    def post_process(self, s: str):
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

    def token2str(self, token):
        """convert token o dang list sang str
        Args:
            token (list): [t1, t2, t3]
        """
        s = ""
        for tk in token:
            s +=tk
        return s


args = {
    "data_train":"D:\Img2Latex\Latex_ocr\datasets\\train.pkl",
    "data_val":"D:\LaTeX-OCR-1\Latex_ocr\hoa.pkl",
    "data_test":"D:\Img2Latex\datasets\\test.pkl",
    "tokenizer":"D:\LaTeX-OCR-1\Latex_ocr\\tokenizer\\tokenizer.json",
    "model":[[128, 4, 256, 3], [128, 4, 256, 3, 1175]],
    "checkpoint":"D:\Img2Latex\Latex_ocr\checkpoints\checkpoint_epoch20",
    "num_epoch":20
}

if __name__ == "__main__":
    run = ExperimentModel(args)
    run.exp()
    # image = cv2.imread("D:\Img2Latex\data_train\\train_5_0_da.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = test_transform(image=image)['image']
    # pred = run.inference(image[:1].unsqueeze(0))
    # print(pred)

