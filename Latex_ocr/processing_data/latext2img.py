from PIL import Image as IMG
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from IPython.lib.latextools import latex_to_png
import matplotlib
import random
import tqdm

myfr = pd.read_csv("D:\Img2Latex\datasets\scio_csv\data_scio_full_Copy.csv", sep='delimiter', engine='python')["formula"]
images = []
formula = []
cout =0
matplotlib.rcParams["mathtext.fontset"] = "cm"
def make_data(start, end):
  cout =0
  with tqdm.tqdm(total=(end-start)) as pbar:
    for i in range(start, end):
      s = myfr.iloc[i]
      stack = []
      sample = np.random.randint(1, 3)
      for k in range(sample):
        temp = s
        font = np.random.choice(["mathit", "mathrm", "da", "mathbf"], p=[0.3, 0.2, 0.3, 0.2])
        if font in stack:
          continue
        stack.append(font)
        if font == "mathit":
          s_k = r'\mathit{%s}'%s
        elif font == "mathrm":
          s_k = r'\mathrm{%s}'%s
        elif font == "mathbf":
          if ("Longrightarrow" in s) or ("star" in s):
            matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
          s_k = r'\mathbf{%s}'%s
        elif font == "da":
          a = np.random.randint(3)
          da = np.random.choice(["A","B","C","D"])
          temp = r"%s.\ %s"%(da, s)
          if a == 0:
            s_k = r'\mathrm{%s}'%temp
          elif a == 1:
            s_k = r'\mathit{%s}'%temp
          elif a == 2:
            if ("Longrightarrow" in s) or ("star" in s):
                matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
            s_k = r'\mathbf{%s}'%temp
        try:
          image = latex_to_png(s_k,backend="matplotlib", wrap=True)
          ig = IMG.open(io.BytesIO(image))
          ig_new = IMG.new('RGB', ig.size, color = 'white')
          ig_new.paste(ig,(0,0),ig)
          ig_new.save(f"D:\Img2Latex\Latex_ocr\data_train_final/train_{i}_{k}_{font}.png")
          images.append(f"train_{i}_{k}_{font}.png")
          formula.append(temp)
          matplotlib.rcParams["mathtext.fontset"] = "cm"
        except:
          cout +=1
      if i%20 == 0:
        save_frame = pd.DataFrame({"image":images,"formula":formula})
        save_frame.to_csv(f"./train_data_{start}.csv")
      pbar.update(1)



make_data(0,97000)