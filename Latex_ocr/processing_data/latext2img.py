import pandas as pd
import sympy

mf = pd.read_csv("D:\LaTeX-OCR-1\pix2tex\preprocess_crawl.csv", sep='delimiter', engine="python")
image =[]
formula = []
for i, equ in enumerate(mf["formula"]):
    try:
        sympy.preview(r'$$%s$$'% equ, viewer='file', filename=f'D:\LaTeX-OCR-1\datasets\data_ch\hoa{i}.png', euler=False)
        image.append(f"hoa{i}.png")
        formula.append(equ)
    except:
        pass
mf = pd.DataFrame({"image":image, "formula":formula})
mf.to_csv("D:\LaTeX-OCR-1\datasets\data_crawl.csv")