from docx import Document
from lxml import etree
import latex2mathml.converter
from tqdm import tqdm
import sys
import pandas as pd

def latex_to_word(latex_input):
    # mathml = latex2mathml.converter.convert(latex_input) \
    #         .replace('<mover>', '<mover accent="true" accentunder="false">')
    mathml = latex2mathml.converter.convert(latex_input)

    # # overline
    # mathml = latex2mathml.converter.convert(latex_input) \
    #         .replace('<mover>', '<mover accent="true" accentunder="false">').replace('<mo accent="true">&#x02015;</mo>', '<mo accent="false">&#x02015;</mo> ')
    
    # print(mathml)
 
    # print(mathml)
    tree = etree.fromstring(mathml)
    xslt = etree.parse(
        'D:\Img2Latex\datasets\MML2OMML.XSL'
        )
    transform = etree.XSLT(xslt)
    new_dom = transform(tree)
    return new_dom.getroot()


document = Document()

# with open(f'{sys.argv[1]}.txt', 'r') as f:
#     eqs = f.read().splitlines()

eqs = pd.read_csv("D:\Img2Latex\datasets\private_test_norm\private_test.csv", sep="delimiter")["formula"]

for x in tqdm(eqs):
    try:
        word_math = latex_to_word(x)
        p = document.add_paragraph()
        p._element.append(word_math)
    except:
        pass
document.save(f'file1.docx')