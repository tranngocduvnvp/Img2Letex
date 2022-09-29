import imghdr
import pandas as pd

def make_csv_final():
    private_test = pd.read_csv("D:\Img2Latex\datasets\private_test_norm\private_test.csv", sep="delimiter")["formula"]
    test_subsample = pd.read_csv("D:\Img2Latex\datasets\private_test_norm\\test_subsample.csv")["image"]

    images = []
    formulas = []
    for image, formula in zip(test_subsample,private_test):
        images.append(image)
        formulas.append(formula)

    save_frame = pd.DataFrame({"image":images, "formula":formulas})
    save_frame.to_csv("D:\Img2Latex\datasets\private_test_norm/private_final.csv")

def make_csv_for_convert_word():
    myfr = pd.read_csv("D:\Img2Latex\datasets\private_test_norm\private_test.csv", sep='delimiter')["formula"]
    formula = ["$ " + x + " $ \\\\" for x in myfr]
    saveFrame = pd.DataFrame({"formula":formula})
    saveFrame.to_csv("D:\Img2Latex\datasets\private_test_norm\\private_convert_word.csv", index=False)

def merge_csv():
    old_frame = pd.read_csv("D:\Img2Latex\datasets\scio_csv\data_scio_1.csv")
    for i in range(2,12):
        new_frame = pd.read_csv(f"D:\Img2Latex\datasets\scio_csv\data_scio_{i}.csv")
        old_frame = pd.concat([old_frame, new_frame], axis=0)
    old_frame.to_csv("D:\Img2Latex\datasets\scio_csv\data_scio_crawl.csv", index=False)
    print("----------------------- Done -------------------------------------------")


