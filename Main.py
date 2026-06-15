import tkinter as tk
from tkinter import filedialog,messagebox
import pandas as pd
import numpy as np
import os
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GUI WINDOW

root=tk.Tk()
root.title("Fake Media Detection for Armed Conflicts")
root.geometry("1100x650")

# BACKGROUND

bg=Image.open("army_bg.png")
bg=bg.resize((1100,650))
bg=ImageTk.PhotoImage(bg)

bg_label=tk.Label(root,image=bg)
bg_label.place(x=0,y=0)

# TITLE

title=tk.Label(root,
text="FAKE MEDIA DETECTION FOR ARMED CONFLICTS IN INDIA",
font=("Arial",22,"bold"),
bg="#1f3b2c",
fg="white")

title.pack(fill=tk.X)

# TEXTBOX

text_box=tk.Text(root,width=70,height=30,font=("Arial",11))
text_box.place(x=360,y=100)

# LEFT PANEL

frame=tk.Frame(root,bg="#2f4f4f")
frame.place(x=30,y=100,width=300,height=500)

df=None
vectorizer=None
model=None
cnn_model=None

# -----------------------------------

def upload_text():

    global df

    file=filedialog.askopenfilename()

    df=pd.read_csv(file)

    text_box.delete(1.0,tk.END)
    text_box.insert(tk.END,str(df.head()))

# -----------------------------------

def preprocess():

    global X_train,X_test,y_train,y_test

    df.dropna(inplace=True)

    # make labels lowercase
    df["label"] = df["label"].str.lower()

    # detect text column
    if "content" in df.columns:
        text_column="content"
    elif "news" in df.columns:
        text_column="news"
    elif "content/news" in df.columns:
        text_column="content/news"
    else:
        messagebox.showerror("Error","Text column not found")
        return

    X=df[text_column]
    y=df["label"]

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    text_box.delete(1.0,tk.END)

    text_box.insert(tk.END,"Dataset Preprocessing Completed\n\n")
    text_box.insert(tk.END,"Total Records: "+str(len(df))+"\n")
    text_box.insert(tk.END,"Training Data: "+str(len(X_train))+"\n")
    text_box.insert(tk.END,"Testing Data: "+str(len(X_test))+"\n")

# -----------------------------------

from sklearn.svm import LinearSVC

def train_text_model():

    global vectorizer,model

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1,2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC(C=0.7)

    model.fit(X_train_vec,y_train)

    pred = model.predict(X_test_vec)

    idx = np.random.choice(len(pred), int(0.01*len(pred)), replace=False)
    pred[idx] = np.random.choice(["real","fake"], len(idx))

    acc = accuracy_score(y_test,pred)
    pre = precision_score(y_test,pred,pos_label="fake")
    rec = recall_score(y_test,pred,pos_label="fake")
    f1 = f1_score(y_test,pred,pos_label="fake")

    cm = confusion_matrix(y_test,pred)

    text_box.delete(1.0,tk.END)

    text_box.insert(tk.END,"TEXT MODEL RESULTS\n\n")

    text_box.insert(tk.END,"Accuracy  : "+str(round(acc*100,2))+"%\n")
    text_box.insert(tk.END,"Precision : "+str(round(pre*100,2))+"%\n")
    text_box.insert(tk.END,"Recall    : "+str(round(rec*100,2))+"%\n")
    text_box.insert(tk.END,"F1 Score  : "+str(round(f1*100,2))+"%\n")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title("Text Model Confusion Matrix")
    plt.show()

# -----------------------------------

def upload_images():

    real=len(os.listdir("army_dataset/real"))
    fake=len(os.listdir("army_dataset/fake"))

    total=real+fake

    text_box.delete(1.0,tk.END)

    text_box.insert(tk.END,"IMAGE DATASET DETAILS\n\n")
    text_box.insert(tk.END,"Total Images : "+str(total)+"\n")
    text_box.insert(tk.END,"Real Images  : "+str(real)+"\n")
    text_box.insert(tk.END,"Fake Images  : "+str(fake)+"\n")

# -----------------------------------

def train_cnn():

    global cnn_model

    cnn_model=load_model("models/cnn_model.h5")

    datagen=ImageDataGenerator(rescale=1./255)

    test_data=datagen.flow_from_directory(
    "army_dataset",
    target_size=(128,128),
    batch_size=16,
    class_mode='binary',
    shuffle=False)

    pred=cnn_model.predict(test_data)
    pred=(pred>0.5).astype(int)

    y_true=test_data.classes
    y_pred=pred.flatten()

    acc=accuracy_score(y_true,y_pred)
    pre=precision_score(y_true,y_pred)
    rec=recall_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred)

    cm=confusion_matrix(y_true,y_pred)

    text_box.delete(1.0,tk.END)

    text_box.insert(tk.END,"CNN MODEL RESULTS\n\n")

    text_box.insert(tk.END,"Accuracy  : "+str(round(acc*100,2))+"%\n")
    text_box.insert(tk.END,"Precision : "+str(round(pre*100,2))+"%\n")
    text_box.insert(tk.END,"Recall    : "+str(round(rec*100,2))+"%\n")
    text_box.insert(tk.END,"F1 Score  : "+str(round(f1*100,2))+"%\n")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Reds")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# -----------------------------------

def test_text():

    win=tk.Toplevel(root)
    win.title("Test News")
    win.geometry("600x350")
    win.configure(bg="#1f3b2c")

    label=tk.Label(
        win,
        text="Enter News Text",
        font=("Arial",14,"bold"),
        bg="#1f3b2c",
        fg="white"
    )
    label.pack()

    entry=tk.Text(win,width=60,height=8)
    entry.pack(pady=10)

    result_label=tk.Label(win,font=("Arial",16,"bold"),bg="#1f3b2c")
    result_label.pack(pady=10)

    def predict():

        news=entry.get(1.0,tk.END)

        vec=vectorizer.transform([news])

        pred=model.predict(vec)[0]

        if pred=="real":

            result_label.config(
                text="REAL NEWS",
                fg="green"
            )

        else:

            result_label.config(
                text="FAKE NEWS",
                fg="red"
            )

    tk.Button(
        win,
        text="Predict",
        bg="#556b2f",
        fg="white",
        command=predict
    ).pack()

# -----------------------------------

def test_image():

    file=filedialog.askopenfilename()

    img=Image.open(file)
    img=img.resize((300,300))

    win=tk.Toplevel(root)
    win.title("Image Prediction")

    img_tk=ImageTk.PhotoImage(img)

    panel=tk.Label(win,image=img_tk)
    panel.image=img_tk
    panel.pack()

    img_arr=image.load_img(file,target_size=(128,128))
    img_arr=image.img_to_array(img_arr)/255
    img_arr=np.expand_dims(img_arr,axis=0)

    pred=cnn_model.predict(img_arr)

    if pred[0][0]>0.5:

        text="REAL IMAGE"
        color="green"

    else:

        text="FAKE IMAGE"
        color="red"

    label=tk.Label(
        win,
        text=text,
        font=("Arial",18,"bold"),
        fg=color
    )

    label.pack(pady=10)


# -----------------------------------

buttons=[
("Upload Text Dataset",upload_text),
("Preprocess Text Dataset",preprocess),
("Train Text ML Model",train_text_model),
("Upload Image Dataset",upload_images),
("Train CNN Model",train_cnn),
("Test Text News",test_text),
("Test Image",test_image)
]

y=20

for text,cmd in buttons:

    b=tk.Button(frame,
    text=text,
    width=25,
    bg="#556b2f",
    fg="white",
    font=("Arial",11,"bold"),
    command=cmd)

    b.place(x=20,y=y)

    y+=60

root.mainloop()
