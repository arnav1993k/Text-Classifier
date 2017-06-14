#! /usr/bin/python3
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import tkinter as tk
import subprocess
import os
import io
import numpy as np
from PIL import Image
import pickle
import json
f_size=50
canvas_size=100
b1 = "up"
xold, yold = None, None
master_data=[]
desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')

json_file=desktop_path+"/TF/"+"full_scale1.json"
print (json_file)
if os.path.isfile(json_file)==False:
    with open(json_file,'w') as fp:
        json.dump(master_data,fp)
else:
    with open(json_file) as fr:
        master_data=json.load(fr)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        global canvas_size
        self.line_start = None
        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonPress-1>", self.b1down)
        self.canvas.bind("<ButtonRelease-1>", self.b1up)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.predict_button=tk.Button(self, text="Predict", command=self.predict)
        self.save_button=tk.Button(self, text="Save", command=self.save)
        self.train_button=tk.Button(self, text="Train", command=self.train)
        self.ip=tk.Entry(self)
        self.ip.pack(pady=5)
        self.canvas.pack(pady=10)
        self.save_button.pack(pady=5)
        self.clear_button.pack(pady=5)
        self.predict_button.pack(pady=5)
        self.train_button.pack(pady=5)
        self.protocol("WM_DELETE_WINDOW", self.close_save)
    def save_data(self):
        global master_data
        with open(json_file,'w') as fp:
            json.dump(master_data,fp)
    def close_save(self):
        self.save_data()
        self.destroy()
    def clear(self):
    	self.canvas.delete("all")
    def predict(self):
        self.clf=pickle.load(open("classifier.pkl","rb"))
        global f_size
        im=self.save_img()
        arr=self.img_Array(im)
        eg=arr
        print(np.transpose(eg.reshape(f_size,f_size)))
        output=self.clf.predict(np.array(arr).reshape(1,-1))[0]
        #print(output)
        self.ip.delete(0,1)
        self.ip.insert(0,output)
    def b1down(self,event):
        global b1
        b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-

    def b1up(self,event):
        global b1, xold, yold
        b1 = "up"
        xold = None           # reset the line when you let go of the button
        yold = None

    def motion(self,event):
        global b1
        if b1 == "down":
            global xold, yold
            if xold is not None and yold is not None:
                event.widget.create_line(xold,yold,event.x,event.y,smooth=bool(1 or 1),width=4)
                          # here's where you draw it. smooth. neat.
            xold = event.x
            yold = event.y

    def save(self):
        im=self.save_img()
        inp=self.ip.get()
        global master_data
        if inp=='':
            print("Enter training value")
        else:
            print("Saving image for ",inp)
            arr=self.img_Array(im)
            dat={"image":arr.tolist(),"val":self.ip.get()}
            #print(dat)
            #with open(json_file) as fr:
                #master_data=json.load(fr)
            #print(master_data)
            master_data.append(dat)
            #with open(json_file,'w') as fp:
                #json.dump(master_data,fp)
            self.canvas.delete("all")
    def find_extremes(self,im):
        global f_size
        tmp_im=im
        pixels=tmp_im.load()
        left=f_size
        right=0
        top=f_size
        bottom=0
        print(pixels)
        for i in range(f_size):
            for j in range(f_size):
                if pixels[i,j]==0:
                    if j<left:
                        left=j
                    if j>right:
                        right=j
                    if i<top:
                        top=i
                    if i>bottom:
                        bottom=i
        #print(left,top,right,bottom)
        tmp_im=im.crop((top,left,bottom,right))
        return tmp_im
    def save_img(self):
        print("***Beginning of save_img()***")
        global f_size
        print("Saving to postscript")
        ps = self.canvas.postscript(colormode='mono')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        print("Saving to temporary image")
        img.save(desktop_path+'/tmp.bmp')
        im = Image.open("/home/arnav/Desktop/tmp.bmp")
        print("Converting to threshold")
        im=im.convert('1')
        print("Resizing to ",f_size,",",f_size)
        im=im.resize((f_size,f_size),Image.ANTIALIAS)
        print("Scaling to fit training data")
        im=self.find_extremes(im)
        print("Resizing to fit training data")
        im=im.resize((f_size,f_size),Image.ANTIALIAS)
        im.save(desktop_path+'/tmp1.bmp')
        print("Save to image complete")
        print("***End of save_img()***")
        return im
    def img_Array(self,im):
        print("***Beginning of img_Array()***")
        print("Starting to convert to array")
        global f_size
        pixels=im.load()
        arr=np.zeros((f_size,f_size),dtype=int)
        #print(pixels[4,4])
        for i in range(f_size):
            for j in range(f_size):
                if pixels[i,j]!=0:
                    arr[i][j]=1
        arr=arr.reshape(f_size**2)
        print("Conversion to array complete")
        print("***End of img_Array()***")
        return arr
    def train_test(self,X,y):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
        classifier=neighbors.KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        accuracy = classifier.score(X_test, y_test)
        #print(accuracy)
        return accuracy,classifier
    def train(self):
        global f_size
        self.clf = neighbors.KNeighborsClassifier()
        self.save_data()
        df = pd.read_json(json_file)
        Img=df[['image']].values.tolist()
        self.X=np.array(Img)
        div_x=f_size**2
        print(div_x,)
        self.X=self.X.reshape(int(self.X.size/div_x),div_x)
        self.y = np.array(df['val'])
        print("Size of training data is: ",self.y.size)
        eg=self.X[-1]
        print("A sample array picture looks like")
        print(np.transpose(eg.reshape(f_size,f_size)))
        acc=0
        for i in range(1000):
            acc_new,classifier=self.train_test(self.X,self.y)
            if(acc<acc_new):
                print("Old accuracy was: ",acc)
                acc=acc_new
                self.clf=classifier
        print("Final accuracy is: ",acc)
        with open("classifier.pkl",'wb') as fid:
            pickle.dump(self.clf,fid)
app = App()
app.mainloop()
