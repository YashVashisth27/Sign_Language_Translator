from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import operator
import time
import sys, os
import matplotlib.pyplot as plt
import hunspell
from string import ascii_uppercase
from datetime import datetime
###########################################################
# WITH MP3 FORMAT
from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os
import subprocess
from playsound import playsound
# 
###########################################################

# WITHOUT SAVING INTO MP3
import pyttsx3
#
class Application:

    def __init__(self):
    	
        self.directory = 'D:/College Major/HBTY/model'
        self.hs = hunspell.Hunspell()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        self.json_file = open(self.directory+"/signs.json", "r")
        # self.json_file = open(self.directory+"/signs.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)

        # self.loaded_model.load_weights(self.directory+"/signs.h5")
        self.loaded_model.load_weights(self.directory+"/signs.h5")
        
        #Orignal
        #self.json_file = open(self.directory+"/model-bw.json", "r")
        #self.model_json = self.json_file.read()
        #self.json_file.close()
        #self.loaded_model = model_from_json(self.model_json)
        #self.loaded_model.load_weights(self.directory+"/model-bw.h5")

        # self.json_file_dru = open(self.directory+"/model-bw_dru.json" , "r")
        # self.model_json_dru = self.json_file_dru.read()
        # self.json_file_dru.close()
        # self.loaded_model_dru = model_from_json(self.model_json_dru)
        # self.loaded_model_dru.load_weights("model/model-bw_dru.h5")

        # self.json_file_tkdi = open(self.directory+"/model-bw_tkdi.json" , "r")
        # self.model_json_tkdi = self.json_file_tkdi.read()
        # self.json_file_tkdi.close()
        # self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        # self.loaded_model_tkdi.load_weights(self.directory+"/model-bw_tkdi.h5")

        # self.json_file_smn = open(self.directory+"/model-bw_smn.json" , "r")
        # self.model_json_smn = self.json_file_smn.read()
        # self.json_file_smn.close()
        # self.loaded_model_smn = model_from_json(self.model_json_smn)
        # self.loaded_model_smn.load_weights(self.directory+"/model-bw_smn.h5")
        
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
          self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1280x720")
        self.panel = tk.Label(self.root)
        self.panel.place(x = 135, y = 10, width = 640, height = 640)
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 460, y = 95, width = 310, height = 310)
        
        self.T = tk.Label(self.root)
        self.T.place(x=31,y = 17)
        self.T.config(text = "Sign Language to Text",font=("courier",20,"bold"))
        self.panel3 = tk.Label(self.root) # Current SYmbol
        self.panel3.place(x = 1170,y = 150)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 800,y = 150)
        self.T1.config(text="Character :",font=("Courier",40,"bold"))
        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 800,y=350)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 800,y = 250)
        self.T2.config(text ="Word :",font=("Courier",40,"bold"))
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350,y=500)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10,y = 500)
        self.T3.config(text ="Sentence :",font=("Courier",20,"bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 20,y = 550)
        self.T4.config(text = "Suggestions",fg="red",font = ("Courier",40,"bold"))

        self.btcall = tk.Button(self.root,command = self.action_call,height = 0,width = 0)
        self.btcall.config(text = "Read",font = ("Courier",14))
        self.btcall.place(x = 1150, y = 0)

        self.btcall = tk.Button(self.root,command = self.action6,height = 0,width = 0)
        self.btcall.config(text = "backspace",font = ("Courier",14))
        self.btcall.place(x = 1000, y = 0)


        self.bt1=tk.Button(self.root, command=self.action1,height = 0,width = 0)
        self.bt1.place(x = 26,y=600)
        #self.bt1.grid(padx = 10, pady = 10)
        self.bt2=tk.Button(self.root, command=self.action2,height = 0,width = 0)
        self.bt2.place(x = 500,y=600)
        #self.panel3.place(x = 10,y=660)
        # self.bt2.grid(row = 4, column = 1, columnspan = 1, padx = 10, pady = 10, sticky = tk.NW)
        self.bt3=tk.Button(self.root, command=self.action3,height = 0,width = 0)
        self.bt3.place(x = 1000,y=600)
        # self.bt3.grid(row = 4, column = 2, columnspan = 1, padx = 10, pady = 10, sticky = tk.NW)
        self.bt4=tk.Button(self.root, command=self.action4,height = 0,width = 0)
        self.bt4.place(x = 250,y=650)
        # self.bt4.grid(row = bt1, column = 0, columnspan = 1, padx = 10, pady = 10, sticky = tk.N)
        self.bt5=tk.Button(self.root, command=self.action5,height = 0,width = 0)
        self.bt5.place(x = 750,y=650)
        # self.bt5.grid(row = 5, column = 1, columnspan = 1, padx = 10, pady = 10, sticky = tk.N)
        self.str=""
        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            frame = cv2.flip(frame, 1)
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,3)
            
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            res = cv2.flip(res,1)
            cv2.imwrite('test.jpg',res)
            res = cv2.imread('test.jpg')
            self.predict(res)
            res = cv2.flip(res,1)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol,font=("Courier",50))
            self.panel4.config(text=self.word,font=("Courier",40))
            self.panel5.config(text=self.str,font=("Courier",40))
            predicts=self.hs.suggest(self.word)
            if(len(predicts) > 0):
                self.bt1.config(text=predicts[0],font = ("Courier",20))
            else:
                self.bt1.config(text="")
            if(len(predicts) > 1):
                self.bt2.config(text=predicts[1],font = ("Courier",20))
            else:
                self.bt2.config(text="")
            if(len(predicts) > 2):
                self.bt3.config(text=predicts[2],font = ("Courier",20))
            else:
                self.bt3.config(text="")
            if(len(predicts) > 3):
                self.bt4.config(text=predicts[3],font = ("Courier",20))
            else:
                self.bt4.config(text="")
            if(len(predicts) > 4):
                self.bt4.config(text=predicts[4],font = ("Courier",20))
            else:
                self.bt4.config(text="")                
        self.root.after(30, self.video_loop)
    def predict(self,test_image):
        dataset = []
        test_image = cv2.resize(test_image, (64,64))
        img_array= img_to_array(test_image)
        dataset.append(img_array)
        dataset_array=np.array(dataset)
        dataset_array=dataset_array/255
        result = self.loaded_model.predict(dataset_array)
        # result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
        # result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
        # result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
        prediction={}
        
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        #LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        #LAYER 2
        # if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
        # 	prediction = {}
        # 	prediction['D'] = result_dru[0][0]
        # 	prediction['R'] = result_dru[0][1]
        # 	prediction['U'] = result_dru[0][2]
        # 	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        # 	self.current_symbol = prediction[0][0]

        # if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
        # 	prediction = {}
        # 	prediction['D'] = result_tkdi[0][0]
        # 	prediction['I'] = result_tkdi[0][1]
        # 	prediction['K'] = result_tkdi[0][2]
        # 	prediction['T'] = result_tkdi[0][3]
        # 	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        # 	self.current_symbol = prediction[0][0]

        # if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
        # 	prediction1 = {}
        # 	prediction1['M'] = result_smn[0][0]
        # 	prediction1['N'] = result_smn[0][1]
        # 	prediction1['S'] = result_smn[0][2]
        # 	prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        # 	if(prediction1[0][0] == 'S'):
        # 		self.current_symbol = prediction1[0][0]
        # 	else:
        # 		self.current_symbol = prediction[0][0]
        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if(self.ct[self.current_symbol] > 20):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol
    def action1(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 0):
            self.word=""
            self.str+=" "
            self.str+=predicts[0]
    def action2(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 1):
            self.word=""
            self.str+=" "
            self.str+=predicts[1]
    def action3(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 2):
            self.word=""
            self.str+=" "
            self.str+=predicts[2]
    def action4(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 3):
            self.word=""
            self.str+=" "
            self.str+=predicts[3]
    def action5(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 4):
            self.word=""
            self.str+=" "
            self.str+=predicts[4]
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self) :
        
        print("Saying current sentense: ")
        mytext = self.str;
        language = 'en'

        myobj = gTTS(text=mytext, lang=language, slow=False)
        date_string = datetime.now().strftime("%d%m%Y%H%M%S")
        myobj.save("welcome"+date_string+".mp3")
        playsound("welcome"+date_string+".mp3")

        engine = pyttsx3.init()
        engine.setProperty('rate', 145)
        voices = engine.getProperty('voices')
        engine.say(mytext)
        engine.runAndWait()

    def action6(self):
    	
        self.word = self.word[:-1]
        print(self.word)


print("Starting Application...")
pba = Application()
pba.root.mainloop()
