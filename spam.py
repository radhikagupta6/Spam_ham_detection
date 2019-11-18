import sys
import nltk
nltk.download
import pandas as pd
import string
import tkinter as tk

from nltk.corpus import stopwords
stop_words=stopwords.words('english')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB

ps=PorterStemmer()
cv=CountVectorizer()

df = pd.read_csv('dataset/sms.txt',sep='\t',names=['label','messages'] )

def clean_text(msg):
    '''
    1:remove punctuation
    2:remove stopwords
    3:steming
    '''
    new_msg=[w for w in msg if w not in string.punctuation]
    new_msg2=''.join(new_msg)
    tmp_list=[]
    for ww in new_msg2.split():        
        if(ww.lower() not in stop_words):
            tmp_list.append(ww.lower())
    new_msg3=' '.join(tmp_list)
    new_msg4=[ps.stem(w) for w in new_msg3.split()]
    return ' '.join(new_msg4)
df['messages']=df.messages.apply(clean_text)
sparse_mtr=cv.fit_transform(df.messages)
X=sparse_mtr.toarray()
y=df.label	
gnb=GaussianNB()
gnb.fit(X,y)

def interface_main():
    global val, w, root
    root = tk.Tk()
    top = front_end (root)
    root.mainloop()

def predict(top):
	test=clean_text(top.Entry1.get())
	test_X=cv.transform([test])
	pred=gnb.predict(test_X.toarray())
	top.Label2.configure(text=pred[0])

class front_end:
	def __init__(self, top=None):
		_bgcolor = 'gray27'    # X11 color: 'gray85'
		_fgcolor = '#000000'   # X11 color: 'black'
		_compcolor = '#d9d9d9' # X11 color: 'gray85'
		_ana1color = '#d9d9d9' # X11 color: 'gray85'
		_ana2color = '#ececec' # Closest X11 color: 'gray92'
		font14 = "-family {Lucida Console} -size 18 -weight bold -slant roman -underline 0 -overstrike 0"
		font15 = "-family {Lucida Console} -size 14 -weight bold -slant roman -underline 0 -overstrike 0"
		font18 = "-family {Lucida Console} -size 25 -weight bold -slant roman -underline 0 -overstrike 0"
		top.geometry("856x450+209+115")
		top.title("SMS Spam Detection")
		top.configure(background="gray27")
		self.Label1 = tk.Label(top)
		self.Label2 = tk.Label(top)
		self.Label1.place(relx=0.390, rely=0.244, height=31, width=190)
		self.Label1.configure(background="gray27")
		self.Label1.configure(font=font14)
		self.Label1.configure(text='''ENTER MESSAGE''',fg='mint cream')
		
		self.Entry1 = tk.Entry(top)
		self.Entry1.place(relx=0.260, rely=0.377,height=30, relwidth=0.500)
		self.Entry1.configure(background="white")
		self.Entry1.configure(width=264)
		self.Entry1.configure('',font=font14)
		
		self.Button1 = tk.Button(top,command=lambda:predict(self))
		self.Button1.place(relx=0.390, rely=0.477, height=35)
		self.Button1.configure(background="honeydew4")
		self.Button1.configure(font=font15)
		self.Button1.configure(text='''DETECT RESULT''')

        
        
		self.Label2.place(relx=0.400, rely=0.600, height=31, width=120)
		self.Label2.configure(background="gray27")
		self.Label2.configure(disabledforeground="#a3a3a3")
		self.Label2.configure(font=font14)
		self.Label2.configure(foreground="#000000")
		self.Label2.configure(text='''''',fg='mint cream')

		
		self.Label3 = tk.Label(top)
		self.Label3.place(relx=0.282, rely=0.067, height=40, width=400)
		self.Label3.configure(activeforeground="gray27")
		self.Label3.configure(background="gray27")
		self.Label3.configure(disabledforeground="#a3a3a3")
		self.Label3.configure(font=font18)
		self.Label3.configure(foreground="#000000")
		self.Label3.configure(text='SMS SPAM DETECTION',fg='white')
		self.Label3.configure(width=317)

if __name__ == '__main__':
    interface_main()




