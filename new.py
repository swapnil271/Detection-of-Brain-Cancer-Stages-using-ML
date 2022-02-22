import tkinter

root = tkinter.Tk()
root.geometry('720x400')

def hello():
    print("Hello World")


label = tkinter.Label(text="Brain Cancer Detection")
label.pack()

rb1 = tkinter.Radiobutton(text="Detection")
rb1.pack()

btn = tkinter.Button(text="Browse", command=hello)
btn.pack()

root.mainloop()