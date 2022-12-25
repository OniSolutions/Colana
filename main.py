from collections import Counter 
from sklearn.cluster import KMeans 
from matplotlib import colors 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

"""Tkinner to generate windows for programm"""

from tkinter import *
from tkinter import ttk

root = Tk()
root.title("Colana")
root.geometry("660x600") 
 
def click():
    window = Tk()
    window.title("Colana Result")
    window.geometry("660x660")
 
button = ttk.Button(text="Open File", command=click)
button.pack(anchor=CENTER, expand=1)
 
root.mainloop()

"""Main code"""

image = cv2.imread('my_image.png') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess(raw):
    image = cv2.resize(raw, (900, 600), interpolation = cv2.INTER_AREA)                                          
    image = image.reshape(image.shape[0]*image.shape[1], 3)
    return image
    
def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        hex_color += ("{:02x}".format(int(i)))
    return hex_color

def analyze(img):
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]

    plt.figure(figsize = (12, 8))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    plt.savefig("result/my_pie.png")
    print("Found the following colors:\n")
    for color in hex_colors:
      print(color)

modified_image = preprocess(image) 
analyze(modified_image)
