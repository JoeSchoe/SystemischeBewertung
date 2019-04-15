import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io
from tkinter import filedialog
from tkinter import *
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
import matplotlib.font_manager as font_manager
import seaborn as sns

font_dirs = ['Schriftarten']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'Heuristica'
#plt.rcParams['mathtext.it'] = 'Heuristica:bold'
#plt.rcParams['mathtext.bf'] = 'Heuristica:bold'
#plt.rcParams['font.family'] = 'Heuristica'
plt.rcParams['figure.figsize'] = [7.5, 5.6]
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

##Einlesen der Daten
root =Tk()
filename =  filedialog.askopenfilename(title = 'Wähle Anteil PV', filetypes = (("mat files","*.mat"),("all files","*.*")))
filename2 =  filedialog.askopenfilename(title = 'Öffne Datei mit Erzeugung durch BHKW', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename2 == '':
    filename2 = 'txt_zeros.txt'
filename3 =  filedialog.askopenfilename(title = 'Öffne Datei für EV-Batterien', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename3 == '':
    filename3 = 'txt_zeros.txt'
filename4 =  filedialog.askopenfilename(title = 'Öffne Datei für Speicherbatterien', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename4 == '':
    filename4 = 'txt_zeros.txt'
filename5 =  filedialog.askopenfilename(title = 'Öffne Datei powerHP', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename5 == '':
    filename5 = 'txt_zeros.txt'
filename6 =  filedialog.askopenfilename(title = 'Öffne Datei powerHR', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename6 == '':
    filename6 = 'txt_zeros.txt'
#filenameeeeeee =  filedialog.askopenfilename(title = 'Öffne Datei dotQDHW', filetypes = (("txt files","*.txt"),("all files","*.*")))

#filenameeeeeeee =  filedialog.askopenfilename(title = 'Öffne Datei SH', filetypes = (("txt files","*.txt"),("all files","*.*")))

#filenameeeeeeeee =  filedialog.askopenfilename(title = 'Öffne Datei mit dem realisierten Erzeugungen', filetypes = (("csv files","*.csv"),("all files","*.*")))

#filenameeeeeeeeee =  filedialog.askopenfilename(title = 'Öffne Datei mit CO2-Emissionensfaktoren', filetypes = (("csv files","*.csv"),("all files","*.*")))





##Erzeugung durch PV
new_dict = dict()
#mat = scipy.io.loadmat('Daten/profiles_LV_suburban_PV_25', mdict= new_dict)
mat = scipy.io.loadmat(filename, mdict= new_dict)
var1 = mat.get('p_gen')
var1 = var1/1000
l = len(var1)
l_35040 = len(var1)

array_p_gen2 = np.array([])   #Summe aller p_gen
for i in range(l):
	x = var1[i]
	y = (np.sum(x))
	array_p_gen2 = np.append(array_p_gen2, y)

array_p_gen = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_p_gen2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_p_gen = np.append(array_p_gen, mean_4)


##Verbrauch durch Haushalte
new_dict_2 = dict()
#mat = scipy.io.loadmat('Daten/profiles_LV_suburban_PV_25', mdict= new_dict_2)
mat = scipy.io.loadmat(filename, mdict= new_dict_2)
var2 = mat.get('p_dem')
var2 = var2/1000

array_p_dem2 = np.array([])   #Summe aller p_dem
for i in range(l):
	x = var2[i]
	y = (np.sum(x))
	array_p_dem2 = np.append(array_p_dem2, y)

array_p_dem = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_p_dem2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_p_dem = np.append(array_p_dem, mean_4)

##Erzeugung durch BHKW
#matrix_BHKW_gen = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerCHP.txt', skiprows=1)#
matrix_BHKW_gen = np.loadtxt(filename2, skiprows=0)
matrix_BHKW_gen = matrix_BHKW_gen[:35040, :]
array_BHKW_gen2 = matrix_BHKW_gen.sum(axis=1)

array_BHKW_gen = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_BHKW_gen2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_BHKW_gen = np.append(array_BHKW_gen, mean_4)

##Einfluss der EV-Batterien
#matrix_EV = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerEV.txt', skiprows=1)
matrix_EV = np.loadtxt(filename3, skiprows=0)
matrix_EV = matrix_EV[:35040, :]
array_EV2 = matrix_EV.sum(axis=1)

array_EV = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_EV2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_EV = np.append(array_EV, mean_4)

##Einfluss der Speicherbatterien
#matrix_Bat = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerBat.txt', skiprows=1)
matrix_Bat = np.loadtxt(filename4, skiprows=0)
matrix_Bat = matrix_Bat[:35040, :]
array_Bat2 = matrix_Bat.sum(axis=1)

array_Bat = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_Bat2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_Bat = np.append(array_Bat, mean_4)
len_Bat = len(array_Bat)

##Lade- und Endladevorgänge voneinander trennen und in 2 arrays packen
array_Bat_einspeisen = np.array([])
array_Bat_speichern = np.array([])
for i in range(len_Bat):
    if array_Bat[i] > 0:
        array_Bat_speichern = np.append(array_Bat_speichern, array_Bat[i])
        array_Bat_einspeisen = np.append(array_Bat_einspeisen, 0)
    elif array_Bat[i] <= 0:
        array_Bat_speichern = np.append(array_Bat_speichern, 0)
        array_Bat_einspeisen = np.append(array_Bat_einspeisen, array_Bat[i])

##Stromverbrauch durch HP
#matrix_HP = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerHP.txt', skiprows=1)
matrix_HP = np.loadtxt(filename5, skiprows=0)
matrix_HP = matrix_HP[:35040, :]
array_HP2 = matrix_HP.sum(axis=1)

array_HP = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_HP2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_HP = np.append(array_HP, mean_4)

##Stromverbrauch durch HR
#matrix_HR = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerHR.txt', skiprows=1)
matrix_HR = np.loadtxt(filename6, skiprows=0)
matrix_HR = matrix_HR[:35040, :]
array_HR2 = matrix_HR.sum(axis=1)

array_HR = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_HR2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_HR = np.append(array_HR, mean_4)


##Wärmebedarf Trinkwarmwasser
matrix_QDHW = np.loadtxt('Daten/dotQDHW.txt', skiprows=0)
#matrix_QDHW = np.loadtxt(filenameeeeeee, skiprows=1)
array_QDHW2 = matrix_QDHW.sum(axis=1)

array_QDHW = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_QDHW2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_QDHW = np.append(array_QDHW, mean_4)

##Wärmebedarf Heizen
matrix_SH = np.loadtxt('Daten/dotQSH.txt', skiprows=0)
#matrix_SH = np.loadtxt(filenameeeeeeee, skiprows=1)
array_SH2 = matrix_SH.sum(axis=1)

array_SH = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_SH2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_SH = np.append(array_SH, mean_4)

### matrizen in 1-Stunden-Takt ändern ###
matrix_HP2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_HP[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_HP2 = np.append(matrix_HP2, array2, axis=0)
matrix_HP2 = np.reshape(matrix_HP2, (120, 8760))
matrix_HP2 = matrix_HP2.T

matrix_HR2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_HR[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_HR2 = np.append(matrix_HR2, array2, axis=0)
matrix_HR2 = np.reshape(matrix_HR2, (120, 8760))
matrix_HR2 = matrix_HR2.T

matrix_Bat2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_Bat[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_Bat2 = np.append(matrix_Bat2, array2, axis=0)
matrix_Bat2 = np.reshape(matrix_Bat2, (120, 8760))
matrix_Bat2 = matrix_Bat2.T

matrix_EV2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_EV[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_EV2 = np.append(matrix_EV2, array2, axis=0)
matrix_EV2 = np.reshape(matrix_EV2, (120, 8760))
matrix_EV2 = matrix_EV2.T

matrix_BHKW_gen2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_BHKW_gen[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_BHKW_gen2 = np.append(matrix_BHKW_gen2, array2, axis=0)
matrix_BHKW_gen2 = np.reshape(matrix_BHKW_gen2, (120, 8760))
matrix_BHKW_gen2 = matrix_BHKW_gen2.T

var12 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = var1[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    var12 = np.append(var12, array2, axis=0)
var12 = np.reshape(var12, (120, 8760))
var12 = var12.T

var22 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = var2[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    var22 = np.append(var22, array2, axis=0)
var22 = np.reshape(var22, (120, 8760))
var22 = var22.T


l = len(array_SH)
number_of_months = int(12)
number_of_days = int(365)
number_of_weeks = int(52)
len_day = l / number_of_days
len_day = int(len_day)
len_week = 168
len_week = int(len_week)
len_month = l / number_of_months
len_month = int(len_month)

##Kabeltypauswahl für Bemessungsstrom
Bemessungsstrom_Kabeltypen = pd.read_csv("Strombelastbarkeit_Kabel_in_Erde.csv", skiprows=2, usecols=range(1,7), header=None, sep=";", engine='python')

##CO2-äquivalente Emissionen
erzeugung = pd.read_csv("TenneT_Realisierte Erzeugung_201501010000_201512312345_1.csv", skiprows=1, usecols=range(2,14), header=None, sep=";")
#erzeugung = pd.read_csv(filenameeeeeeeee, skiprows=1, usecols=range(2,14), header=None, sep=";")

ABC = pd.read_csv("CO2_Emissionsfaktoren.csv", skiprows=1, usecols=range(0,12), header=None, sep=";")
#ABC = pd.read_csv(filenameeeeeeeeee, skiprows=1, usecols=range(0,12), header=None, sep=";")

cos_phi = 0.95 #für Umrechnung von Wirk zu Scheinleistung

###Engpässe
##Engpassgefahr##

#Grundlast_min = np.min(var22)
#Engpassgefahr_Last = 630 - (AnzahlHP*LeistungHP)/cos_phi - Grundlast_min/cos_phi
#Engpassgefahr_Erzeugung = 630 - (AnzahlPV*LeistungPV)/cos_phi - (AnzahlBHKW*LeistungBHKW)/cos_phi + Grundlast_min/cos_phi


##Engpässe an den Hausanschlüssen
Leistung_Hausanschluss = var22 + matrix_EV2 + matrix_Bat2 + matrix_HP2 + matrix_HR2 + var12 - matrix_BHKW_gen2

Stromsicherungen = pd.read_csv("SicherungenHausanschlüsse.csv", skiprows=1, usecols=range(0,120), header=None, sep=";", engine='python')

Scheinleistung_Hausanschluss = Leistung_Hausanschluss / cos_phi

#matrix_Scheinleistung_Hausanschluss_pos = np.array([])
#matrix_Scheinleistung_Hausanschluss_neg = np.array([])
#for i in range(l):
#    array_pos = np.array([])
#    array_neg = np.array([])
#    for j in range(120):
#        if Scheinleistung_Hausanschluss[i, j] > 0:
#            array_pos = np.append(array_pos, Scheinleistung_Hausanschluss[i, j])
#            array_neg = np.append(array_neg, 0)
#        elif Scheinleistung_Hausanschluss[i, j] < 0:
#            array_pos = np.append(array_pos, 0)
#            array_neg = np.append(array_neg, Scheinleistung_Hausanschluss[i, j])
#    matrix_Scheinleistung_Hausanschluss_pos = np.append(matrix_Scheinleistung_Hausanschluss_pos, array_pos, axis=0)
#    matrix_Scheinleistung_Hausanschluss_neg = np.append(matrix_Scheinleistung_Hausanschluss_neg, array_neg, axis=0)


Bemessungsleistung_Hausanschluss = Stromsicherungen *3 *230 /1000       #/1000 -> kWh
array_Bemessungsleistung_Hausanschluss = Bemessungsleistung_Hausanschluss.values     #120 Häuser -> 120 Spalten

##Engpässe in den Kabeln
buildingsFeeder = np.loadtxt('Daten/buildingFeeder.txt')
unique, counts = np.unique(buildingsFeeder, return_counts=True)    #counts gibt Anzahl der Haushalte an der Feeder
dict(zip(unique, counts))                                          #unique gibt Nummerierung der Feeder an

u = counts[0]
v = counts[1]
w = counts[2]
x = counts[3]
y = counts[4]
z = counts[5]

feeder1 = Scheinleistung_Hausanschluss[:, :u].sum(axis=1)         #Wirkleistungen an den Sammelschienen
feeder2 = Scheinleistung_Hausanschluss[:, u:(u+v)].sum(axis=1)
feeder3 = Scheinleistung_Hausanschluss[:, (u+v):(u+v+w)].sum(axis=1)
feeder4 = Scheinleistung_Hausanschluss[:, (u+v+w):(u+v+w+x)].sum(axis=1)
feeder5 = Scheinleistung_Hausanschluss[:, (u+v+w+x):(u+v+w+x+y)].sum(axis=1)
feeder6 = Scheinleistung_Hausanschluss[:, (u+v+w+x+y):].sum(axis=1)

Bemessungsleistung_feeder1 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V    /1000 kVA
Bemessungsleistung_feeder2 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V    #Kabelauswahl nach Kerber Vorstadtnetz
Bemessungsleistung_feeder3 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V
Bemessungsleistung_feeder4 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V
Bemessungsleistung_feeder5 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V
Bemessungsleistung_feeder6 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V

array_Netzengpassleistung_feeder1_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder1_pos = np.array([])
array_Netzengpassleistung_feeder1_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder1_neg = np.array([])
for i in range(l):
    if feeder1[i] > 0:
        if feeder1[i] - Bemessungsleistung_feeder1 > 0:
            array_Netzengpassleistung_feeder1_pos = np.append(array_Netzengpassleistung_feeder1_pos, (feeder1[i] - Bemessungsleistung_feeder1))
        elif Bemessungsleistung_feeder1 - feeder1[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder1_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder1_pos, (Bemessungsleistung_feeder1 - feeder1[i]))
    if feeder1[i] < 0:
        if feeder1[i] + Bemessungsleistung_feeder1 < 0:
            array_Netzengpassleistung_feeder1_neg = np.append(array_Netzengpassleistung_feeder1_neg, (-1) * (feeder1[i] + Bemessungsleistung_feeder1))
        elif Bemessungsleistung_feeder1 + feeder1[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder1_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder1_neg, (Bemessungsleistung_feeder1 + feeder1[i]))

array_Netzengpassleistung_feeder2_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder2_pos = np.array([])
array_Netzengpassleistung_feeder2_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder2_neg = np.array([])
for i in range(l):
    if feeder2[i] > 0:
        if feeder2[i] - Bemessungsleistung_feeder2 > 0:
            array_Netzengpassleistung_feeder2_pos = np.append(array_Netzengpassleistung_feeder2_pos, (feeder2[i] - Bemessungsleistung_feeder2))
        elif Bemessungsleistung_feeder2 - feeder2[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder2_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder2_pos, (Bemessungsleistung_feeder2 - feeder2[i]))
    if feeder2[i] < 0:
        if feeder2[i] + Bemessungsleistung_feeder2 < 0:
            array_Netzengpassleistung_feeder2_neg = np.append(array_Netzengpassleistung_feeder2_neg, (-1) * (feeder2[i] + Bemessungsleistung_feeder2))
        elif Bemessungsleistung_feeder2 + feeder2[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder2_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder2_neg, (Bemessungsleistung_feeder2 + feeder2[i]))

array_Netzengpassleistung_feeder3_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder3_pos = np.array([])
array_Netzengpassleistung_feeder3_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder3_neg = np.array([])
for i in range(l):
    if feeder3[i] > 0:
        if feeder3[i] - Bemessungsleistung_feeder3 > 0:
            array_Netzengpassleistung_feeder3_pos = np.append(array_Netzengpassleistung_feeder3_pos, (feeder3[i] - Bemessungsleistung_feeder3))
        elif Bemessungsleistung_feeder3 - feeder3[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder3_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder3_pos, (Bemessungsleistung_feeder3 - feeder3[i]))
    if feeder3[i] < 0:
        if feeder3[i] + Bemessungsleistung_feeder3 < 0:
            array_Netzengpassleistung_feeder3_neg = np.append(array_Netzengpassleistung_feeder3_neg, (-1) * (feeder3[i] + Bemessungsleistung_feeder3))
        elif Bemessungsleistung_feeder3 + feeder3[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder3_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder3_neg, (Bemessungsleistung_feeder3 + feeder3[i]))

array_Netzengpassleistung_feeder4_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder4_pos = np.array([])
array_Netzengpassleistung_feeder4_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder4_neg = np.array([])
for i in range(l):
    if feeder4[i] > 0:
        if feeder4[i] - Bemessungsleistung_feeder4 > 0:
            array_Netzengpassleistung_feeder4_pos = np.append(array_Netzengpassleistung_feeder4_pos, (feeder4[i] - Bemessungsleistung_feeder4))
        elif Bemessungsleistung_feeder4 - feeder4[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder4_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder4_pos, (Bemessungsleistung_feeder4 - feeder4[i]))
    if feeder4[i] < 0:
        if feeder4[i] + Bemessungsleistung_feeder4 < 0:
            array_Netzengpassleistung_feeder4_neg = np.append(array_Netzengpassleistung_feeder4_neg, (-1) * (feeder4[i] + Bemessungsleistung_feeder4))
        elif Bemessungsleistung_feeder4 + feeder4[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder4_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder4_neg, (Bemessungsleistung_feeder4 + feeder4[i]))

array_Netzengpassleistung_feeder5_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder5_pos = np.array([])
array_Netzengpassleistung_feeder5_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder5_neg = np.array([])
for i in range(l):
    if feeder5[i] > 0:
        if feeder5[i] - Bemessungsleistung_feeder5 > 0:
            array_Netzengpassleistung_feeder5_pos = np.append(array_Netzengpassleistung_feeder5_pos, (feeder5[i] - Bemessungsleistung_feeder5))
        elif Bemessungsleistung_feeder5 - feeder5[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder5_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder5_pos, (Bemessungsleistung_feeder5 - feeder5[i]))
    if feeder5[i] < 0:
        if feeder5[i] + Bemessungsleistung_feeder5 < 0:
            array_Netzengpassleistung_feeder5_neg = np.append(array_Netzengpassleistung_feeder5_neg, (-1) * (feeder5[i] + Bemessungsleistung_feeder5))
        elif Bemessungsleistung_feeder5 + feeder5[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder5_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder5_neg, (Bemessungsleistung_feeder5 + feeder5[i]))

array_Netzengpassleistung_feeder6_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder6_pos = np.array([])
array_Netzengpassleistung_feeder6_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder6_neg = np.array([])
for i in range(l):
    if feeder6[i] > 0:
        if feeder6[i] - Bemessungsleistung_feeder6 > 0:
            array_Netzengpassleistung_feeder6_pos = np.append(array_Netzengpassleistung_feeder6_pos, (feeder6[i] - Bemessungsleistung_feeder6))
        elif Bemessungsleistung_feeder6 - feeder6[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder6_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder6_pos, (Bemessungsleistung_feeder6 - feeder6[i]))
    if feeder6[i] < 0:
        if feeder6[i] + Bemessungsleistung_feeder6 < 0:
            array_Netzengpassleistung_feeder6_neg = np.append(array_Netzengpassleistung_feeder6_neg, (-1) * (feeder6[i] + Bemessungsleistung_feeder6))
        elif Bemessungsleistung_feeder6 + feeder6[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder6_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder6_neg, (Bemessungsleistung_feeder6 + feeder6[i]))

plt.hist(array_Netzengpassleistung_feeder1_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Netzengpassleistung_feeder1_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung von Strang 1')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 1.png')
plt.ylabel('Häufigkeit')
plt.show()

plt.hist(array_Netzengpassleistung_feeder2_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Netzengpassleistung_feeder2_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung von Strang 2')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 2.png')
plt.show()

plt.hist(array_Netzengpassleistung_feeder3_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Netzengpassleistung_feeder3_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung von Strang 3')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 3.png')
plt.show()

plt.hist(array_Netzengpassleistung_feeder4_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Netzengpassleistung_feeder4_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung von Strang 4')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 4.png')
plt.show()

plt.hist(array_Netzengpassleistung_feeder5_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Netzengpassleistung_feeder5_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung von Strang 5')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 5.png')
plt.show()

plt.hist(array_Netzengpassleistung_feeder6_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Netzengpassleistung_feeder6_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung von Strang 6')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 6.png')
plt.show()


##Engpässe an der Ortnetzstation
Bemessungsleistung_Ortsnetzstation = 630                 # kVA maximale Transformatorleistung
Last_Ortsnetzstation = Scheinleistung_Hausanschluss.sum(axis=1)

array_Netzengpassleistung_Ortsnetzstation_pos = np.array([])
array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos = np.array([])
array_Netzengpassleistung_Ortsnetzstation_neg = np.array([])
array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg = np.array([])
for i in range(l):
    if Last_Ortsnetzstation[i] > 0:
        if Last_Ortsnetzstation[i] - Bemessungsleistung_Ortsnetzstation > 0:
            array_Netzengpassleistung_Ortsnetzstation_pos = np.append(array_Netzengpassleistung_Ortsnetzstation_pos, (Last_Ortsnetzstation[i] - Bemessungsleistung_Ortsnetzstation))
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos, 0)
        elif Bemessungsleistung_Ortsnetzstation - Last_Ortsnetzstation[i] > 0:
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos, (Bemessungsleistung_Ortsnetzstation - Last_Ortsnetzstation[i]))
    elif Last_Ortsnetzstation[i] < 0:
        if Last_Ortsnetzstation[i] + Bemessungsleistung_Ortsnetzstation < 0:
            array_Netzengpassleistung_Ortsnetzstation_neg = np.append(array_Netzengpassleistung_Ortsnetzstation_neg, (-1) * (Last_Ortsnetzstation[i] + Bemessungsleistung_Ortsnetzstation))
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg, 0)
        elif Bemessungsleistung_Ortsnetzstation + Last_Ortsnetzstation[i] > 0:
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg, (Bemessungsleistung_Ortsnetzstation + Last_Ortsnetzstation[i]))


plt.hist(array_Netzengpassleistung_Ortsnetzstation_pos, color='dimgray', bins=60, range=[0, 300])
plt.hist(array_Netzengpassleistung_Ortsnetzstation_neg, color='red', bins=60, alpha=0.5, range=[0, 300])
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung an der Ortsnetzstation')
plt.xlabel('Leistung [kVA]')
plt.ylabel('Anzahl stündlicher Messwerte')
plt.ylim((0, 25))
plt.xlim((0, 300))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_Ortsnetzstation.png')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos, color='dimgray', bins=140, range=[0, 700], label='Strombezug')
plt.hist(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg, color='red', bins=140, alpha=0.5, range=[0, 700], label='Stromeinspeisung')
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Leistungsaufnahmefähigkeit an der Ortsnetzstation')
plt.legend()
plt.xlabel('Leistung [kVA]')
plt.ylabel('Anzahl stündlicher Messwerte')
plt.ylim((0,400))
plt.xlim((0, 700))
plt.savefig('Häufigkeitsverteilung_der_Leistungsaufnahmefähigkeit_Ortsnetzstation.png')
plt.show()

##Engpassarbeit

Engpassarbeit_neg = np.sum(array_Netzengpassleistung_Ortsnetzstation_neg)   #kWh
print('Die Engpassarbeit an der Ortnetzstation, die durch Erzeugung entsteht, beträgt ' + str(Engpassarbeit_neg) + ' kWh')
Engpassarbeit_pos = np.sum(array_Netzengpassleistung_Ortsnetzstation_pos)      #kWh
print('Die Engpassarbeit an der Ortnetzstation, die durch Last entsteht, beträgt ' + str(Engpassarbeit_pos) + ' kWh')


###   GSC mit EE-Erzeugung im Quartier   ###

#array_p_gen_mean_d = np.array([]) #durchschn erzeugte Leistung pro tag
#for i in range(0, l, len_day):
#    P_GEN = np.array([])
#    for j in range(len_day):
#        b = np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])
#        P_GEN = np.append(P_GEN, b)
#    mean_d = np.mean(P_GEN)
#    array_p_gen_mean_d = np.append(array_p_gen_mean_d, mean_d)    #mittelwert erzeugte leistung pro tag
#
#array_p_gen_all = np.array([])
#for i in range(l):
#    a = np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])
#    array_p_gen_all = np.append(array_p_gen_all, a)
#
#
#array_p_dem_d = np.array([])  #Verbrauch pro tag
#for i in range(0, l, len_day):
#    P_DEM = np.array([])
#    for j in range(len_day):
#        c = array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]
#        P_DEM = np.append(P_DEM, c)
#
#    sum_d = np.sum(P_DEM)
#    array_p_dem_d = np.append(array_p_dem_d, sum_d)  # Verbrauch pro tag für 365 tage
#
#
#array_GSC_abs = np.array([])
#for jj in range(0, l, len_day):
#    i = 0
#    array_dem_mal_gen = np.array([])
#    for ii in range(len_day):
#        yy = (array_p_dem[ii + jj] + array_HP[ii + jj] + array_HR[ii + jj] + array_EV[ii + jj] + array_Bat_speichern[ii + jj]) * array_p_gen_all[ii + jj]
#        array_dem_mal_gen = np.append(array_dem_mal_gen, yy)
#
#    sum_array_dem_mal_gen = np.sum(array_dem_mal_gen)
#    GSC_abs = (sum_array_dem_mal_gen) / (array_p_dem_d[i] * array_p_gen_mean_d[i])
#    array_GSC_abs = np.append(array_GSC_abs, GSC_abs)
#    i = i + 1
#
#plt.hist(array_GSC_abs, bins=50)
#plt.title('Häufigkeitsverteilung GSC_abs')
#plt.ylabel('Menge an GSC_abs')
#plt.xlabel('GSC_abs')
#plt.xlim((0,5))
#plt.xticks(([0, 0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4, 4.5, 5]))
#plt.ylim((0, 65))
#plt.savefig('GSC_abs.png')
#plt.show()
#
#
#
#W_el_d_array = np.array([])
#for j in range(0, l, len_day):
#    W_el_d = np.array([])
#    for i in range(len_day):
#        y = array_p_dem[j + i] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]                    #alle Stromverbraucher!!!
#        W_el_d = np.append(W_el_d, y)       #
#
#    sum_W_el_d = np.sum(W_el_d)
#    W_el_d_array = np.append(W_el_d_array, sum_W_el_d)
#
#
#
#h_fl = W_el_d_array / ((120*4.8) + (0*10))   #Anzahl und Leistung der Erzeuger(PV und BHKW)   #BHKW 10 weil 9, höchste in txt datei
#
#h_fl_round = np.around(h_fl)
#h_fl_round2 = h_fl_round
#h_fl_round2 = h_fl_round2.astype(int)
#sum_h_fl_round2 = np.sum(h_fl_round2)
#
#h = -1
#array_pos_min = np.array([])
#array_pos_max = np.array([])
#for j in range(0, l, len_day):
#    h = h + 1#
#    x = h_fl_round2[h]
#    x = int(x)
#
#    array_gen_best_worst2 = np.array([])
#    for i in range(len_day):
#            z = np.absolute(array_p_gen[j + i]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])
#            array_gen_best_worst2 = np.append(array_gen_best_worst2, z)
#    array_gen_best_worst2_sorted = np.sort(array_gen_best_worst2)
#    array_gen_best_worst2_sorted = np.unique(array_gen_best_worst2_sorted)
#
#    pos_min_values = np.array([])
#    for q in range(x):
#        pos_min = np.where(array_gen_best_worst2 == array_gen_best_worst2_sorted[q])
#        pos_min = np.asarray(pos_min)
#        pos_min_values = np.append(pos_min_values, pos_min[0][:h_fl_round2[h]] + h * len_day)
#        if len(pos_min_values) >= h_fl_round2[h]:
#            pos_min_values = pos_min_values[:h_fl_round2[h]]
#            break
#    array_pos_min = np.append(array_pos_min, pos_min_values)
#
#    pos_max_values = np.array([])
#    for q in range(x):
#        pos_max = np.where(array_gen_best_worst2 == array_gen_best_worst2_sorted[-q - 1])
#
#        pos_max = np.asarray(pos_max)
#        pos_max_values = np.append(pos_max_values, pos_max[0][:h_fl_round2[h]] + h * len_day)
#        if len(pos_max_values) >= h_fl_round2[h]:
#            pos_max_values = pos_max_values[:h_fl_round2[h]]
#            break
#    array_pos_max = np.append(array_pos_max, pos_max_values)
#
#max_load_per_step = W_el_d_array / h_fl_round2      # Watt
#max_load_per_step = max_load_per_step.astype(int)
#
#
#Last_verlegt_max = np.zeros(l, dtype=int)
#Last_verlegt_min = np.zeros(l, dtype=int)
#
#m = 0
#for i in range(365):
#    for j in range(h_fl_round2[i]):
#        array_max = np.put(Last_verlegt_max, (array_pos_max[m + j]), max_load_per_step[i])
#    m = m + h_fl_round2[i]
#
#m = 0
#for i in range(365):
#    for j in range(h_fl_round2[i]):
#        array_min = np.put(Last_verlegt_min, (array_pos_min[m + j]), max_load_per_step[i])
#    m = m + h_fl_round2[i]
#
#
#array_p_dem_d_max = np.array([])  #Verbrauch pro tag
#for i in range(0, l, len_day):
#    P_DEM_max = np.array([])
#    for j in range(len_day):
#        c = Last_verlegt_max[i + j]
#        P_DEM_max = np.append(P_DEM_max, c)
#
#    sum_d = np.sum(P_DEM_max)
#    array_p_dem_d_max = np.append(array_p_dem_d_max, sum_d)  # Verbrauch pro tag für 365 tage
#
#array_GSC_abs_max = np.array([])
#for j in range(0, l, len_day):
#    läufer = 0
#    array_dem_mal_gen_max = np.array([])
#    for i in range(len_day):
#        yy = Last_verlegt_max[i + j] * array_p_gen_all[i + j]
#        array_dem_mal_gen_max = np.append(array_dem_mal_gen_max, yy)
#
#    sum_array_dem_mal_gen_max = np.sum(array_dem_mal_gen_max)
#    GSC_abs_max = (sum_array_dem_mal_gen_max) / (array_p_dem_d_max[läufer] * array_p_gen_mean_d[läufer])
#    array_GSC_abs_max = np.append(array_GSC_abs_max, GSC_abs_max)
#    läufer = läufer + 1
#
#array_GSC_abs_min = np.array([])
#for j in range(0, l, len_day):
#    läufer = 0
#    array_dem_mal_gen_min = np.array([])
#    for i in range(len_day):
#        yy = Last_verlegt_min[i + j] * array_p_gen_all[i + j]
#        array_dem_mal_gen_min = np.append(array_dem_mal_gen_min, yy)
#
#    sum_array_dem_mal_gen_min = np.sum(array_dem_mal_gen_min)
#    GSC_abs_min = (sum_array_dem_mal_gen_min) / (array_p_dem_d_max[läufer] * array_p_gen_mean_d[läufer])
#    array_GSC_abs_min = np.append(array_GSC_abs_min, GSC_abs_min)
#    läufer = läufer + 1
#
#
#GSC_rel = 200 *((array_GSC_abs_min - array_GSC_abs) / (array_GSC_abs_min - array_GSC_abs_max)) - 100
#
#GSC_rel_max = np.max(GSC_rel)
#GSC_rel_min = np.min(GSC_rel)
#
#print('Der maximale Tageswert von GSC_rel beträgt' + str(GSC_rel_max))
#print('Der minimale Tageswert von GSC_rel beträgt' + str(GSC_rel_min))
#
GSC_rel = np.array([])

plt.hist(GSC_rel, bins=100, color='red', range=[-100, 100])
#plt.title('Häufigkeitsverteilung des relativen GSC')
plt.xlim((-100,100))
plt.xticks([-100, -50, 0, 50, 100])
plt.ylim((0, 24))
plt.ylabel('Anzahl täglicher Werte')
plt.savefig('GSC_rel.png')
plt.show()

###   Deckungsgrad   ###

##pro tag
list1 = []
for i in range(0, l, len_day):
    p_dem_96 = np.array([])
    for j in range(len_day):
        cc = array_p_dem[i + j]
        p_dem_96 = np.append(p_dem_96, cc)
    sum1 = np.sum(p_dem_96)
    list1.append(sum1)
p_dem_ndarray = np.asarray(list1)

list2 = []
for i in range(0, l, len_day):
    p_gen_96 = np.array([])
    for j in range(len_day):
        cc = array_p_gen[i + j]
        p_gen_96 = np.append(p_gen_96, cc)
    sum2 = np.sum(p_gen_96)
    list2.append(sum2)
p_gen_ndarray = np.asarray(list2)

list3 = []
for i in range(0, l, len_day):
    BHKW_gen_96 = np.array([])
    for j in range(len_day):
        cc = array_BHKW_gen[i + j]
        BHKW_gen_96 = np.append(BHKW_gen_96, cc)
    sum3 = np.sum(BHKW_gen_96)
    list3.append(sum3)
BHKW_gen_ndarray = np.asarray(list3)

list4 = []
for i in range(0, l, len_day):
    EV_96 = np.array([])
    for j in range(len_day):
        cc = array_EV[i + j]
        EV_96 = np.append(EV_96, cc)
    sum4 = np.sum(EV_96)
    list4.append(sum4)
EV_ndarray = np.asarray(list4)

list5 = []
for i in range(0, l, len_day):
    Bat_einspeisen_96 = np.array([])
    for j in range(len_day):
        cc = array_Bat_einspeisen[i + j]
        Bat_einspeisen_96 = np.append(Bat_einspeisen_96, cc)
    sum5 = np.sum(Bat_einspeisen_96)
    list5.append(sum5)
Bat_einspeisen_ndarray = np.asarray(list5)

list6 = []
for i in range(0, l, len_day):
    Bat_speichern_96 = np.array([])
    for j in range(len_day):
        cc = array_Bat_speichern[i + j]
        Bat_speichern_96 = np.append(Bat_speichern_96, cc)
    sum6 = np.sum(Bat_speichern_96)
    list6.append(sum6)
Bat_speichern_ndarray = np.asarray(list6)

list7 = []
for i in range(0, l, len_day):
    HP_96 = np.array([])
    for j in range(len_day):
        cc = array_HP[i + j]
        HP_96 = np.append(HP_96, cc)
    sum7 = np.sum(HP_96)
    list7.append(sum7)
HP_ndarray = np.asarray(list7)

list8 = []
for i in range(0, l, len_day):
    HR_96 = np.array([])
    for j in range(len_day):
        cc = array_HR[i + j]
        HR_96 = np.append(HR_96, cc)
    sum8 = np.sum(HR_96)
    list8.append(sum8)
HR_ndarray = np.asarray(list8)
ll = len(HR_ndarray)

#Deckungsgrad
array_gamma_DG = np.array([])
for i in range(ll):

    if (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ) \
            > ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]) ):
        gamma_DG = (( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]) )
                    / (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] )) * 100
    elif ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]) ) \
            > (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ):
        gamma_DG = 100
    array_gamma_DG = np.append(array_gamma_DG, gamma_DG)

DG_min_monat = np.array([])
DG_max_monat = np.array([])
for i in range(0, number_of_days-5, 30):
    DG_min_max = np.array([])
    for j in range(30):
        cc = array_gamma_DG[i + j]
        DG_min_max = np.append(DG_min_max, cc)
    DG_min = np.min(DG_min_max)
    DG_max = np.max(DG_min_max)
    DG_min_monat = np.append(DG_min_monat, DG_min)
    DG_max_monat = np.append(DG_max_monat, DG_max)

DG_min_monat = DG_min_monat.astype(int)
DG_max_monat = DG_max_monat.astype(int)
print('Die minimalen täglichen Deckungsgrade der Monate' + str(DG_min_monat))
print('Die maximalen täglichen Deckungsgrade der Monate' + str(DG_max_monat))

#Eigenverbrauch
array_gamma_EV = np.array([])
for i in range(ll):

    if ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i])) \
            > (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ):
        gamma_EV = ((p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] )
                    / (np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]))) *100
    elif ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i])) == 0:
        gamma_EV = 0
    elif (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ) \
            > (  np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i])):
        gamma_EV = 100
    array_gamma_EV = np.append(array_gamma_EV, gamma_EV)

EV_min_monat = np.array([])
EV_max_monat = np.array([])
for i in range(0, number_of_days-5, 30):
    EV_min_max = np.array([])
    for j in range(30):
        cc = array_gamma_EV[i + j]
        EV_min_max = np.append(EV_min_max, cc)
    EV_min = np.min(EV_min_max)
    EV_max = np.max(EV_min_max)
    EV_min_monat = np.append(EV_min_monat, EV_min)
    EV_max_monat = np.append(EV_max_monat, EV_max)

EV_min_monat = EV_min_monat.astype(int)
EV_max_monat = EV_max_monat.astype(int)
print('Die minimalen täglichen Deckungsgrade der Monate' + str(EV_min_monat))
print('Die maximalen täglichen Deckungsgrade der Monate' + str(EV_max_monat))


#Autarkie
netto_stromlast = (array_p_dem + array_HP + array_HR + array_EV + array_Bat_speichern) \
                  - np.absolute(array_p_gen) - array_BHKW_gen - np.absolute(array_Bat_einspeisen)

#Autarkie
array_Autarkie_15_min = np.array([])
for q in range(l):
    if netto_stromlast[q] > 0:
        y = 1
    elif netto_stromlast[q] <= 0:
        y = 0
    array_Autarkie_15_min = np.append(array_Autarkie_15_min, y)

LOLP = np.sum(array_Autarkie_15_min) / l
Autarkie_Jahr = 1 - LOLP

print('Die Autarkie für das gesamte Jahr Beträgt' + str(Autarkie_Jahr))

LOLP_pro_Tag = np.array([])
for i in range(0, l, len_day):
    array_y_96 = np.array([])
    for y in range(len_day):
        a = array_Autarkie_15_min[i + y]
        array_y_96 = np.append(array_y_96, a)
        LOLP = np.sum(array_y_96) / len_day
    LOLP_pro_Tag = np.append(LOLP_pro_Tag, LOLP)
    Autarkie_pro_Tag = 1 - LOLP_pro_Tag

min_Autarkie = np.min(Autarkie_pro_Tag)
max_Autarkie = np.max(Autarkie_pro_Tag)
min_Autarkie = round(min_Autarkie, 2)
max_Autarkie = round(max_Autarkie, 2)

print('Die geringste tägliche Autarkie des Jahres beträgt' + str(min_Autarkie))
print('Die höchste tägliche Autarkie des Jahres beträgt' + str(max_Autarkie))

LOLP_pro_Monat = np.array([])
for i in range(0, l, len_month):
    array_y_2920 = np.array([])
    for y in range(len_month):
        a = array_Autarkie_15_min[i + y]
        array_y_2920 = np.append(array_y_2920, a)
        LOLP = np.sum(array_y_2920) / len_month
    LOLP_pro_Monat = np.append(LOLP_pro_Monat, LOLP)
    Autarkie_pro_Monat = 1 - LOLP_pro_Monat
Autarkie_pro_Monat = np.around(Autarkie_pro_Monat, decimals=2)

plt.bar(np.arange(len(Autarkie_pro_Monat)), Autarkie_pro_Monat, width = 0.3, color = 'red')
#plt.title('Autarkie pro Monat')
plt.ylabel('Energieautonomie')
#plt.xlabel('Monate')
plt.xticks(np.arange(len(Autarkie_pro_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.axhline(1, 0, 12, color='orange')
plt.ylim((0, 1.1))
plt.savefig('Autarkie_Monat.png')
plt.show()


##pro Monat
list1 = []
for i in range(0, l, len_month):
    p_dem_2920 = np.array([])
    for j in range(len_month):
        cc = array_p_dem[i + j]
        p_dem_2920 = np.append(p_dem_96, cc)
    sum1 = np.sum(p_dem_2920)
    list1.append(sum1)
p_dem_ndarray_monat = np.asarray(list1)

list2 = []
for i in range(0, l, len_month):
    p_gen_2920 = np.array([])
    for j in range(len_month):
        cc = array_p_gen[i + j]
        p_gen_2920 = np.append(p_gen_2920, cc)
    sum2 = np.sum(p_gen_2920)
    list2.append(sum2)
p_gen_ndarray_monat = np.asarray(list2)

list3 = []
for i in range(0, l, len_month):
    BHKW_gen_2920 = np.array([])
    for j in range(len_month):
        cc = array_BHKW_gen[i + j]
        BHKW_gen_2920 = np.append(BHKW_gen_2920, cc)
    sum3 = np.sum(BHKW_gen_2920)
    list3.append(sum3)
BHKW_gen_ndarray_monat = np.asarray(list3)

list4 = []
for i in range(0, l, len_month):
    EV_2920 = np.array([])
    for j in range(len_month):
        cc = array_EV[i + j]
        EV_2920 = np.append(EV_2920, cc)
    sum4 = np.sum(EV_2920)
    list4.append(sum4)
EV_ndarray_monat = np.asarray(list4)

list5 = []
for i in range(0, l, len_month):
    Bat_einspeisen_2920 = np.array([])
    for j in range(len_month):
        cc = array_Bat_einspeisen[i + j]
        Bat_einspeisen_2920 = np.append(Bat_einspeisen_2920, cc)
    sum5 = np.sum(Bat_einspeisen_2920)
    list5.append(sum5)
Bat_einspeisen_ndarray_monat = np.asarray(list5)

list6 = []
for i in range(0, l, len_month):
    Bat_speichern_2920 = np.array([])
    for j in range(len_month):
        cc = array_Bat_speichern[i + j]
        Bat_speichern_2920 = np.append(Bat_speichern_2920, cc)
    sum6 = np.sum(Bat_speichern_2920)
    list6.append(sum6)
Bat_speichern_ndarray_monat = np.asarray(list6)

list7 = []
for i in range(0, l, len_month):
    HP_2920 = np.array([])
    for j in range(len_month):
        cc = array_HP[i + j]
        HP_2920 = np.append(HP_2920, cc)
    sum7 = np.sum(HP_2920)
    list7.append(sum7)
HP_ndarray_monat = np.asarray(list7)

list8 = []
for i in range(0, l, len_month):
    HR_2920 = np.array([])
    for j in range(len_month):
        cc = array_HR[i + j]
        HR_2920 = np.append(HR_2920, cc)
    sum8 = np.sum(HR_2920)
    list8.append(sum8)
HR_ndarray_monat = np.asarray(list8)




#Deckungsgrad
array_gamma_DG_monat = np.array([])
for i in range(number_of_months):

    if (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ) \
            > ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]) ):
        gamma_DG = (( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]) )
                    / (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] )) * 100
    elif ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]) ) \
            > (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ):
        gamma_DG = 100
    array_gamma_DG_monat = np.append(array_gamma_DG_monat, gamma_DG)
array_gamma_DG_monat = array_gamma_DG_monat.astype(int)


#Eigenverbrauch
array_gamma_EV_monat = np.array([])
for i in range(number_of_months):

    if ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i])) \
            > (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ):
        gamma_EV = ((p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] )
                    / (np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]))) *100
    elif ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i])) == 0:
        gamma_EV = 0
    elif (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ) \
            > (  np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i])):
        gamma_EV = 100
    array_gamma_EV_monat = np.append(array_gamma_EV_monat, gamma_EV)
array_gamma_EV_monat = array_gamma_EV_monat.astype(int)

###für gesamte Jahr

##Deckungsgrad
if (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)) \
        > (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))):
    gamma_DG_Jahr = ((np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen)))
                     / (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern))) * 100
elif (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))) > \
            (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)):
        gamma_DG_Jahr = 100


##Eigenverbrauch
if (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))) \
        > (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)):
    gamma_EV_Jahr = ((np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern))/(np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen)))) * 100
elif (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))) == 0:
    gamma_EV_Jahr = 0
elif (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)) \
        > (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))):
    gamma_EV_Jahr = 100

print('Der bilanzielle Deckungsgrad für das gesamte Jahr beträgt' + str(gamma_DG_Jahr))
print('Der bilanzielle Eigenverbrauch für das gesamte Jahr beträgt' + str(gamma_EV_Jahr))

##Deckungsgrad
gamma_DG_Jahr_real = np.array([])
Zähler = np.array([])
Nenner = np.array([])
for i in range(l):
    if (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]) \
            > (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Zähler = np.append(Zähler, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
        Nenner = np.append(Nenner, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))

    elif (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]) < (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Zähler = np.append(Zähler, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))
        Nenner = np.append(Nenner, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))
DG_Jahr = np.sum(Zähler)/np.sum(Nenner)
gamma_DG_monat_real = np.append(gamma_DG_Jahr_real, DG_Jahr*100)

##Eigenverbrauch
gamma_EV_Jahr_real = np.array([])
Zähler = np.array([])
Nenner = np.array([])
for i in range(l):
        if (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])) \
              > (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]):
            Zähler = np.append(Zähler, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))

        elif (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]) > (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
            Zähler = np.append(Zähler, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
EV_Jahr = np.sum(Zähler)/np.sum(Nenner)
gamma_EV_Jahr_real = np.append(gamma_EV_Jahr_real, EV_Jahr*100)



###Berechnung der realen Werte pro Monat
##Deckungsgrad
array_gamma_DG_monat_real = np.array([])
for i in range(0, l, len_month):
    Zähler = np.array([])
    Nenner = np.array([])
    for j in range(len_month):
        if (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]) \
              > (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])):
            Zähler = np.append(Zähler, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))
            Nenner = np.append(Nenner, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))

        elif (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]) < (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])):
            Zähler = np.append(Zähler, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))
            Nenner = np.append(Nenner, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))
    DG_pro_Monat = np.sum(Zähler)/np.sum(Nenner)
    array_gamma_DG_monat_real = np.append(array_gamma_DG_monat_real, DG_pro_Monat*100)

##Eigenverbrauch
array_gamma_EV_monat_real = np.array([])
for i in range(0, l, len_month):
    Zähler = np.array([])
    Nenner = np.array([])
    for j in range(len_month):
        if (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])) \
              > (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]):
            Zähler = np.append(Zähler, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))

        elif (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]) > (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])):
            Zähler = np.append(Zähler, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))
    EV_pro_Monat = np.sum(Zähler)/np.sum(Nenner)
    array_gamma_EV_monat_real = np.append(array_gamma_EV_monat_real, EV_pro_Monat*100)


plt.bar(np.arange(len(array_gamma_DG_monat_real)),array_gamma_DG_monat_real, width=0.3, color = 'gray')
#plt.bar(np.arange(len(array_gamma_DG_monat))+0.15,array_gamma_DG_monat, width=0.3, color = 'gray')
#plt.plot(np.arange(len(DG_Monat_mean)), DG_min_monat, '.', color='black', markersize=12)
#plt.plot(np.arange(len(DG_Monat_mean)), DG_max_monat, '.', color='black', markersize=12)
#plt.title('Durchschnittlicher DG (blau) und \n bilanzieller DG(rot) pro Monat im Vergleich')
plt.ylabel('Deckungsgrad [%]')
#plt.xlabel('Monat')
plt.xticks(np.arange(len(array_gamma_DG_monat_real)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.axhline(100, 0, 12, color='orange')
plt.ylim((0, 110))
plt.savefig('Vergleich_Durchschnittlicher_bilanzieller_DG_Monat.png')
plt.show()

plt.bar(np.arange(len(array_gamma_EV_monat_real)),array_gamma_EV_monat_real, width=0.3, color = 'red')
#plt.bar(np.arange(len(array_gamma_EV_monat))+0.15,array_gamma_EV_monat, width=0.3, color = 'gray')
#plt.plot(np.arange(len(EV_Monat_mean)), EV_min_monat, '.', color='black', markersize=12)
#plt.plot(np.arange(len(EV_Monat_mean)), EV_max_monat, '.', color='black', markersize=12)
#plt.title('Durchschnittlicher EV (grau) und \n bilanzieller EV(dunkelrot) pro Monat im Vergleich')
plt.ylabel('Eigenverbauch [%]')
plt.axhline(100, 0, 12, color='orange')
#plt.xlabel('Monat')
plt.xticks(np.arange(len(array_gamma_EV_monat_real)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((0, 110))
plt.savefig('Vergleich_Durchschnittlicher_bilanzieller_EV_Monat.png')
plt.show()



###   Last am Ortsnetztransformator   ###

Residuallast = ((array_p_dem + array_HP + array_HR + array_EV + array_Bat_speichern )\
			   - (np.absolute(array_p_gen) + array_BHKW_gen + np.absolute(array_Bat_einspeisen)))  # --> kW


Residuallast_w = np.array([])                  #Werte für P_GL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l-len_day, len_week):
    Residuallast_Woche = np.array([])
    for j in range(len_week):
        cc = Residuallast[i + j]
        Residuallast_Woche = np.append(Residuallast_Woche, cc)
    Residuallast_w = np.append(Residuallast_w, Residuallast_Woche)
Residuallast_w  = np.reshape(Residuallast_w , (52, 168))

Residuallast_w_min = np.amin(Residuallast_w, axis=1)
Residuallast_w_max = np.amax(Residuallast_w, axis=1)

plt.bar(np.arange(len(Residuallast_w_min)),Residuallast_w_min, width=0.3, color = 'red')
plt.bar(np.arange(len(Residuallast_w_max)),Residuallast_w_max, width=0.3, color = 'gray')
plt.axhline(630, 0, 52, color='orange')
plt.axhline(-630, 0, 52, color='orange')
#plt.title('Maximale und minimale Residuallast des \n Quartiers innerhalb einer Woche')
#plt.xlabel('Woche')
plt.ylim((-900, 900))
plt.yticks((-900, -600, -300, 0, 300, 600, 900))
plt.ylabel('Last [kW]')
plt.savefig('Last_am_Ortsnetztransformator.png')
plt.show()

d = np.sum(Residuallast)

sum_Residualenergie = int(d /1000) #von kWh in MWh
print('Menge der Energie, die das Quartier bezieht(+)/abgibt(-)' + str(sum_Residualenergie) + ' MWh')

gradient = np.gradient(Residuallast)
min = gradient.min()
max = gradient.max()
print('Minimale Steigrate der Residuallast' + str(min))
print('Maximale Steigrate der Residuallast' + str(max))

plt.hist(gradient, bins=900, color='red', range=[-300, 300])
plt.ylabel('Anzahl stündlicher Messwerte')
plt.xlabel('Gradient [kW/h]')
plt.ylim((0, 200))
plt.xlim((-300,300))
plt.xticks(( -300, -200, -100, 0, 100, 200, 300))
#plt.title('Häufigkeitsverteilung der Gradienten')
plt.savefig('Häufigkeitsverteilung_der_Gradienten.png')
plt.show()

plt.hist(Residuallast, bins=800, color='red', range=[-800, 800])
#plt.title('Häufigkeitsverteilung der Residuallast im Quartier')
plt.ylabel('Anzahl stündlicher Messwerte')
plt.xlabel('Last [kW]')
plt.ylim((0,120))
plt.xlim((-800,600))
plt.xticks((-800, -600, -400, -200,  0, 200, 400, 600, 800))
#sns.distplot(Residuallast, norm_hist=False , kde=True, rug=True, bins=48, color = 'darkblue', rug_kws = {'linewidth': 3})
plt.savefig('Histogramm_Last_Ortsnetzstation.png')
plt.show()

###   Theoretischer Flexibilitätsbedarf des Quartiers   ###


P_GL = array_p_dem + array_HP + array_HR
P_PV = array_p_gen
P_BHKW = array_BHKW_gen

P_NL = P_GL + P_PV - P_BHKW #Werte in P_PV sind bereits negativ
p_base = np.mean(P_GL)

#Berechnung für jeden Tag d

P_GL_mean_d = np.array([])    #P_BASE für jeden Tag
for i in range(0, l, len_day):        #ändere step um Startpunkt für horizon k festzulegen, =4 entspricht 1 Stunde wie im paper
	P_GL_96 = np.array([])
	for j in range(len_day):          #ändere step um horizon k (siehe paper) zu ändern, 96 = 1 Tag
		aa = P_GL[i + j]
		P_GL_96 = np.append(P_GL_96,aa)
	P_BASE = np.mean(P_GL_96)
	P_GL_mean_d = np.append(P_GL_mean_d, P_BASE)
P_GL_mean_d = np.reshape(P_GL_mean_d, (365, 1))


P_NL_mean_d = np.array([])    #P_NL_mean für jeden Tag
for i in range(0, l, len_day):
	P_NL_96 = np.array([])
	for j in range(len_day):
		bb = P_NL[i + j]
		P_NL_96 = np.append(P_NL_96,bb)
	P_NL_mean = np.mean(P_NL_96)
	P_NL_mean_d = np.append(P_NL_mean_d, P_NL_mean)
P_NL_mean_d = np.reshape(P_NL_mean_d, (365, 1))


list1 = []                    #Werte für P_GL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l, len_day):
	P_GL_96 = np.array([])
	for j in range(len_day):
		cc = P_GL[i + j]
		P_GL_96 = np.append(P_GL_96, cc)
	list1.append(P_GL_96)
P_GL_ndarray = np.asarray(list1)


list2 = []                   #Werte für P_NL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l, len_day):
	P_NL_96 = np.array([])
	for j in range(len_day):
		dd = P_NL[i + j]
		P_NL_96 = np.append(P_NL_96, dd)
	list2.append(P_NL_96)
P_NL_ndarray = np.asarray(list2)

P_pu_NL = (P_NL_ndarray - P_NL_mean_d) / p_base     #"both with there mean value subtracted
P_pu_GL = (P_GL_ndarray - P_GL_mean_d) / p_base

acc_NL = np.cumsum(P_pu_NL, axis=1)
acc_GL = np.cumsum(P_pu_GL, axis=1)

A = P_pu_NL.flatten()   #blau
B = P_pu_GL.flatten()   #orange
plt.plot(A)
plt.plot(B)
plt.title('Normalisierte Lasten')
plt.ylabel('[Pu]')
plt.show()

C = acc_NL.flatten()
D = acc_GL.flatten()
plt.plot(C)
plt.plot(D)
plt.title('Akkumulierte Lasten pro Tag')
plt.show()


acc_NL_min = np.array([])
acc_NL_max = np.array([])
acc_GL_min = np.array([])
acc_GL_max = np.array([])

for i in range(0, l, len_day):
	a_96 = np.array([])
	b_96 = np.array([])

	for j in range(len_day):
		c_96 = C[i + j]
		a_96 = np.append(a_96, c_96)

		d_96 = D[i + j]
		b_96 = np.append(b_96, d_96)

	min_NL = np.min(a_96)
	max_NL = np.max(a_96)
	min_GL = np.min(b_96)
	max_GL = np.max(b_96)

	acc_NL_min = np.append(acc_NL_min, min_NL)
	acc_NL_max = np.append(acc_NL_max, max_NL)
	acc_GL_min = np.append(acc_GL_min, min_GL)
	acc_GL_max = np.append(acc_GL_max, max_GL)


storage_needed = ( (acc_NL_max - acc_NL_min) - (acc_GL_max - acc_GL_min) )*p_base /1000  #MegaWatt
plt.plot(storage_needed)
plt.title('Durch EE verursachte benötigte Speicher pro Tag')
plt.show()

yx = 0
for i in range(365):
    if storage_needed[i] < 0.3*0.95:
        yx = yx + 1
Abdeckung_tägl_Speicherbedarf = 100 * yx / 365

storage_needed_d_min = np.min(storage_needed)
storage_needed_d_max = np.max(storage_needed)

storage_needed_sorted = np.sort(storage_needed)

number_of_days = len(storage_needed)
blabla = number_of_days * 90 /100      #Auswahl der Tage um XX Prozent der benötigte Speicherkapazitäten (von tief nach hoch) abzudecken
blabla_round = round(blabla + 0.5)

storage_needed_percent = storage_needed_sorted[:blabla_round]
AA = np.max(storage_needed_percent)                       #Größe des Speichers wenn XY prozent abgedeckt werden sollen

##Berechnung für jede Woche w

P_GL_mean_w = np.array([])    #P_BASE für jede Woche
for i in range(0, l-len_day, len_week):        #ändere step um Startpunkt für horizon k festzulegen, =4 entspricht 1 Stunde wie im paper
	P_GL_730 = np.array([])
	for j in range(len_week):          #ändere step um horizon k (siehe paper) zu ändern, 96 = 1 Tag
		aa = P_GL[i + j]
		P_GL_730 = np.append(P_GL_730,aa)
	P_BASE = np.mean(P_GL_730)
	P_GL_mean_w = np.append(P_GL_mean_w, P_BASE)
P_GL_mean_w = np.reshape(P_GL_mean_w, (52, 1))


P_NL_mean_w = np.array([])    #P_NL_mean für jede woche
for i in range(0, l-len_day, len_week):    #730 statt 672
	P_NL_730 = np.array([])
	for j in range(len_week):
		bb = P_NL[i + j]
		P_NL_730 = np.append(P_NL_730,bb)
	P_NL_mean = np.mean(P_NL_730)
	P_NL_mean_w = np.append(P_NL_mean_w, P_NL_mean)
P_NL_mean_w = np.reshape(P_NL_mean_w, (52, 1))


list3 = []                    #Werte für P_GL für jede Woche in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l-len_day, len_week):
	P_GL_730 = np.array([])
	for j in range(len_week):
		cc = P_GL[i + j]
		P_GL_730 = np.append(P_GL_730, cc)
	list3.append(P_GL_730)
P_GL_ndarray_w = np.asarray(list3)


list4 = []                   #Werte für P_NL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l-len_day, len_week):
	P_NL_730 = np.array([])
	for j in range(len_week):
		dd = P_NL[i + j]
		P_NL_730 = np.append(P_NL_730, dd)
	list4.append(P_NL_730)
P_NL_ndarray_w = np.asarray(list4)

P_pu_NL_w = (P_NL_ndarray_w - P_NL_mean_w) / p_base
P_pu_GL_w = (P_GL_ndarray_w - P_GL_mean_w) / p_base

acc_NL_w = np.cumsum(P_pu_NL_w, axis=1)
acc_GL_w = np.cumsum(P_pu_GL_w, axis=1)

#AA = P_pu_NL_w.flatten()   #blau
#BB = P_pu_GL_w.flatten()   #orange
#plt.plot(AA)
#plt.plot(BB)
#plt.show()

CC = acc_NL_w.flatten()
DD = acc_GL_w.flatten()
plt.plot(CC)
plt.plot(DD)
plt.title('Wöchentlich akkumulierte Lasten')
plt.show()


acc_NL_min_w = np.array([])
acc_NL_max_w = np.array([])
acc_GL_min_w = np.array([])
acc_GL_max_w = np.array([])

for i in range(0, l-len_day, len_week):
	aa = np.array([])
	bb = np.array([])

	for j in range(len_week):
		cc = CC[i + j]
		aa = np.append(aa, cc)

		dd = DD[i + j]
		bb = np.append(bb, dd)

	min_NL_w = np.min(aa)
	max_NL_w = np.max(aa)
	min_GL_w = np.min(bb)
	max_GL_w = np.max(bb)

	acc_NL_min_w = np.append(acc_NL_min_w, min_NL_w)
	acc_NL_max_w = np.append(acc_NL_max_w, max_NL_w)
	acc_GL_min_w = np.append(acc_GL_min_w, min_GL_w)
	acc_GL_max_w = np.append(acc_GL_max_w, max_GL_w)


storage_needed_w = ( (acc_NL_max_w - acc_NL_min_w) - (acc_GL_max_w - acc_GL_min_w) )*p_base /1000 #MegaWatt
plt.bar(np.arange(len(storage_needed_w)), storage_needed_w, width=0.2, color = 'red')
plt.ylabel('Speichergröße [MWh]')
plt.ylim((0, 15))
#plt.title('Durch EE verursachte benötigte Speicher pro Woche')
plt.savefig('Benötigte_Speicher_pro_Woche.png')
plt.show()


storage_needed_sorted_w = np.sort(storage_needed_w)
storage_needed_w_max = np.max(storage_needed_w)

number_of_weeks = len(storage_needed_w)
blabla_weeks = number_of_weeks * 90 /100      #Auswahl der Tage um XX Prozent der benötigte Speicherkapazitäten (von tief nach hoch) abzudecken
blabla_round_weeks = round(blabla_weeks + 0.5)

storage_needed_percent_w = storage_needed_sorted_w[:blabla_round_weeks]
storage_needed_w_max_sorted = np.max(storage_needed_percent_w)

##Berechnung für jeden Monat

P_GL_mean_m = np.array([])    #P_BASE für jede Woche
for i in range(0, l, len_month):        #ändere step um Startpunkt für horizon k festzulegen, =4 entspricht 1 Stunde wie im paper
	P_GL_2920 = np.array([])
	for j in range(len_month):          #ändere step um horizon k (siehe paper) zu ändern, 96 = 1 Tag
		aa = P_GL[i + j]
		P_GL_2920 = np.append(P_GL_2920,aa)
	P_BASE = np.mean(P_GL_2920)
	P_GL_mean_m = np.append(P_GL_mean_m, P_BASE)
P_GL_mean_m = np.reshape(P_GL_mean_m, (12, 1))


P_NL_mean_m = np.array([])    #P_NL_mean für jeden Tag
for i in range(0, l, len_month):
	P_NL_2920 = np.array([])
	for j in range(len_month):
		bb = P_NL[i + j]
		P_NL_2920 = np.append(P_NL_2920,bb)
	P_NL_mean = np.mean(P_NL_2920)
	P_NL_mean_m = np.append(P_NL_mean_m, P_NL_mean)
P_NL_mean_m = np.reshape(P_NL_mean_m, (12, 1))


list5 = []                    #Werte für P_GL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l, len_month):
	P_GL_2920 = np.array([])
	for j in range(len_month):
		cc = P_GL[i + j]
		P_GL_2920 = np.append(P_GL_2920, cc)
	list5.append(P_GL_2920)
P_GL_ndarray_m = np.asarray(list5)


list6 = []                   #Werte für P_NL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l, len_month):
	P_NL_2920 = np.array([])
	for j in range(len_month):
		dd = P_NL[i + j]
		P_NL_2920 = np.append(P_NL_2920, dd)
	list6.append(P_NL_2920)
P_NL_ndarray_m = np.asarray(list6)

P_pu_NL_m = (P_NL_ndarray_m - P_NL_mean_m) / p_base
P_pu_GL_m = (P_GL_ndarray_m - P_GL_mean_m) / p_base

acc_NL_m = np.cumsum(P_pu_NL_m, axis=1)
acc_GL_m = np.cumsum(P_pu_GL_m, axis=1)

#AAA = P_pu_NL_m.flatten()   #blau
#BBB = P_pu_GL_m.flatten()   #orange
#plt.plot(AAA)
#plt.plot(BBB)
#plt.show()

CCC = acc_NL_m.flatten()
DDD = acc_GL_m.flatten()
plt.plot(CCC)
plt.plot(DDD)
plt.title('Monatlich akkumulierte Lasten ')
plt.show()


acc_NL_min_m = np.array([])
acc_NL_max_m = np.array([])
acc_GL_min_m = np.array([])
acc_GL_max_m = np.array([])

for i in range(0, l, len_month):
	aaa = np.array([])
	bbb = np.array([])

	for j in range(len_month):
		ccc = CCC[i + j]
		aaa = np.append(aaa, ccc)

		ddd = DDD[i + j]
		bbb = np.append(bbb, ddd)

	min_NL_m = np.min(aaa)
	max_NL_m = np.max(aaa)
	min_GL_m = np.min(bbb)
	max_GL_m = np.max(bbb)

	acc_NL_min_m = np.append(acc_NL_min_m, min_NL_m)
	acc_NL_max_m = np.append(acc_NL_max_m, max_NL_m)
	acc_GL_min_m = np.append(acc_GL_min_m, min_GL_m)
	acc_GL_max_m = np.append(acc_GL_max_m, max_GL_m)


storage_needed_m = ((acc_NL_max_m - acc_NL_min_m) - (acc_GL_max_m - acc_GL_min_m) )*p_base /1000 #MegaWatt
plt.bar(np.arange(len(storage_needed_m)), storage_needed_m, width=0.3, color = 'red')
plt.ylabel('Speicherkapazität [MWh]')
plt.ylim((0, 20))
#plt.title('Durch EE verursachte benötigte Speicher pro Monat')
plt.xticks(np.arange(len(storage_needed_m)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.savefig('Benötigter_Speicher_pro_Monat.png')
plt.show()

storage_needed_sorted_m = np.sort(storage_needed_m)
storage_needed_m_max = np.max(storage_needed_m)

number_of_months = len(storage_needed_m)
blabla_months = number_of_months * 90 /100      #Auswahl der Tage um XX Prozent der benötigte Speicherkapazitäten (von tief nach hoch) abzudecken
blabla_round_months = round(blabla_months + 0.5)

storage_needed_percent_m = storage_needed_sorted_m[:blabla_round_months]
storage_needed_m_max_sorted = np.max(storage_needed_percent_m)

##Berechnung für das gesamte Jahr

P_NL_mean_j = np.mean(P_NL)
P_GL_mean_j = np.mean(P_GL)

P_pu_NL_j = (P_NL - P_NL_mean_j) / p_base
P_pu_GL_j = (P_GL - P_GL_mean_j) / p_base

acc_NL_j = np.cumsum(P_pu_NL_j)
acc_GL_j = np.cumsum(P_pu_GL_j)

plt.plot(acc_NL_j)
plt.plot(acc_GL_j)
plt.title('Akkumulierte Lasten für das gesamte Jahr')
plt.savefig('Akkumulierte_Lasten_Jahr.png')
plt.ylim((-1800, 1800))
plt.show()

acc_NL_max_j = np.max(acc_NL_j)
acc_NL_min_j = np.min(acc_NL_j)
acc_GL_max_j = np.max(acc_GL_j)
acc_GL_min_j = np.min(acc_GL_j)
storage_needed_j = ((acc_NL_max_j - acc_NL_min_j) - (acc_GL_max_j - acc_GL_min_j) )* p_base /1000
storage_needed_j = int(storage_needed_j)
print('Die benötigte Speicherkapazität für den Betrachtungszeitraum von einem Jahr würde' + str(storage_needed_j) + ' MWh betragen')

### tatsächlich vorhandene Speicherkapazitäten

array_Umschaltpunkt_ges = np.zeros(365)
for q in range(120):

    loadFeed1 = np.zeros(l)
    loadFeed2 = np.zeros(l)

    for i in range(l):
        if matrix_Bat[i, q] > 0:
            loadFeed1[i] = matrix_Bat[i, q]
        else:
            loadFeed2[i] = matrix_Bat[i, q]

    soc = np.zeros(l)

    for t in range(l):
        if t == 0:
            if np.sum(matrix_Bat[:, q]) == 0:
                SOC_previous = 0
            else: SOC_previous = 5
        else:
            SOC_previous = soc[t-1]

        soc[t] = ((SOC_previous + (0.25*(loadFeed1[t]*0.91 + loadFeed2[t]))) - (1e-4)*SOC_previous*0.25)

    gradient = np.gradient(matrix_Bat[:, q], axis=0)

    array_Umschaltpunkt = np.array([])
    for i in range(0, l, 96):
        max_Umschaltpunkt = np.array([])
        array_Speicherstand_Bat2_Tag = np.array([])
        for j in range(95):
            array_Speicherstand_Bat2_Tag = np.append(array_Speicherstand_Bat2_Tag, soc[i + j])
            if gradient[i + j] > 0 and gradient[i + j +1] < 0:
                 max_Umschaltpunkt = np.append(max_Umschaltpunkt, soc[i + j])
            else: max_Umschaltpunkt = np.append(max_Umschaltpunkt, 0)
        if np.sum(max_Umschaltpunkt) == 0:
            array_Umschaltpunkt = np.append(array_Umschaltpunkt, np.max(array_Speicherstand_Bat2_Tag))
        else: array_Umschaltpunkt = np.append(array_Umschaltpunkt, np.max(max_Umschaltpunkt))

    array_Umschaltpunkt_ges = array_Umschaltpunkt_ges + array_Umschaltpunkt


array_Umschaltpunkt_ges = - np.sort(-array_Umschaltpunkt_ges)

plt.plot(array_Umschaltpunkt_ges, color='red')
plt.fill_between(np.arange(365), 0, array_Umschaltpunkt_ges, color='red', alpha= 0.5)
plt.ylabel('Speicherkapazität [kWh]')
plt.xlabel('Tage')
plt.xlim((0, 370))
plt.ylim((0, 1200))
plt.axhline(0.1*1200, 0, 365, color='orange')
plt.axhline(0.95*1200, 0, 365, color='orange')
plt.savefig('Speicherausnutzung_Bat')       #bbox_extra_artist=
plt.show()

area = np.trapz(array_Umschaltpunkt_ges, dx=1)
area = area - 0.1 * 1200 * 365
area_opti = 1200 * 365

Nutzungsgrad = (area / area_opti) *100

plt.plot(soc)
plt.show()

###   CO2-äquivalente Emissionen   ###

eta_netz = 0.9         #Verluste im Netz

total_rows = erzeugung.shape[0]   #Anzahl der Zeilen
total_columns = ABC.shape[1]

P_EE = np.absolute(array_p_gen) + array_BHKW_gen    #Erzeugung aus EE
Last_ges = array_p_dem + array_HP + array_HR + array_EV + array_Bat_speichern + array_Bat_einspeisen

Emissionsfaktor_PV = 0.101     #kg/kWh
Emissionsfaktor_BHKW = 0.4203    #Gas-BHKW  #https://www.umweltbundesamt.de/sites/default/files/medien/publikation/long/3476.pdf

array_Emissionsfaktor_EE = np.array([])
for i in range(l):
    if (np.absolute(array_p_gen[i]) + array_BHKW_gen[i]) == 0:
        Emissionsfaktor_EE = 0
    elif (np.absolute(array_p_gen[i]) + array_BHKW_gen[i]) > 0:
        Emissionsfaktor_EE = Emissionsfaktor_PV * (np.absolute(array_p_gen[i]) / (np.absolute(array_p_gen[i]) + array_BHKW_gen[i])) + \
                             Emissionsfaktor_BHKW * (array_BHKW_gen[i] / (np.absolute(array_p_gen[i]) + array_BHKW_gen[i]))
    array_Emissionsfaktor_EE = np.append(array_Emissionsfaktor_EE, Emissionsfaktor_EE)

##Stromerzeugung DEA
array_Stromerzeugung_DEA_Monat = np.array([])
for i in range(0, l, len_month):
    Stromerzeugung_DEA_Monat = np.array([])
    for j in range(len_month):
        a = array_BHKW_gen[i + j] + np.absolute(array_p_gen[i + j])
        Stromerzeugung_DEA_Monat = np.append(Stromerzeugung_DEA_Monat, a)
    sum = np.sum(Stromerzeugung_DEA_Monat)
    array_Stromerzeugung_DEA_Monat = np.append(array_Stromerzeugung_DEA_Monat, sum)

##Stromverbrauch pro Monat
array_Stromverbrauch_Monat = np.array([])
for i in range(0, l, len_month):
    Stromverbrauch_Monat = np.array([])
    for j in range(len_month):
        a = Last_ges[i + j]
        Stromverbrauch_Monat = np.append(Stromverbrauch_Monat, a)
    sum = np.sum(Stromverbrauch_Monat)
    array_Stromverbrauch_Monat = np.append(array_Stromverbrauch_Monat, sum)

array_Stromverbrauch_HP_Monat = np.array([])
for i in range(0, l, len_month):
    Stromverbrauch_HP_Monat = np.array([])
    for j in range(len_month):
        a = array_HP[i + j] + array_HR[i + j]
        Stromverbrauch_HP_Monat = np.append(Stromverbrauch_HP_Monat, a)
    sum = np.sum(Stromverbrauch_HP_Monat)
    array_Stromverbrauch_HP_Monat = np.append(array_Stromverbrauch_HP_Monat, sum)


##Diagramm für Stromverbrauch pro Monat
plt.bar(np.arange(len(array_Stromverbrauch_Monat))-0.15, array_Stromverbrauch_Monat/1000, width=0.3, color = 'red')
plt.bar(np.arange(len(array_Stromverbrauch_Monat))+0.15, array_Stromerzeugung_DEA_Monat/1000, width=0.3, color = 'gray')
plt.ylabel('Elektrische Energie [MWh]')
#plt.xlabel('Monate')
plt.xticks(np.arange(len(array_Stromverbrauch_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((0, 250))
plt.savefig('Stromverbrauch_pro_Monat.png')
plt.show()

##dynamischer CO2-Faktor des Quartiers
array_f_CO2_mix = np.array([])
for i in range(l):

    if (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])) >\
            (array_p_dem[i] + array_HR[i] + array_HP[i] + array_EV[i] + array_Bat_speichern[i]):
        f_CO2_mix = 0
        array_f_CO2_mix = np.append(array_f_CO2_mix, f_CO2_mix)

    else:
        array_P_f_CO2 = np.array([])
        array_P_ges = np.array([])
        for y in range(total_columns):
            P_ges = erzeugung.iat[i,y]      #Erzeugte Leistung pro viertel Stunde  #Leistung in Erzeugungsdaten von Tennet in MWh
            P_f_CO2 = P_ges * ABC.iat[0,y] #el Leistung * CO2-äquivalente Emissionen  #2016 durchsch. CO2-Emissionsfaktor 0.516 kg/kWh

            array_P_f_CO2 = np.append(array_P_f_CO2, P_f_CO2)
            array_P_ges = np.append(array_P_ges, P_ges)

        b = np.sum(array_P_f_CO2)      #summe aller P_f_CO2 für einen Zeitunkt
        c = np.sum(array_P_ges)
        d = b / eta_netz
        f_CO2_mix = d / c      #[kg/kWh]
        array_f_CO2_mix = np.append(array_f_CO2_mix, f_CO2_mix)



##Emissionsfaktor gemittelt für jeden Monat
list2 = []
for i in range(0, l, len_month):
    Emissionsfaktor_Monat = np.array([])
    for j in range(len_month):
        a = array_f_CO2_mix[i + j]
        Emissionsfaktor_Monat = np.append(Emissionsfaktor_Monat, a)
    mean2 = np.mean(Emissionsfaktor_Monat)
    list2.append(mean2)
Emissionsfaktor_Monatswerte = np.asarray(list2)

f = len(array_f_CO2_mix)
ff = np.empty(f)

##Diagramm für Emissionsfaktor pro Monat
plt.bar(np.arange(len(Emissionsfaktor_Monatswerte)), Emissionsfaktor_Monatswerte, width=0.2, color = 'orange')
plt.ylabel('Dynamischer CO2-äquivalenter Emissionsfaktor [kg/kWh]')
plt.xlabel('Monate')
plt.xticks(np.arange(len(Emissionsfaktor_Monatswerte)), ['Jan', 'Feb', 'März', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
plt.ylim((0, 1))
plt.savefig('Dynamischer CO2-äquivalenter Emissionsfaktor.png')
plt.show()

Last_Ortsnetztransformator = np.array([])    #Strom der ins Quartier bzw. aus dem Qiurtier fließt    #ohne Stromverbrauch der Wärmepumpen
Last_Ortsnetztransformator = (array_p_dem + array_EV + array_Bat_speichern + array_HP + array_HR)\
                    - (  np.absolute(array_p_gen) + array_BHKW_gen + np.absolute(array_Bat_einspeisen))

#array mit den Gesamtlasten in 2 arrays; Verbrauch und Einspeisung ins übergeordnete Netz
Last_einspeisen = np.array([])  #einspeisen ins übergeordnete Netz
Last_Strommix = np.array([])    #benötigter Strom wird mit Strommix gedeckt
for i in range(l):
    if Last_Ortsnetztransformator[i] > 0:
        Last_Strommix = np.append(Last_Strommix, Last_Ortsnetztransformator[i])
        Last_einspeisen = np.append(Last_einspeisen, 0)
    elif Last_Ortsnetztransformator[i] <= 0:
        Last_Strommix = np.append(Last_Strommix, 0)
        Last_einspeisen = np.append(Last_einspeisen, Last_Ortsnetztransformator[i])

array_Emissionen_HP = np.array([])
for i in range(l):
    if (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i]+ array_HP[i] + array_HR[i]) < ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Emissionen_HP = array_Emissionsfaktor_EE[i] * (array_HP[i] + array_HR[i])
    elif (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i] ) > ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Emissionen_HP = array_f_CO2_mix[i] * (array_HP[i] + array_HR[i])
    elif (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i]) < ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Emissionen_HP_EE = (( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])) - (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i])) * array_Emissionsfaktor_EE[i]
        Emissionen_HP_Strommix = array_f_CO2_mix[i] * ((array_p_dem[i] + array_EV[i] + array_Bat_speichern[i]+ array_HP[i] + array_HR[i]) - ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
        Emissionen_HP = Emissionen_HP_EE + Emissionen_HP_Strommix
    array_Emissionen_HP = np.append(array_Emissionen_HP, Emissionen_HP)

array_Emissionen_HP_Monat = np.array([])
for i in range(0, l, len_month):
    Emissionen_HP_Monat = np.array([])
    for j in range(len_month):
        a = array_Emissionen_HP[i + j]
        Emissionen_HP_Monat = np.append(Emissionen_HP_Monat, a)
    sum = np.sum(Emissionen_HP_Monat)
    array_Emissionen_HP_Monat = np.append(array_Emissionen_HP_Monat, sum)


array_Emissionen_EE_Verbrauch = np.array([])
for i in range(l):
    if P_EE[i] >= Last_ges[i]:
        array_Emissionen_EE_Verbrauch = np.append(array_Emissionen_EE_Verbrauch, Last_ges[i])
    elif Last_ges[i] > P_EE[i]:
        array_Emissionen_EE_Verbrauch = np.append(array_Emissionen_EE_Verbrauch, P_EE[i])


Emissionen_strommix = f_CO2_mix * Last_Strommix
Emissionen_EE = array_Emissionen_EE_Verbrauch * array_Emissionsfaktor_EE

list = []              #Emissionen pro Monat(siehe range)
for i in range(0, l, len_month):
    Emissionen_Monat = np.array([])
    for j in range(len_month):
        a = Emissionen_strommix[i + j] + Emissionen_EE[i + j]
        Emissionen_Monat  = np.append(Emissionen_Monat , a)
    sum = np.sum(Emissionen_Monat )
    list.append(sum)
Emissionen_Monatswerte = np.asarray(list)    #tatsächliche Emissionen durch gesamten Stromverbrauch; siehe a

list = []              #Emissionen pro Monat(siehe range)
for i in range(0, l, len_month):
    Emissionen_Monat_EE = np.array([])
    for j in range(len_month):
        a = Emissionen_EE[i + j]
        Emissionen_Monat_EE  = np.append(Emissionen_Monat_EE , a)
    sum = np.sum(Emissionen_Monat_EE )
    list.append(sum)
Emissionen_Monatswerte_EE = np.asarray(list)


array_f_CO2_mix_ohneEE = np.array([])    #dynamischer CO2-Faktor des übergeordneten System
for i in range(l):
    array_P_f_CO2_ohneEE = np.array([])
    array_P_ges_ohneEE = np.array([])
    for y in range(total_columns):
        P_ges = erzeugung.iat[i,y]
        P_f_CO2 = P_ges * ABC.iat[0,y]
        array_P_f_CO2_ohneEE = np.append(array_P_f_CO2_ohneEE, P_f_CO2)
        array_P_ges_ohneEE = np.append(array_P_ges_ohneEE, P_ges)

    b2 = np.sum(array_P_f_CO2_ohneEE)
    c2 = np.sum(array_P_ges_ohneEE)
    d2 = b2 / eta_netz
    f_CO2_mix = d2 / c2
    array_f_CO2_mix_ohneEE = np.append(array_f_CO2_mix_ohneEE, f_CO2_mix)

Mittelwert_f_CO2_mix = np.mean(array_f_CO2_mix_ohneEE)


list = []              #Emissionen pro Monat(siehe range)
for i in range(0, l, len_month):
    Emissionen_Monat_stat_strommix = np.array([])
    for j in range(len_month):
        a =  Last_Strommix[i + j] * Mittelwert_f_CO2_mix
        Emissionen_Monat_stat_strommix  = np.append(Emissionen_Monat_stat_strommix , a)
    sum = np.sum(Emissionen_Monat_stat_strommix )
    list.append(sum)
Emissionen_Monatswerte_statisch_strommix = np.asarray(list)


#Deckung des monatlichen Strombedarfs mit statischem Emissionsfaktor
Emissionen_Monatswerte_statisch = Emissionen_Monatswerte_statisch_strommix + Emissionen_Monatswerte_EE #auf die monatlichen Emissionen müssen noch die für Deckungs des Wärmebedarfs, wenn keine HP oder CHP vorhanden
array_Wärmebedarf_monat = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
array_Wärmebedarf_monat2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
if array_BHKW_gen.sum() + array_HP.sum() == 0:
    array_Wärmebedarf_monat = np.array([])
    for i in range(0, l, len_month):
        Wärmebedarf_monat = np.array([])
        for j in range(len_month):
            a = array_SH[i+j] + array_QDHW[i+j]
            Wärmebedarf_monat = np.append(Wärmebedarf_monat, a)
        sum = np.sum(Wärmebedarf_monat)
        array_Wärmebedarf_monat = np.append(array_Wärmebedarf_monat, sum)
if array_BHKW_gen.sum() > 0:
    array_Wärmebedarf_monat2 = np.array([])
    for i in range(0, l, len_month):
        Wärmebedarf_monat = np.array([])
        for j in range(len_month):
            a = array_SH[i + j] + array_QDHW[i + j]
            Wärmebedarf_monat = np.append(Wärmebedarf_monat, a)
        sum = np.sum(Wärmebedarf_monat)
        array_Wärmebedarf_monat2 = np.append(array_Wärmebedarf_monat2, sum)
Emissionen_Wärme_BHKW = array_Wärmebedarf_monat2 * 0.1956    #0.1956 g/kWh CO2e durch Gas-BHKW für Wärmeproduktion
Emissionen_Wärme_Gaskessel = array_Wärmebedarf_monat * 0.25/0.77
Emissionen_Gas_real = Emissionen_Wärme_Gaskessel + Emissionen_Wärme_BHKW
#Emissionen_Monatswerte = Emissionen_Monatswerte + Emissionen_Gas_real

array_Wärmebedarf_monat3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
if array_HP.sum() == 0:
    array_Wärmebedarf_monat3 = np.array([])
    for i in range(0, l, len_month):
        Wärmebedarf_monat = np.array([])
        for j in range(len_month):
            a = array_SH[i+j] + array_QDHW[i+j]
            Wärmebedarf_monat = np.append(Wärmebedarf_monat, a)
        sum = np.sum(Wärmebedarf_monat)
        array_Wärmebedarf_monat3 = np.append(array_Wärmebedarf_monat3, sum)
Emissionen_Gas_theo = array_Wärmebedarf_monat3 * 0.25/0.77


Emissionen_vergleich_ges = Emissionen_Monatswerte_statisch + Emissionen_Gas_theo
#Wirkungsgrad Gaskessel 77%  #https://www.haustechnikdialog.de/SHKwissen/1888/Wirkungs-und-Nutzungsgrad-einer-Heizungsanlage
#CO2e von Erdgas 250 g/kWh

Emissionen_ges = np.sum(Emissionen_strommix) + np.sum(Emissionen_EE) + np.sum(Emissionen_Gas_real)  #kg    #Emissionen die sich aus dem CO2-Faktor des Quartiers und der Gesamtlast/-nachfrage ergeben
Emissionen_ges = round(Emissionen_ges)
print('Verursachte Emissionen im Jahr in kg' + str(Emissionen_ges))

##Einsparungen durch Einspeisung
Fall_1 = array_f_CO2_mix_ohneEE * np.absolute(Last_einspeisen)
Fall_1_stat = Mittelwert_f_CO2_mix * np.absolute(Last_einspeisen)
Fall_2 = array_Emissionsfaktor_EE * np.absolute(Last_einspeisen)
list = []              #Einsparung pro Monat dynamisch
for i in range(0, l, len_month):
    Einsparung_pro_Monat = np.array([])
    for j in range(len_month):
        a =  Fall_1[i + j] - Fall_2[i + j]
        Einsparung_pro_Monat  = np.append(Einsparung_pro_Monat , a)
    sum = np.sum(Einsparung_pro_Monat)
    list.append(sum)
Einsparung_Monate = np.asarray(list)
list = []              #Einsparung pro Monat statisch
for i in range(0, l, len_month):
    Einsparung_pro_Monat = np.array([])
    for j in range(len_month):
        a =  Fall_1_stat[i + j] - Fall_2[i + j]
        Einsparung_pro_Monat  = np.append(Einsparung_pro_Monat , a)
    sum = np.sum(Einsparung_pro_Monat)
    list.append(sum)
Einsparung_Monate_stat = np.asarray(list)

Einsparung_Monate_stat_pos = np.array([])
Einsparung_Monate_stat_neg = np.array([])
Einsparung_Monate_pos = np.array([])
Einsparung_Monate_neg = np.array([])
for i in range(12):
    if Einsparung_Monate_stat[i] > 0:
        Einsparung_Monate_stat_pos = np.append(Einsparung_Monate_stat_pos, Einsparung_Monate_stat[i])
        Einsparung_Monate_stat_neg = np.append(Einsparung_Monate_stat_neg, 0)
    elif Einsparung_Monate_stat[i] <= 0:
        Einsparung_Monate_stat_pos = np.append(Einsparung_Monate_stat_pos, 0)
        Einsparung_Monate_stat_neg = np.append(Einsparung_Monate_stat_neg, Einsparung_Monate_stat[i])
for i in range(12):
    if Einsparung_Monate[i] > 0:
        Einsparung_Monate_pos = np.append(Einsparung_Monate_pos, Einsparung_Monate[i])
        Einsparung_Monate_neg = np.append(Einsparung_Monate_neg, 0)
    elif Einsparung_Monate[i] <= 0:
        Einsparung_Monate_pos = np.append(Einsparung_Monate_pos, 0)
        Einsparung_Monate_neg = np.append(Einsparung_Monate_neg, Einsparung_Monate[i])


plt.bar(np.arange(len(Emissionen_Monatswerte))-0.15, ((Emissionen_Monatswerte - Einsparung_Monate_neg)/1000), width=0.3, color = 'red')
plt.bar(np.arange(len(Emissionen_Monatswerte))-0.15,  (- Einsparung_Monate_pos /1000), width=0.3, color = 'orange')
plt.bar(np.arange(len(Emissionen_Monatswerte))-0.15, (Emissionen_Gas_real/1000), width=0.3, color = 'salmon', bottom=(Emissionen_Monatswerte/1000))
plt.bar(np.arange(len(Emissionen_Monatswerte_statisch))+0.15, ((Emissionen_Monatswerte_statisch - Einsparung_Monate_stat_neg)/1000), width=0.3, color = 'gray')
plt.bar(np.arange(len(Emissionen_Monatswerte_statisch))+0.15, ((- Einsparung_Monate_stat_pos)/1000), width=0.3, color = 'tan')
plt.bar(np.arange(len(Emissionen_Monatswerte_statisch))+0.15, (Emissionen_Gas_theo/1000), width=0.3, color = 'silver', bottom=(Emissionen_Monatswerte_statisch/1000))
#plt.title('Emissionen mit dynamischen Emissionsfaktor')
plt.ylabel('Emissionen [t]')
plt.yticks((-50, -25, 0, 25, 50, 75, 100, 125, 150))
#plt.xlabel('Monat')
plt.xticks(np.arange(len(Emissionen_Monatswerte)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((-50, 150))
plt.savefig('CO2_äquivalente_Emissionen_mit_dynamischen_Emissionsfaktor.png')
plt.show()

Emissionen_Mittelwert_strommix = Mittelwert_f_CO2_mix * Last_Strommix + array_Emissionen_EE_Verbrauch * array_Emissionsfaktor_EE
Emissionen_ges2 = np.sum(Emissionen_Mittelwert_strommix)  #kg
Emissionen_ges2 = round(Emissionen_ges2)    #Emissionen mit gemitteltem CO2 Faktor

list = []              #Emissionen pro Tag
for i in range(0, l, len_month):
    Emissionen_Tag = np.array([])
    for j in range(len_month):
        a = Emissionen_Mittelwert_strommix[i + j]
        Emissionen_Tag = np.append(Emissionen_Tag, a)
    sum = np.sum(Emissionen_Tag)
    list.append(sum)
Emissionen_Tageswerte2 = np.asarray(list)

plt.bar(np.arange(len(Emissionen_Tageswerte2)), Emissionen_Tageswerte2, width=0.2, color = 'orange')
plt.title('Emissionen mit statischem Emissionsfaktor')
plt.ylabel('Emissionen [kg]')
plt.xlabel('Monat')
plt.xticks(np.arange(len(Emissionen_Tageswerte2)), ['Jan', 'Feb', 'März', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
plt.ylim((0, 65000))
plt.savefig('CO2_äquivalente_Emissionen_mit_statischem_Emissionsfaktor.png')
plt.show()


X = Emissionen_Monatswerte_statisch + Emissionen_Gas_theo
X = np.sum(X)
print('Theoretische Emissionen bei Berechnung mit statischem Emissionsfaktor und Gasboiler: ' +str(X))

###Einsparung durch zu viel EE-Erzeugung im Quartier

Einsparung = np.sum(Fall_1) - np.sum(Fall_2)
Einsparung = int(Einsparung)
print('Die Einsparungen durch zu viel EE-Erzeugung beträgt' + str(Einsparung) + ' kg')

##Dynamischer Emissionsfaktor des Quartiers

array_Emissionsfaktor_Quartier = np.array([])
for i in range(l):
    Emissionsfaktor_Quartier = (Emissionen_strommix[i] + Emissionen_EE[i]) / (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i])
    array_Emissionsfaktor_Quartier = np.append(array_Emissionsfaktor_Quartier, Emissionsfaktor_Quartier)

list = []  # Emissionsfaktor des Quartiers pro Monat(siehe range)
for i in range(0, l, len_month):
    Emissionsfaktor_Monat = np.array([])
    for j in range(len_month):
        a = array_Emissionsfaktor_Quartier[i + j]
        Emissionsfaktor_Monat = np.append(Emissionsfaktor_Monat, a)
    mean = np.mean(Emissionsfaktor_Monat)
    list.append(mean)
Emissionsfaktor_Quartier_Monatswerte = np.asarray(list)

plt.bar(np.arange(len(Emissionsfaktor_Quartier_Monatswerte)), Emissionsfaktor_Quartier_Monatswerte, width=0.2, color = 'orange')
plt.title('Monatlicher Emissionsfaktor des Quartiers')
plt.ylabel('Emissionsfaktor [kg/kWh]')
plt.xlabel('Monat')
plt.xticks(np.arange(len(Emissionsfaktor_Quartier_Monatswerte)), ['Jan', 'Feb', 'März', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
plt.ylim((0, 1))
plt.savefig('Emissionsfaktor des Quartiers pro Monat.png')
plt.show()

Verbrauch_min = np.min(array_p_dem)

###PDF Erzeugung
c = canvas.Canvas('Ergebnisse_Systemische_Bewertung.pdf')
c.setFont('Helvetica', 11)
c.line(0.5*cm, 27.5*cm, 20.5*cm, 27.5*cm)
c.drawImage('Logo_EON.png', 14*cm, 28*cm, 176, 30)
c.drawImage('Stromverbrauch_pro_Monat.png', 1*cm, 18*cm, 320, 240)
c.drawImage('CO2_äquivalente_Emissionen_mit_dynamischen_Emissionsfaktor.png', 1*cm, 10*cm, 320, 240)
c.drawString(1*cm, 9*cm, 'Die Emissionen für das gesamte Jahr betragen ' + str(Emissionen_ges) + ' kg')
c.drawString(1*cm, 8*cm, 'Die Emissionen mit statischen Emissionsfaktor und Gaskesseln ' + str(X) + ' kg betragen')
c.drawString(1*cm, 7*cm,'Die Einsparungen durch zu viel EE-Erzeugung beträgt' + str(Einsparung) + ' kg')
c.showPage()
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_Ortsnetzstation.png', 1*cm, 20*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Leistungsaufnahmefähigkeit_Ortsnetzstation.png', 1*cm, 12*cm, 320, 240)
c.drawString(1*cm, 8*cm, 'Die Engpassarbeit an der Ortnetzstation, die durch Erzeugung entsteht, beträgt ' + str(Engpassarbeit_neg) + ' kWh')
#c.drawString(1*cm, 7*cm, 'Engpassgefahr durch Last besteht, wenn Wert negativ ist: ' + str(Engpassgefahr_Last))
#c.drawString(1*cm, 6*cm, 'Engpassgefahr durch Erzeugung besteht, wenn Wert negativ ist: ' + str(Engpassgefahr_Erzeugung))
c.showPage()
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 1.png', 1*cm, 20*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 2.png', 1*cm, 12*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 3.png', 1*cm, 4*cm, 320, 240)
c.showPage()
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 4.png', 1*cm, 20*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 5.png', 1*cm, 12*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 6.png', 1*cm, 4*cm, 320, 240)
c.showPage()
c.drawImage('Vergleich_Durchschnittlicher_bilanzieller_DG_Monat.png', 1*cm, 20*cm, 320, 240)
c.drawImage('Vergleich_Durchschnittlicher_bilanzieller_EV_Monat.png', 1*cm, 12*cm, 320, 240)
c.drawString(1*cm, 29*cm, 'Werte bilanzielle DG: ' + str(array_gamma_DG_monat))
c.drawString(1*cm, 28*cm, 'Werte bilanzielle DG: ' + str(array_gamma_EV_monat))
c.drawString(1*cm, 11*cm, 'Werte durchschnittlicher EV: ' + str(array_gamma_EV_monat_real))
c.drawString(1*cm, 10*cm, 'Werte durchschnittlicher DG: ' + str(array_gamma_DG_monat_real))
c.drawString(1*cm, 9*cm,'Der bilanzielle Deckungsgrad für das gesamte Jahr beträgt ' + str(gamma_DG_Jahr))
c.drawString(1*cm, 8*cm,'Der reale DG für das Jahr beträgt' + str(gamma_DG_Jahr_real))
c.drawString(1*cm, 7*cm,'Der bilanzielle Eigenverbrauch für das gesamte Jahr beträgt ' + str(gamma_EV_Jahr))
c.drawString(1*cm, 6*cm,'Der reale EV für das Jahr beträgt' + str(gamma_EV_Jahr_real))
c.drawString(1*cm, 5*cm,'Der minimale tägliche Deckungsgrad der Monate beträgt' + str(DG_min_monat))
c.drawString(1*cm, 4*cm,'Der maximale tägliche Deckungsgrad der Monate beträgt' + str(DG_max_monat))
c.drawString(1*cm, 3*cm,'Der minimale tägliche Eigenverbrauch der Monate beträgt' + str(EV_min_monat))
c.drawString(1*cm, 2*cm,'Der maximale tägliche Eigenverbrauch der Monate beträgt' + str(EV_min_monat))
c.showPage()
c.drawImage('Autarkie_Monat.png', 1*cm, 20*cm, 320, 240)
c.drawString(1*cm, 18*cm, 'Die Autarkie für das gesamte Jahr Beträgt ' + str(Autarkie_Jahr))
c.drawString(1*cm, 17*cm,'Die geringste tägliche Autarkie des Jahres beträgt ' + str(min_Autarkie))
c.drawString(1*cm, 16*cm,'Die höchste tägliche Autarkie des Jahres beträgt ' + str(max_Autarkie))
c.drawString(1*cm, 10*cm, 'Monatliche Autarkien: ' + str(Autarkie_pro_Monat))
c.showPage()
#c.drawImage('GSC_abs.png', 1*cm, 20*cm, 320, 240)
#c.drawImage('GSC_rel.png', 1*cm, 12*cm, 320, 240)
c.showPage()
c.drawImage('Last_am_Ortsnetztransformator.png', 1*cm, 20*cm, 320, 240)
c.drawString(1*cm, 3*cm, 'Menge der Energie, die das Quartier bezieht(+)/abgibt(-)' + str(sum_Residualenergie) + ' MWh')
c.drawImage('Histogramm_Last_Ortsnetzstation.png', 1*cm, 12*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Gradienten.png', 1*cm, 4*cm, 320, 240)
c.showPage()
c.drawImage('Benötigte_Speicher_pro_Woche.png', 1*cm, 20*cm, 320, 240)
c.drawImage('Benötigter_Speicher_pro_Monat.png', 1*cm, 12*cm, 320, 240)
c.drawString(1*cm, 11*cm, 'Maximaler Speicherbedarf für Tageszyklus, wenn 90% abgedeckt werden sollen ' + str(AA) + ' MWh')
c.drawString(1*cm, 10*cm, 'Maximaler Speicherbedarf für Tageszyklus ' + str(storage_needed_d_max) + ' MWh')
c.drawString(1*cm, 9*cm, 'Maximaler Speicherbedarf für Wochenzyklus ' + str(storage_needed_w_max) + ' MWh')
c.drawString(1*cm, 8*cm, 'Maximaler Speicherbedarf für Wochenzyklus, wenn 90% abgedeckt werden sollen ' + str(storage_needed_w_max_sorted)+ ' MWh')
c.drawString(1*cm, 7*cm, 'Maximaler Speicherbedarf für Monatszyklus, wenn 90% abgedeckt werden sollen ' + str(storage_needed_m_max_sorted) + ' MWh')
c.drawString(1*cm, 6*cm, 'Maximaler Speicherbedarf für Monatszyklus ' + str(storage_needed_m_max)+ ' MWh')
c.drawString(1*cm, 5*cm, 'Die Speicherkapazität für den Betrachtungszeitraum von einem Jahr würde' + str(storage_needed_j) + ' MWh betragen')
c.save()

root.quit()