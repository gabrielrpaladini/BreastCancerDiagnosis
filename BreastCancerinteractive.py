import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Gabriel Paladini\Desktop\projetos\Breast cancer\data.csv')
del df['id']
del df['Unnamed: 32']
x = df.drop('diagnosis', axis=1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(x_train, y_train)
rfc_prd = rfc.predict(x_test)

#Interactive way

a1 = int(input('If you want to predict digit 1, if you want to see some metrcis digit 2 '))

if a1 == 1:
    print('Say more about the data: ')
    rm = int(input('radius mean: 	'))
    tm = int(input('texture mean:   '))
    pm = int(input('perimeter mean: '))
    am = int(input('area mean:  '))
    sm = int(input('smoothness mean:    '))
    cm = int(input('compactness mean:   '))
    cm1 = int(input('concavity mean: '))
    cm2 = int(input('concave points mean: '))
    tw = int(input('texture worst:  '))
    pw = int(input('perimeter worst: '))
    aw = int(input('area worst: '))
    sm2 = int(input('smoothness worst:	'))
    cw = int(input('compactness worst: '))
    cw2 = int(input('concavity worst:   '))
    cpw = int(input('concave points worst:	'))
    syw = int(input('symmetry worst: '))
    fdw = int(input('fractal dimension worst: '))

    p1 = rfc.predict([[rm, tm, pm, am, sm, cm, cm1, cm2, tw, pw, aw, sm2, cw, cw2, cpw, syw, fdw]])
    print('The diagnosis is {}'.format(p1))
elif a1 == 2:
    from sklearn.metrics import classification_report, confusion_matrix

    print('Some metrics: \n')
    print(classification_report(y_test, rfc_prd))
    print('\n')
    print(confusion_matrix(y_test, rfc_prd))