from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def home(request):
    return render(request, 'home.html')


def predictDiabetes(request):
    return render(request, 'diabetes.html')


def predictHeartDisease(request):
    return render(request, 'heartDisease.html')


def predictParkinson(request):
    return render(request, 'parkinson.html')


def diabetesResult(request):
    data = pd.read_csv(r'C:\Users\manuk\multipleDiseasePrediction\Diabetes\diabetes.csv')
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=12)
    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    predictions = model.predict(xtest)
    accuracy = accuracy_score(predictions, ytest)
    print(accuracy)

    csvVals = request.GET['csvVals']
    if len(csvVals) > 0:
        csvInput = np.fromstring(csvVals, dtype=float, sep=',')
        reshapedData = csvInput.reshape(1, -1)
    else:
        val1 = float(request.GET['n1'])
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val4 = float(request.GET['n4'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = float(request.GET['n8'])

        inputData = (val1, val2, val3, val4, val5, val6, val7, val8)
        npData = np.asarray(inputData)
        reshapedData = npData.reshape(1, -1)

    prediction = model.predict(reshapedData)

    if prediction == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'diabetes.html', {"result2": result1})


def heartDiseaseResult(request):
    data = pd.read_csv(r'C:\Users\manuk\multipleDiseasePrediction\heartDisease\heartNew.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)

    predictions = model.predict(xtest)
    accuracy = accuracy_score(predictions, ytest)
    print(accuracy)

    csvVals = request.GET['csvVals']
    if len(csvVals) > 0:
        csvInput = np.fromstring(csvVals, dtype=float, sep=',')
        reshapedData = csvInput.reshape(1, -1)
    else:
        val1 = float(request.GET['n1'])
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val4 = float(request.GET['n4'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = float(request.GET['n8'])
        val9 = float(request.GET['n9'])
        val10 = float(request.GET['n10'])
        val11 = float(request.GET['n11'])
        val12 = float(request.GET['n12'])
        val13 = float(request.GET['n13'])

        inputData = (val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13)
        npData = np.asarray(inputData)
        reshapedData = npData.reshape(1, -1)

    input_data = scaler.transform(reshapedData)

    prediction = model.predict(input_data)

    if prediction == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'heartDisease.html', {"result2": result1})


def parkinsonResult(request):
    data = pd.read_csv(r'C:\Users\manuk\multipleDiseasePrediction\parkinson\parkinsons.csv')
    X = data.drop(columns=['name', 'status'], axis=1)
    y = data['status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier()
    model.fit(xtrain, ytrain)

    predictions = model.predict(xtest)
    accuracy = accuracy_score(predictions, ytest)
    print(accuracy)

    csvVals = request.GET['csvVals']
    if len(csvVals) > 0:
        csvInput = np.fromstring(csvVals, dtype=float, sep=',')
        reshapedData = csvInput.reshape(1, -1)
    else:
        val1 = float(request.GET['n1'])
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val4 = float(request.GET['n4'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = float(request.GET['n8'])
        val9 = float(request.GET['n9'])
        val10 = float(request.GET['n10'])
        val11 = float(request.GET['n11'])
        val12 = float(request.GET['n12'])
        val13 = float(request.GET['n13'])
        val14 = float(request.GET['n14'])
        val15 = float(request.GET['n15'])
        val16 = float(request.GET['n16'])
        val17 = float(request.GET['n17'])
        val18 = float(request.GET['n18'])
        val19 = float(request.GET['n19'])
        val20 = float(request.GET['n20'])
        val21 = float(request.GET['n21'])
        val22 = float(request.GET['n22'])

        inputData = (val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13,
                     val14, val15, val16, val17, val18, val19, val20, val21, val22)
        npData = np.asarray(inputData)
        reshapedData = npData.reshape(1, -1)

    input_data = scaler.transform(reshapedData)
    print(input_data)

    prediction = model.predict(input_data)

    if prediction == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'parkinson.html', {"result2": result1})
