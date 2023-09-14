import cv2
import pytesseract
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {
    "AN": "Andaman and Nicobar",
    "AP": "Andhra Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chattisgarh",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD/OR": "Odisha",
    "PY": "Pondicherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "CH": "Chandigarh",
    "DN": "Dadar & Nagar Haveli",
    "LA": "Ladakh"
}


def extract_num(img_path):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    nplate = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize =(25, 25))
    for (x,y,w,h) in nplate:
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, :]
        
        # Preprocess the license plate image
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate = cv2.GaussianBlur(plate, (5, 5), 0)
        plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 

        read = pytesseract.image_to_string(plate).strip()
        print( "Read:", read)

        # Clean the recognized text to keep only alphanumeric characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', read).upper()
        print("Cleaned Text:", cleaned_text)
        

        #Try to extract the state code from the text

        state_code = None
        if cleaned_text[:2] in states:
            state_code = cleaned_text[:2]

        if state_code: 
            print('Car Belongs to', states[state_code])
        else:
            print('State not recognized!!')
        
        print( "License Plate:", cleaned_text)
        cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (51,51,255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Plate', plate)

    cv2.imshow("Result", img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)  

extract_num('./test_images/test1.jpg')      



            
