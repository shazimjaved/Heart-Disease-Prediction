import cv2
import pytesseract
import numpy as np
import re
import os
from PIL import Image
import io

class OCRProcessor:
    
    def __init__(self):
        try:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        except:
            alternative_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
                'tesseract'
            ]
            
            for path in alternative_paths:
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
                except:
                    continue
    
    def test_ocr(self):
        print("Testing OCR functionality...")
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR is available")
            return True
        except Exception as e:
            print(f"❌ Tesseract OCR error: {e}")
            return False
    
    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text(self, image):
        try:
            processed_image = self.preprocess_image(image)
            
            configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;()[]{}%/- ',
                r'--oem 3 --psm 3',
                r'--oem 3 --psm 4',
                r'--oem 3 --psm 6',
                r'--oem 3 --psm 8'
            ]
            
            best_text = ""
            max_length = 0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config)
                    if len(text.strip()) > max_length:
                        max_length = len(text.strip())
                        best_text = text.strip()
                except:
                    continue
            
            return best_text
            
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""
    
    def parse_health_parameters(self, text):
        parameters = {}
        
        print(f"DEBUG: Extracted text length: {len(text)}")
        print(f"DEBUG: First 200 chars: {text[:200]}...")
        
        patterns = {
            'age': [
                r'(?:age|AGE)[\s:]*(\d{1,3})',
                r'(\d{1,3})\s*(?:years?|yrs?|yo)',
                r'age[\s:]*(\d{1,3})',
                r'(\d{1,3})\s*years?\s*old',
                r'(\d{1,3})\s*yo'
            ],
            'sex': [
                r'(?:sex|gender|SEX|GENDER)[\s:]*([mfMF]|male|female|Male|Female)',
                r'(male|female|Male|Female)',
                r'([mfMF])'
            ],
            'chest_pain': [
                r'(?:chest pain|cp|CP)[\s:]*(\d)',
                r'chest[\s]*pain[\s:]*(\d)',
                r'cp[\s:]*(\d)'
            ],
            'blood_pressure': [
                r'(?:blood pressure|bp|BP|trestbps)[\s:]*(\d{2,3})',
                r'bp[\s:]*(\d{2,3})',
                r'pressure[\s:]*(\d{2,3})',
                r'(\d{2,3})\s*mmhg',
                r'(\d{2,3})\s*mmHg',
                r'systolic[\s:]*(\d{2,3})',
                r'diastolic[\s:]*(\d{2,3})'
            ],
            'cholesterol': [
                r'(?:cholesterol|chol|CHOL)[\s:]*(\d{2,4})',
                r'chol[\s:]*(\d{2,4})',
                r'(\d{2,4})\s*mg/dl',
                r'(\d{2,4})\s*mg/dL',
                r'total[\s]*cholesterol[\s:]*(\d{2,4})'
            ],
            'blood_sugar': [
                r'(?:blood sugar|fbs|FBS|glucose)[\s:]*(\d{2,3})',
                r'glucose[\s:]*(\d{2,3})',
                r'sugar[\s:]*(\d{2,3})',
                r'(\d{2,3})\s*mg/dl',
                r'(\d{2,3})\s*mg/dL',
                r'fasting[\s]*glucose[\s:]*(\d{2,3})'
            ],
            'ecg': [
                r'(?:ecg|ECG|restecg)[\s:]*(\d)',
                r'ecg[\s:]*(\d)',
                r'electrocardiogram[\s:]*(\d)',
                r'resting[\s]*ecg[\s:]*(\d)'
            ],
            'heart_rate': [
                r'(?:heart rate|hr|HR|thalach)[\s:]*(\d{2,3})',
                r'heart[\s]*rate[\s:]*(\d{2,3})',
                r'hr[\s:]*(\d{2,3})',
                r'pulse[\s:]*(\d{2,3})',
                r'(\d{2,3})\s*bpm',
                r'max[\s]*heart[\s]*rate[\s:]*(\d{2,3})'
            ],
            'angina': [
                r'(?:angina|exang|EXANG)[\s:]*([01])',
                r'angina[\s:]*([01])',
                r'exercise[\s]*angina[\s:]*([01])',
                r'exercise[\s]*induced[\s]*angina[\s:]*([01])'
            ],
            'oldpeak': [
                r'(?:oldpeak|ST depression)[\s:]*(\d+\.?\d*)',
                r'st[\s]*depression[\s:]*(\d+\.?\d*)',
                r'oldpeak[\s:]*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*mm'
            ],
            'slope': [
                r'(?:slope|SLOPE)[\s:]*(\d)',
                r'slope[\s:]*(\d)',
                r'st[\s]*slope[\s:]*(\d)'
            ],
            'vessels': [
                r'(?:vessels|ca|CA|major vessels)[\s:]*(\d)',
                r'vessels[\s:]*(\d)',
                r'major[\s]*vessels[\s:]*(\d)',
                r'coronary[\s]*vessels[\s:]*(\d)'
            ],
            'thalassemia': [
                r'(?:thal|thalassemia|THAL)[\s:]*(\d)',
                r'thal[\s:]*(\d)',
                r'thalassemia[\s:]*(\d)'
            ]
        }
        
        for param, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    try:
                        if param == 'sex':
                            if value.lower() in ['m', 'male']:
                                parameters[param] = 1
                            else:
                                parameters[param] = 0
                        else:
                            parameters[param] = float(value)
                        print(f"DEBUG: Found {param} = {value}")
                        break
                    except ValueError:
                        continue
        
        print(f"DEBUG: Total parameters extracted: {len(parameters)}")
        return parameters
    
    def process_image_file(self, uploaded_file):
        try:
            image = Image.open(uploaded_file)
            
            text = self.extract_text(image)
            
            if not text.strip():
                return {
                    'success': False,
                    'extracted_text': '',
                    'parameters': {},
                    'message': "No text could be extracted from the image. Please try with a clearer image."
                }
            
            parameters = self.parse_health_parameters(text)
            
            if not parameters:
                return {
                    'success': False,
                    'extracted_text': text,
                    'parameters': {},
                    'message': "No parameters could be extracted from the image. Please try with a clearer image or use manual input."
                }
            
            return {
                'success': True,
                'extracted_text': text,
                'parameters': parameters,
                'message': f"Successfully extracted {len(parameters)} parameters"
            }
            
        except Exception as e:
            return {
                'success': False,
                'extracted_text': '',
                'parameters': {},
                'message': f"Error processing image: {str(e)}"
            }

NORMAL_RANGES = {
    'age': (18, 100),
    'sex': (0, 1),
    'chest_pain': (0, 3),
    'blood_pressure': (90, 200),
    'cholesterol': (100, 400),
    'blood_sugar': (70, 200),
    'ecg': (0, 2),
    'heart_rate': (60, 200),
    'angina': (0, 1),
    'oldpeak': (0, 6),
    'slope': (0, 2),
    'vessels': (0, 3),
    'thalassemia': (0, 3)
}

def validate_parameters(parameters):
    validated = {}
    warnings = []
    
    for param, value in parameters.items():
        if param in NORMAL_RANGES:
            min_val, max_val = NORMAL_RANGES[param]
            if min_val <= value <= max_val:
                validated[param] = value
            else:
                warnings.append(f"{param}: {value} is outside normal range ({min_val}-{max_val})")
                validated[param] = value
        else:
            validated[param] = value
    
    return validated, warnings