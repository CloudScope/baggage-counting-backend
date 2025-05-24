# edge_device_app/anpr_module.py
import cv2
import pytesseract # Make sure Tesseract OCR is installed on the system
import re

def preprocess_for_ocr(image_roi):
    """Basic preprocessing for OCR."""
    if image_roi is None or image_roi.size == 0:
        return None
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    # Further preprocessing like thresholding, blurring can be added here
    # Example:
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # return thresh
    return gray

def extract_plate_number(frame):
    """
    Highly simplified ANPR. In a real system, this needs:
    1. Plate detection (e.g., using a Haar cascade, YOLO, or other detector).
    2. ROI extraction and perspective correction.
    3. Robust OCR.
    This example assumes the plate is somewhat clear in the frame or a known ROI.
    """
    if frame is None:
        return None

    # For POC, let's assume we OCR a central region or the whole frame
    # h, w, _ = frame.shape
    # roi = frame[h//3 : 2*h//3, w//4 : 3*w//4] # Example ROI
    roi = frame # Process whole frame for simplicity in this stub

    processed_roi = preprocess_for_ocr(roi)
    if processed_roi is None:
        return None

    # Tesseract OCR configuration
    # Adjust whitelist and psm based on expected plate characters and layout
    # psm 6: Assume a single uniform block of text.
    # psm 7: Treat the image as a single text line.
    # psm 11: Sparse text. Find as much text as possible in no particular order.
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        text = pytesseract.image_to_string(processed_roi, config=custom_config, timeout=2) # Added timeout
    except RuntimeError as e: # Catches Tesseract call errors (e.g. timeout)
        print(f"ANPR OCR Error: {e}")
        return None
    except Exception as e: # Catch any other pytesseract error
        print(f"ANPR Pytesseract unknown error: {e}")
        return None


    # Clean and validate extracted text (very basic)
    plate_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())

    # Basic validation (e.g., length based on common plate formats)
    # This needs to be adapted to your local license plate formats.
    if 3 <= len(plate_text) <= 10: # Example length check
        # print(f"ANPR: Potential plate detected: {plate_text} (Raw OCR: '{text.strip()}')")
        return plate_text
    else:
        # if text.strip(): print(f"ANPR: OCR'd text '{text.strip()}' -> '{plate_text}' did not pass validation.")
        return None

if __name__ == '__main__':
    # Create a dummy image for testing
    import numpy as np
    dummy_plate_img = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(dummy_plate_img, "AB12CD345", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    # cv2.imshow("Dummy Plate", dummy_plate_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    plate = extract_plate_number(dummy_plate_img)
    if plate:
        print(f"Extracted Plate from dummy: {plate}")
    else:
        print("No plate extracted from dummy.")