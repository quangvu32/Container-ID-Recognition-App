import pytesseract

def error_check(container_dict):
    # Character-level correction

    fixed_container_dict = {}
    try:
        for c in container_dict:
            fixed_code = list(container_dict[c])
            if c == 'size':
                for i in range(4):
                    if i == 2:
                        fixed_code[i] = 'G'
                    else:
                        if fixed_code[i] == 'I' or fixed_code[i] == 'T':
                            fixed_code[i] = '1'
        
            elif c == 'company':
                for i in range(4):
                    if fixed_code[i] == '1' or fixed_code[i] == '7':
                        fixed_code[i] = 'I'
                    if fixed_code[i] == '4':
                        fixed_code[i] = 'A'
                    if fixed_code[i] == '6':
                        fixed_code[i] = 'G'
                    if fixed_code[i] == '8':
                        fixed_code[i] = 'B'
        
                if fixed_code[3] == 'Y':
                    fixed_code[3] = 'U'
        
            else:
                if len(fixed_code) > 7:
                    fixed_code = fixed_code[:6] + fixed_code[-1:]
                for i in range(7):
                    if fixed_code[i] == 'I':
                        fixed_code[i] = '1'
                    if fixed_code[i] == 'A':
                        fixed_code[i] = '4'
                    if fixed_code[i] == 'G':
                        fixed_code[i] = '6'
                    if fixed_code[i] == 'B':
                        fixed_code[i] = '8'
                
            fixed_container_dict[c] = ''.join(fixed_code)
        return fixed_container_dict
    except IndexError:
        return container_dict     


def run_tesseract_on_image(img, psm: int = 7, oem: int = 3, whitelist: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """Run Tesseract with sensible options for single-line alphanumeric.
    psm 7 = treat image as a single text line. psm 8 = single word. Try 7.
    """
    config = f'-c tessedit_char_whitelist={whitelist} --psm {psm} --oem {oem}'
    text = pytesseract.image_to_string(img, config=config)
    text = text.strip()
    text = text.replace('\n','')
    text = ''.join([c for c in text if c.isalnum()])
    return text