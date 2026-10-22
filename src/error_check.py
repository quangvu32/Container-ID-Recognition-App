def error_check(container_dict):
    # Define mapping dictionaries for character correction
    num_to_letter = {'1': 'I', '7': 'I', '4': 'A', '6': 'G', '8': 'B'}
    letter_to_num = {'I': '1', 'T': '1', 'A': '4', 'G': '6', 'B': '8'}
    
    fixed_container_dict = {}
    
    try:
        for c, text in container_dict.items():
            if not text:
                fixed_container_dict[c] = text
                continue
                
            fixed_code = list(text)
            length = len(fixed_code)
            
            if c == 'size':
                for i in range(min(4, length)):
                    if i == 2:
                        fixed_code[i] = num_to_letter.get(fixed_code[i], fixed_code[i])
                    else:
                        fixed_code[i] = letter_to_num.get(fixed_code[i], fixed_code[i])
            
            elif c == 'company':
                for i in range(min(4, length)):
                    fixed_code[i] = num_to_letter.get(fixed_code[i], fixed_code[i])
                if length > 3 and fixed_code[3] == 'Y':
                    fixed_code[3] = 'U'
            
            else:
                if length > 7:
                    fixed_code = fixed_code[:6] + fixed_code[-1:]
                    length = len(fixed_code)
                for i in range(min(7, length)):
                    fixed_code[i] = letter_to_num.get(fixed_code[i], fixed_code[i])
            
            fixed_container_dict[c] = ''.join(fixed_code)
            
        return fixed_container_dict
        
    except Exception:
        # Fallback to original dictionary if any unexpected error occurs
        return container_dict