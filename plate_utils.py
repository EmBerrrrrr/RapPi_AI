import re

def normalize_plate(text):
    if not text:
        return None

    text = text.upper().replace(" ", "").replace(".", "")
    text = re.sub(r'[^A-Z0-9]', '', text)

    if len(text) < 7:
        return None

    head = list(text[:4])
    tail = text[4:]

    def fix_to_letter(c):
        mapping = {
            '8': 'B',
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '2': 'Z',
            '6': 'G',
            '9': 'Q'
        }
        return mapping.get(c, c)

    def fix_to_digit(c):
        mapping = {
            'B': '8',
            'O': '0',
            'I': '1',
            'L': '1',
            'S': '5',
            'Z': '2',
            'G': '6',
            'Q': '9'
        }
        return mapping.get(c, c)

    if not (head[0].isdigit() and head[1].isdigit()):
        return None

    head[2] = fix_to_letter(head[2])
    if not head[2].isalpha():
        return None

    if head[3].isdigit():
        pass
    else:
        head[3] = fix_to_letter(head[3])
        if not (head[3].isalpha() or head[3].isdigit()):
            return None

    head = "".join(head)

    tail_fixed = ""
    for c in tail:
        c = fix_to_digit(c)
        if not c.isdigit():
            return None
        tail_fixed += c

    return f"{head}-{tail_fixed}"