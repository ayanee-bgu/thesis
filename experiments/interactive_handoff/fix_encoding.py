path = r'c:\Users\ayane\OneDrive\Documents\Masters\Thesis\experiments\interactive_handoff\main.py'
with open(path, 'rb') as f:
    content = f.read()

replacements = [
    (b'\xe2\x80\x94', b' -- '),
    (b'\xe2\x80\x93', b' -- '),
    (b'\xe2\x80\x98', b"'"),
    (b'\xe2\x80\x99', b"'"),
    (b'\xe2\x80\x9c', b'"'),
    (b'\xe2\x80\x9d', b'"'),
    (b'\xc3\xa2\xc2\x80\xc2\x94', b' -- '),
    (b'\xc3\xa2\xc2\x80\xc2\x93', b' -- '),
    (b'\xc3\xa2\xc2\x80\xc2\x9c', b'"'),
    (b'\xc3\xa2\xc2\x80\xc2\x9d', b'"'),
    (b'\xc3\xa2\xc2\x80\xc2\x98', b"'"),
    (b'\xc3\xa2\xc2\x80\xc2\x99', b"'"),
]
for old, new in replacements:
    content = content.replace(old, new)

with open(path, 'wb') as f:
    f.write(content)
print('done')
