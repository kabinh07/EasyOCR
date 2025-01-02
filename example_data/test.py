from easyocr.easyocr import Reader

reader = Reader(["bn"])
print(reader.readtext('144.png'))