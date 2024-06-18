import pyopenjtalk

pl = []
with open('./stn_text.txt', 'r') as f, open('./phone.txt', 'w') as p:
    for line in f:
        pho = pyopenjtalk.g2p(line, kana=False)
        p.write(pho + '\n')
