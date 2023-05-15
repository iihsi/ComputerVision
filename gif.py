from PIL import Image
pict=[]
img=Image.open('kaz.jpg')
pict.append(img)
for i in range(9):
    pic_name='cou-00' +str(i+1)+ '.jpg'
    img = Image.open(pic_name)
    pict.append(img)
for j in range(10, 100):
    pic_name='cou-0' +str(j)+ '.jpg'
    img = Image.open(pic_name)
    pict.append(img)
img=Image.open('kar.jpg')
pict.append(img)
pict[0].save('cou.gif', save_all=True, append_images=pict[1:],
optimize=False, duration=100, loop=0)