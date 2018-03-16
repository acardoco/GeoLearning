from PIL import Image

'redimensión de imagen a probar'
basewidth = 40
img = Image.open('piscina3.png')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('piscina3.jpg')



