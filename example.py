from PIL import Image
import numpy

path='C:/Users/Rafik/Pictures/'
filename='pomme.jpeg'

#Read image
image = Image.open( path+filename )
#Read pixels
pixels = image.load()

# image size
size = image.size
print("Taille de l'image (largeur,heuteur): "+str(size))

data=[]
for x in range(size[0]):     
    for y in range(size[1]):
        data.append(pixels[x, y][1])  # pour Histogramme
        R =  pixels[x, y][0] # Couleur Rouge
        V =  pixels[x, y][1] # Couleur Verte
        B = pixels[x, y][2] # Couleur Bleu
        pixels[x, y] = (B,V,R)
def create_image_gray(image):
    pixels_gray = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            R =  pixels_gray[x, y][0] # Couleur Rouge
            V =  pixels_gray[x, y][1] # Couleur Verte
            B = pixels_gray[x, y][2] # Couleur Bleu
            pixels_gray[x, y] = ((R+B+V)//3,(R+B+V)//3,(R+B+V)//3)
    image.show()

create_image_gray(image)
# Affiche l'image
image.show()
#  Enregictre l'image
image.save(path +'Save.png', "PNG")






# Ouvre dans un tableau 3D l'image [Ligne][Colonne][Couleur]
pix = numpy.array(image)

# Pixel Rouge(0) en position 0,0
print(pix[0][0][0])
print(pixels[0,0][0])

# Création d'une nouvelle image et l'affiche
newimage = Image.fromarray(pix)
newimage.show()





# Affiche un Histogramme
import matplotlib.pyplot as plt
plt.hist(data,range=(0,255),bins=256)
plt.show()