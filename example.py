from PIL import Image,ImageFilter
import numpy 
import math
path='C:/Users/Rafik/Pictures/'
filename='captureRamazan.png'

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
            G = (R+B+V)//3
            pixels_gray[x, y] = (G,G,G)
    image.show()
    return image
def transform_image_to_binary(image,seuil):
    img = numpy.array(image)
    binarr = numpy.where(img > seuil, 255, 0).astype(numpy.uint8)
    binimg = Image.fromarray(binarr)
    binimg.show()
def transform_image_to_negatif(image):
    pixels_neg = image.load()
    negatif = 255
    for x in range(size[0]):
        for y in range(size[1]):
            R =  pixels_neg[x, y][0] # Couleur Rouge
            V =  pixels_neg[x, y][1] # Couleur Verte
            B = pixels_neg[x, y][2] # Couleur Bleu
            pixels_neg[x, y] = (negatif-R,negatif-V,negatif-B)
    image.show()

def transform_image_to_contraste(image):
    img = numpy.array(image)
    max = img.max()
    min = img.min()
    pixels_gray = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            R =  pixels_gray[x, y][0] # Couleur Rouge
            V =  pixels_gray[x, y][1] # Couleur Verte
            B = pixels_gray[x, y][2] # Couleur Bleu
            G = (R+B+V)//3
            gray_c =   (G -min)//( max -min ) * 255
            pixels_gray[x, y] = (gray_c,gray_c,gray_c)
    image.show()
def transform_image_to_contraste_coloree(image):
    img = numpy.array(image)
    max = img.max()
    min = img.min()
    pixels_gray = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            R =  pixels_gray[x, y][0] # Couleur Rouge
            V =  pixels_gray[x, y][1] # Couleur Verte
            B = pixels_gray[x, y][2] # Couleur Bleu
            pixels_gray[x, y] = ((R -min)//( max -min ) * 255,(V -min)//( max -min ) * 255,(B -min)//( max -min ) * 255)
    image.show()

def floute_image_uniforme(img, radius):
    image = img.load()
    height, width = img.height, img.width
    blurred_image = image

    for y in range(height):
        for x in range(width):
            sum_r = sum_g = sum_b = count = 0

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        r, g, b = image[ny, nx]
                        sum_r += r
                        sum_g += g
                        sum_b += b
                        count += 1

            blurred_image[y, x] = (sum_r // count, sum_g // count, sum_b // count)
    img.show()


def floute_image_non_uniforme(img, radius):
    image = img.load()
    height, width = img.height, img.width
    blurred_image = image

    kernel_size = 2 * radius + 1
    noyau = numpy.zeros((kernel_size, kernel_size))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            noyau[dy+ radius,dx+ radius] = math.exp(-2 * (dx**2 + dy**2))
    noyau = noyau / noyau.sum()


def floute_image_non_uniforme(img, radius):
    image = img.load()
    image_array = numpy.array(img)
    height, width,channels = image_array.shape
    blurred_image = numpy.zeros((height, width, channels), dtype=float)

    kernel_size = 2 * radius + 1
    noyau = numpy.zeros((kernel_size, kernel_size))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            noyau[dy + radius, dx + radius] = math.exp(-2 * (dx**2 + dy**2))
    noyau /= noyau.sum()

    for y in range(height):
        for x in range(width):
            weighted_sum = numpy.zeros(3)

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        weight = noyau[dy + radius, dx + radius]
                        pixel = numpy.array(image[ny, nx], dtype=float)
                        weighted_sum += pixel * weight

            blurred_image[y, x] = numpy.clip(weighted_sum, 0, 255)
    img.show()

def ajouter_bruit(image, amplitude, pourcentage):

    image_bruitee =  image.copy()
    
    height, width, channels =image.shape
    
    total_pixels = height * width
    
    pixels_a_bruiter = int(total_pixels * pourcentage / 100)
    
    indices_x = numpy.random.randint(0, width, size=pixels_a_bruiter)
    indices_y = numpy.random.randint(0, height, size=pixels_a_bruiter)
    
    for x, y in zip(indices_x, indices_y):
        bruit = numpy.random.randint(-amplitude, amplitude + 1, size=channels)
        image_bruitee[y, x] = numpy.clip(image[y, x] + bruit, 0, 255)  
    return image_bruitee


def image_debruitee(image, taille_filtre=3):
    height, width = image.shape
    
    rayon = taille_filtre // 2
    
    image_debruitee = numpy.zeros_like(image, dtype=float)
    
    for y in range(height):
        for x in range(width):
            somme = numpy.zeros(3, dtype=float)
            compteur = 0
            
            for dy in range(-rayon, rayon + 1):
                for dx in range(-rayon, rayon + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        somme += image[ny, nx]
                        compteur += 1
            
            image_debruitee[y, x] = somme / compteur
    
    return numpy.clip(image_debruitee, 0, 255).astype(numpy.uint8)

image_bruitee_array = numpy.array(image)

taille_filtre = 3
image_debruitee_array = image_debruitee(image_bruitee_array, taille_filtre)

image_debruitee = Image.fromarray(image_debruitee_array)
image_debruitee.show()



# create_image_gray(image)
# transform_image_to_negatif(image)
# transform_image_to_binary(create_image_gray(image),130)
# image.save(path +'Save.png', "PNG")
# image_array = numpy.array(image)

# image_bruitee_array = ajouter_bruit(image_array,90,10)
# image_bruitee = Image.fromarray(image_bruitee_array.astype(numpy.uint8))
# image_bruitee.show()



# Ouvre dans un tableau 3D l'image [Ligne][Colonne][Couleur]
pix = numpy.array(image)

# Pixel Rouge(0) en position 0,0
print(pix[0][0][0])
print(pixels[0,0][0])

# Création d'une nouvelle image et l'affiche
# newimage = Image.fromarray(pix)
# newimage.show()





# Affiche un Histogramme
import matplotlib.pyplot as plt
plt.hist(data,range=(0,255),bins=256)
plt.show()