import cv2
import numpy as np

# Passo 1: Carregue a imagem original
imagem_original = cv2.imread('bocha.jpg')

# Passo 2: Converta a imagem para tons de cinza
imagem_gray = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

# Passo 3: Use a transformada de Hough para detectar círculos na imagem
circulos = cv2.HoughCircles(imagem_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                            param1=50, param2=30, minRadius=10, maxRadius=50)

# Passo 4: Crie uma imagem binarizada preta do mesmo tamanho da imagem original
imagem_binarizada = np.zeros_like(imagem_gray)

# Passo 5: Se círculos forem encontrados, desenhe-os na imagem binarizada como branco
if circulos is not None:
    circulos = np.uint16(np.around(circulos))
    for circulo in circulos[0, :]:
        centro = (circulo[0], circulo[1])
        raio = circulo[2]
        
        # Passo 6: Desenhe o círculo branco na imagem binarizada
        cv2.circle(imagem_binarizada, centro, raio, 255, -1)  # Preencha o círculo com branco

# Passo 7: Salve a imagem binarizada
cv2.imwrite('imagem_binarizada.jpg', imagem_binarizada)
