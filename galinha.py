import numpy as np
import cv2

# Carregue a imagem e converta para escala de cinza
img = cv2.imread('galinha1.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar um filtro de desfoque Gaussiano para reduzir o ruído
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Aplicar o operador Sobel
sobelX = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1)

# Calcule a magnitude do gradiente (Sobel)
sobel_magnitude = np.sqrt(sobelX**2 + sobelY**2)

# Aplicar um threshold na imagem do gradiente
_, thresholded_img = cv2.threshold(sobel_magnitude, 20, 255, cv2.THRESH_BINARY)

# Defina as coordenadas da região de interesse (ROI)
x1, x2, y1, y2 = 100, 300, 150, 350  # Substitua esses valores pelos valores desejados

# Defina a região de interesse (ROI)
roi = thresholded_img[y1:y2, x1:x2]

# Converta a imagem da ROI para o formato CV_8UC1
roi = np.uint8(roi)

# Encontre e conte os contornos na ROI
contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Conte o número de "galinhas" na região
num_galinhas = len(contours)

# Exiba o resultado
print(f'Número de galinhas na região selecionada: {num_galinhas}')

# Exiba a imagem de saída com as regiões de interesse
cv2.imshow("Sobel", thresholded_img)
cv2.waitKey(0)
