import numpy as np
import cv2

# Carregue a imagem 'galinha.png'
img = cv2.imread('galinha.png')

# Converta a imagem para tons de cinza
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplique o filtro de Sobel para detectar bordas
sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)

# Converta os resultados para valores absolutos e converta para tipo uint8
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# Combine as imagens de borda usando bitwise_or
sobel = cv2.bitwise_or(sobelX, sobelY)

# Aplique uma operação de dilatação para preencher as regiões com margens
kernel = np.ones((5, 5), np.uint8)
sobel_dilated = cv2.dilate(sobel, kernel, iterations=1)

# Empilhe as imagens para exibição
resultado = np.vstack([
    np.hstack([img_gray, sobel]),
    np.hstack([sobel_dilated, sobel])
])

# Exiba a imagem resultante
cv2.imshow("Sobel com Preenchimento", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
