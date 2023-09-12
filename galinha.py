import cv2
import numpy as np



# Carrega as imagens
imagem_galinha = cv2.imread('galinha.png')
imagem_galinhafade = cv2.imread('galinhafade.png', -1)  # Carrega a imagem com o canal de transparência

# Redimensiona a imagem da máscara para o tamanho da imagem da galinha
imagem_galinhafade = cv2.resize(imagem_galinhafade, (imagem_galinha.shape[1], imagem_galinha.shape[0]))

# Separa o canal de opacidade da imagem da máscara
canal_alpha = imagem_galinhafade[:, :, 3]

# Remove o canal de opacidade da imagem da máscara
imagem_galinhafade = imagem_galinhafade[:, :, 0:3]

# Aplica a máscara com opacidade na imagem da galinha
imagem_resultante = cv2.addWeighted(imagem_galinha, 0.5, imagem_galinhafade, 2, 0)

# Converte a imagem resultante em escala de cinza
imagem_cinza = cv2.cvtColor(imagem_resultante, cv2.COLOR_BGR2GRAY)

# Cria uma máscara radial gaussiana invertida
altura, largura = imagem_cinza.shape
y, x = np.ogrid[-altura/2:altura/2, -largura/2:largura/2]
raio = np.sqrt(x**2 + y**2)
sigma = largura / 3.0  # Ajuste o valor de sigma conforme necessário para controlar o efeito
mascara_radial_invertida = 1.33 - np.exp(-raio**2 / (2 * sigma**2))

# Ajusta o escurecimento tornando a máscara mais opaca
fator_escurecimento = 2.5  # Ajuste conforme necessário
mascara_ajustada = mascara_radial_invertida * fator_escurecimento

# Aplica a máscara radial ajustada à imagem em tons de cinza
imagem_ajustada = imagem_cinza * mascara_ajustada

# Converte a imagem ajustada para o tipo CV_8U
imagem_ajustada = cv2.convertScaleAbs(imagem_ajustada)

# Aplica o operador Top-Hat
kernel = np.ones((250, 250), np.uint8)
imagem_tophat = cv2.morphologyEx(imagem_ajustada, cv2.MORPH_TOPHAT, kernel)

# Normaliza a imagem para aumentar o contraste
imagem_tophat_normalizada = cv2.normalize(imagem_tophat, None, 0, 255, cv2.NORM_MINMAX)

# Cria uma lista vazia para rastrear as posições das bolas brancas
posicoes_anteriores = []

# Cria bolas brancas nos locais apropriados
threshold = 200  # Valor de intensidade para considerar um pixel como claro
imagem_preta = np.zeros_like(imagem_tophat_normalizada)
# Função para criar uma bola branca no local de um pixel claro
def criar_bola_branca(imagem_tophat_normalizada, threshold, posicoes_anteriores):
    altura, largura = imagem_tophat_normalizada.shape
    cor_branca = (255, 255, 255)  # Cor branca
    raio_bola = 10
    threshold = 70
    for y in range(altura):
        for x in range(largura):
            if imagem_tophat_normalizada[y, x] >= threshold:  # Verifica se o pixel é claro o suficiente
                # Verifica se a posição já foi usada
                posicao_atual = (x, y)
                criar = True
                for posicao_anterior in posicoes_anteriores:
                    if np.linalg.norm(np.array(posicao_atual) - np.array(posicao_anterior)) < raio_bola + 25:
                        criar = False
                        break

                if criar:
                    cv2.circle(imagem_preta, (x, y), raio_bola, cor_branca, -1)  # Cria uma bola branca no local
                    posicoes_anteriores.append(posicao_atual)
                    

criar_bola_branca(imagem_tophat_normalizada, threshold, posicoes_anteriores)


# Exibe a imagem com as bolas brancas
cv2.imshow('Imagem com Bolas Brancas', imagem_tophat_normalizada)
cv2.imshow('Imagem com Bolas preta', imagem_preta)
cv2.waitKey(0)
cv2.destroyAllWindows()
