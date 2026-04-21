import cv2
import numpy as np

cap = cv2.VideoCapture(0) #iniciar câmera, 0 significa que é uma câmera integrada à maquina.

while True:
    ret, frame = cap.read() #ret é um boleano que indica se o frame foi capturado com sucesso ou não.

    if not ret: #se o frame foi capturado com sucesso, então...
        print('Câmera não encontrada')
        False
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #cvtcolor altera a cor dos frames

    cv2.imshow('Gravando', frame) #tela que aparece o vídeo da gravação

    if cv2.waitKey(1) & 0xFF == ord('x'): #condicional para parar de "gravar", clicando no 'x'
        break

    if cv2.getWindowProperty("Câmera de Cinto com IA", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release() #encerra o uso da câmera no código
cv2.destroyAllWindows() #fecha todas janelas abertas pelo código