# Detectando Objetos em tempo real com OpenCV + Deep Learning

# Importação das bibliotecas
import numpy as np              # Biblioteca para cálculos numéricos e manipulação de arrays
import cv2                      # OpenCV, usado para câmera, imagens e rede neural
import time                     # Biblioteca para medir o tempo de processamento

# Necessário para criar o arquivo contendo os objetos detectados
from csv import DictWriter      # Permite escrever dados em um arquivo CSV no formato de dicionário

# Defini qual câmera será utilizada na captura
camera = cv2.VideoCapture(0)    # Abre a câmera do computador (0 geralmente é a câmera padrão)

# Cria variáveis para captura de altura e largura
h, w = None, None               # Ainda não sabemos o tamanho do frame; isso será definido depois

# Carrega o arquivo com o nome dos objetos que o modelo foi treinado para detectar
with open('yoloDados/YoloNames.names') as f:
    # Lê cada linha do arquivo e remove espaços/quebras de linha
    # Cada linha representa uma classe detectável pelo YOLO
    labels = [line.strip() for line in f]

# Carrega os arquivos treinados pelo framework YOLO
network = cv2.dnn.readNetFromDarknet(
    'yoloDados/yolov3.cfg',     # Arquivo de configuração da rede
    'yoloDados/yolov3.weights'  # Arquivo com os pesos treinados
)

# Captura uma lista com todos os nomes de camadas da rede
layers_names_all = network.getLayerNames()

# Obtém apenas os nomes das camadas de saída da rede
# getUnconnectedOutLayers() retorna os índices das camadas de saída
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

# Define a probabilidade mínima para aceitar uma detecção
probability_minimum = 0.5     # Só aceita detecções com confiança maior que 50%

# Define o limite para filtrar caixas delimitadoras repetidas
threshold = 0.3               # Usado na Non-Maximum Suppression (NMS)

# Gera cores aleatórias para desenhar caixas de cada objeto detectado
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Loop de captura e detecção dos objetos
with open('teste.csv', 'w') as arquivo:
    # Define os nomes das colunas do CSV
    cabecalho = ['Detectado', 'Acuracia']

    # Cria o escritor do CSV
    escritor_csv = DictWriter(arquivo, fieldnames=cabecalho)

    # Escreve o cabeçalho no arquivo CSV
    escritor_csv.writeheader()

    # Loop infinito para processar os frames da câmera
    while True:
        # Captura da câmera frame por frame
        ret, frame = camera.read()   # ret = se capturou com sucesso; frame = imagem capturada

        # Verifica se a câmera realmente retornou uma imagem
        if not ret:
            print('Erro ao capturar frame da câmera')
            break

        # Define a altura e largura do frame apenas na primeira leitura
        if w is None or h is None:
            # frame.shape[:2] retorna (altura, largura)
            h, w = frame.shape[:2]

        # Converte a imagem em blob, que é o formato esperado pela rede neural
        # 1/255.0 normaliza os pixels entre 0 e 1
        # (416, 416) é o tamanho esperado pelo YOLOv3
        # swapRB=True troca BGR para RGB
        # crop=False evita cortar a imagem
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)

        # Define o blob como entrada da rede neural
        network.setInput(blob)

        # Marca o tempo de início do processamento
        start = time.time()

        # Executa a passagem direta na rede apenas nas camadas de saída
        output_from_network = network.forward(layers_names_output)

        # Marca o tempo de fim do processamento
        end = time.time()

        # Mostra o tempo gasto para processar o frame atual
        print('Tempo gasto atual {:.5f} segundos'.format(end - start))

        # Listas para guardar os resultados das detecções
        bounding_boxes = []   # Caixas delimitadoras
        confidences = []      # Confiança de cada detecção
        class_numbers = []    # Índice da classe detectada

        # Percorre todas as saídas da rede
        for result in output_from_network:
            # Percorre cada detecção encontrada naquela saída
            for detected_objects in result:
                # As 5 primeiras posições são:
                # [center_x, center_y, width, height, objectness_score]
                # A partir da posição 5 estão as probabilidades das classes
                scores = detected_objects[5:]

                # Pega o índice da classe com maior probabilidade
                class_current = np.argmax(scores)

                # Pega a confiança da classe escolhida
                confidence_current = scores[class_current]

                # Filtra previsões fracas
                if confidence_current > probability_minimum:
                    # Converte as coordenadas relativas da rede para pixels
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Separa as coordenadas e dimensões da caixa
                    x_center, y_center, box_width, box_height = box_current

                    # Calcula a posição do canto superior esquerdo da caixa
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adiciona a caixa delimitadora na lista
                    bounding_boxes.append([x_min, y_min,
                                           int(box_width), int(box_height)])

                    # Adiciona a confiança na lista
                    confidences.append(float(confidence_current))

                    # Adiciona o índice da classe na lista
                    class_numbers.append(class_current)

        # Remove caixas repetidas ou sobrepostas usando NMS
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)

        # Verifica se existe pelo menos um objeto detectado
        if len(results) > 0:
            # Percorre as detecções que sobraram após a NMS
            for i in results.flatten():
                # Recupera a posição da caixa
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Pega a cor da classe detectada
                colour_box_current = colours[class_numbers[i]].tolist()

                # Desenha o retângulo na imagem
                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                # Monta o texto com o nome da classe e a confiança
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                       confidences[i])

                # Escreve o nome do objeto na imagem
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                # Salva a detecção no CSV
                escritor_csv.writerow({
                    "Detectado": text_box_current.split(':')[0],
                    "Acuracia": text_box_current.split(':')[1]
                })

                # Mostra no terminal o objeto detectado e sua confiança
                print(text_box_current.split(':')[0] + " - " + text_box_current.split(':')[1])

        # Cria uma janela redimensionável com o nome indicado
        cv2.namedWindow('YOLO v3 WebCamera', cv2.WINDOW_NORMAL)

        # Mostra o frame com as caixas desenhadas
        cv2.imshow('YOLO v3 WebCamera', frame)

        # Se apertar a tecla 'q', encerra o loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera a câmera
camera.release()

# Fecha todas as janelas abertas pelo OpenCV
cv2.destroyAllWindows()