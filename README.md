# XADREZIA

### XADREZIA é um projeto experimental para desenvolver uma engine de xadrez baseada exclusivamente em reconhecimento de padrões com redes neurais. Ao contrário de engines tradicionais que utilizam busca em árvore (como Minimax ou Monte Carlo Tree Search), XADREZIA emprega redes neurais para avaliar posições e selecionar movimentos, explorando o potencial do aprendizado de máquina no xadrez. O projeto está em fase inicial e é voltado para pesquisa e experimentação.

## Objetivo

Construir uma engine de xadrez que aprende a jogar por meio de uma rede neural, utilizando representações do tabuleiro como entrada para prever movimentos com base em probabilidades.

## Estrutura do Repositório

O repositório contém scripts em Python que implementam os componentes principais da engine, incluindo manipulação de dados, modelo de rede neural, geração de movimentos e inferência. Abaixo está uma descrição dos arquivos principais:

**data_loader.py:** Carrega e pré-processa dados de treinamento, convertendo posições de xadrez em tensores 8x8x7 que representam o tabuleiro para alimentar a rede neural.

**move_dict.py:** Define um dicionário de mapeamento para movimentos de xadrez, convertendo notações (e.g., "e2e4") em índices para o vetor de saída da rede neural.

**generate_training_data.py:** Gera dados de treinamento a partir de partidas de xadrez ou autojogo, criando pares de entrada-saída (tensores de tabuleiro, vetores de movimento).

**matches:** Contém exemplos de partidas ou scripts para simular jogos, usados para testar a engine ou gerar dados adicionais.

**pieces.py:** Define as peças de xadrez e suas propriedades (e.g., movimentos válidos, tipos), servindo como base para a representação do tabuleiro.

**table.py:** Implementa a representação do tabuleiro de xadrez, incluindo métodos para gerar o tensor 8x8x7 com informações sobre todas as peças.

**inference.py:** Realiza inferência com o modelo treinado, recebendo um tensor de tabuleiro e produzindo um vetor de probabilidades para selecionar movimentos.

**model.py:** Define a arquitetura da rede neural, composta por uma convolução 2D inicial, convoluções residuais 2D e um Transformer encoder-only.

**utils.py:** Contém funções utilitárias, compartilhadas entre os outros scripts.

## Arquitetura da Rede Neural

A rede neural da XADREZIA é projetada para processar representações do tabuleiro e prever movimentos. Abaixo estão os detalhes do input, output e da arquitetura:

### Input:

O input da rede é um tensor 8x8x7, que representa o estado do tabuleiro de xadrez:

**Dimensões 8x8:** Correspondem às 64 casas do tabuleiro (8 linhas x 8 colunas).

**7 canais:** Cada canal representa um tipo de peça ou informação específica:

**6 canais para as peças:** (peão, cavalo, bispo, torre, dama, rei) de cada cor (brancas e pretas, totalizando 12 tipos de peças).

**canal adicional:** para informar a cor da peça.

Cada elemento do tensor indica a presença (ou ausência) de uma peça em uma casa específica.

### Output

**O output da rede é um vetor de probabilidades que codifica:**

O movimento selecionado - movimento específico, coluna de orimge e linha de origem (ex: "c5", "De2", "Bg4").

O vetor contém probabilidades para todos os movimentos possíveis, mapeados via move_dict.py. Por exemplo, o índice do vetor pode corresponder a um movimento específico, como "mover o peão de e2 para e4".

Durante a inferência, o movimento com maior probabilidade é selecionado.

### Arquitetura

A rede neural combina algumas técnicas comuns no aprendizado profundo para processar o tensor de entrada e produzir o vetor de saída.

### Convolução 2D Inicial:

Aplica uma camada convolucional 2D ao tensor 8x8x7 para extrair características espaciais do tabuleiro (ex: padrões de peças, controle de casas).

Usa filtros (8x8, 4x4 e 2x2) para capturar relações locais entre casas adjacentes.


### Convoluções Residuais 2D:

Uma sequência de blocos residuais (ResNet-style) refina as características extraídas.

Cada bloco contém camadas convolucionais com conexões de salto (skip connections) para evitar gradientes desaparecendo e melhorar o treinamento.

Preserva a dimensionalidade espacial (8x8) enquanto aumenta a profundidade das características na última dimensão.


### Transformer Encoder-Only:

Após as convoluções, as características extraídas alimentam um Transformer encoder-only.

O Transformer modela relações de longo alcance entre casas e peças, capturando dependências globais no tabuleiro (ex: coordenação entre peças distantes).

Usa atenção multi-cabeça (multi-head attention) para priorizar informações relevantes.

A saída do Transformer é processada por uma camada densa para produzir o vetor de probabilidades, que é decodificando em um lance.

![engine vs human](https://sdmntprwestus.oaiusercontent.com/files/00000000-f1f8-6230-98b5-085b88fc7147/raw?se=2025-05-02T22%3A22%3A50Z&sp=r&sv=2024-08-04&sr=b&scid=b77e90ee-bb50-5c51-bf70-d52d9bb7c754&skoid=51916beb-8d6a-49b8-8b29-ca48ed86557e&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-05-02T20%3A37%3A35Z&ske=2025-05-03T20%3A37%3A35Z&sks=b&skv=2024-08-04&sig=sMeg9UmUBNngent2CRI/Z7HXn5yJaViJjZr%2B85OW5BM%3D)