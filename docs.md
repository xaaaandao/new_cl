# Documentação das alterações realizadas no código

Todas as alterações serão explicadas com exemplos do motivo de sua alteração.

## NETWORK

O código original (network.py) implementa a ResNet para CIFAR-10.

Uma modificação específica da arquitetura original para funcionar bem com imagens de resolução 32×32 pixels.

Ao ser aplicada em imagens maiores como 112x112 ou 224×224 terá problemas de performance e convergência.


O problema está no código:
```
self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(64)
self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
```

Que faz:

1. Kernel 3×3: Olha para regiões muito pequenas.

2. Stride 1: A convolução desliza de 1 em 1 pixel.

3. Ausência de MaxPool: Não há redução agressiva de tamanho.

Se a imagem de entrada é 32×32, a saída dessa primeira camada continua sendo 32×32. Isso é bom para o CIFAR-10, porque se fosse reduzido a imagem no início, sobrariam pouquíssimos pixels (ex: 8×8) para as camadas profundas processarem.

### Problema custo computacional:

Na ResNet padrão (ImageNet), a primeira camada é agressiva para reduzir a dimensionalidade:

1. Standard: Conv7x7, stride 2 → MaxPool3x3, stride 2.

Oque resulta em uma imagem anteriormente 224×224 virando 56×56 quase imediatamente (redução de 4x na altura e largura, ou 16x na área).

No código atual, a imagem 224×224 entra na layer1 com o tamanho total 224×224.

Impacto: A rede a processa 16 vezes mais pixels nas primeiras camadas profundas do que ela foi projetada para fazer.

Consequência: O consumo de VRAM vai aumentar drasticamente, obrigando o uso de batch_size minúsculos para não dar erro de memoria, o que prejudica o Contrastive Learning.

### Problema campo receptivo:
 
As primeiras camadas das Redes Neurais Convencionais "veem" bordas, as do meio "veem" formas, as finais "veem" objetos.

ResNet padrão: Com o downsampling rápido, um pixel na camada 3 já representa uma área grande da imagem original. A rede consegue "ver o todo" rapidamente.

ResNet CIFAR: Com stride=1 e kernel=3, a rede demora muito para "ver" o contexto global.

### Comparativo

| Característica      | Código antigo                        | Código novo                         |
| ------------------- | ------------------------------------ | ----------------------------------- |
| Primeira Conv       | 3×3, Stride 1                        | 7×7, Stride 2                       |
| Pooling Inicial     | Nenhum                               | 3×3 MaxPool, Stride 2               |
| Tamanho após Stem   | W×H (Igual à entrada)                | W/4×H/4 (16x menor área)            |
| Uso de Memória      | Altíssimo em altas resoluções        | Otimizado                           |
| Foco                | Detalhes finos em imagens minúsculas | Estrutura global em imagens normais |
| Caso de Uso Inicial | CIFAR-10/100, TinyImageNet2          | ImageNet, COCO, Dados Reais         |

## Normalização

A média e desvio padrão do espectro RGB é parte do MODELO, não apenas dos dados.

### Problema

É assumido que os dados de treino e teste são amostrados de uma mesma distribuição subjacente.

A normalização é uma transformação que mapeia o espaço de entrada original para um espaço latente normalizado.

Ao utilizar a média do treino para normalizar o teste, mantemos o sistema de referência fixo.

Usar a média do teste, é alterar o sistema de referência, introduzindo artificialmente um Desvio de Covariável. Desta forma o modelo interpretaria diferenças estatísticas reais (ex: um conjunto de teste mais escuro) como sendo dados "normais" (média zero), eliminando a informação que permitiria ao modelo distinguir essa variação.

Além disso, o pipeline de teste deve simular o ambiente de produção com fidelidade máxima. Em um cenário de inferência, as requisições chegam unitariamente.

Não é possível calcular média e desvio padrão estatisticamente significativos de uma única amostra.

Calcular a média de uma única imagem e subtraí-la de si mesma resultaria em um tensor de zeros (ou próximo disso), destruindo a informação visual.

Portanto, a única abordagem viável é aplicar as estatísticas pré-calculadas e armazenadas durante a fase de treinamento.