# Relatorio: desenvolvimento do algoritmo de gemeo digital em tempo real

## 1. Visao geral

O arquivo `realtime_simulator.py` implementa um gemeo digital em tempo real para pocos de petroleo usando a base 3W. O objetivo do algoritmo e reproduzir, em uma interface web, uma linha do tempo operacional composta por duas fases: operacao normal e ocorrencia de falha. Enquanto os dados sao emitidos em pequenos blocos, o sistema tambem executa um modelo treinado para classificar a condicao atual do poco como normal ou falha.

O desenvolvimento foi estruturado para unir tres partes principais:

- geracao da linha do tempo operacional a partir de dados reais, simulados ou desenhados do dataset 3W;
- emissao continua de amostras, simulando chegada de dados de sensores em tempo real;
- validacao online de um modelo de deteccao, com metricas exibidas em uma dashboard Flask.

Assim, o algoritmo nao apenas reproduz dados historicos. Ele organiza esses dados como um fluxo vivo, com controle de execucao, janela de amostras, predicao por modelo e acompanhamento de desempenho.

## 2. Base do gemeo digital

O `realtime_simulator.py` depende diretamente do modulo `well_simulator.py`. Esse modulo define a estrutura fisica-sintetica do gemeo digital: uma sequencia de leituras de sensores de um poco com comportamento normal seguida por um evento de falha.

As variaveis usadas pelo simulador sao:

- `P-PDG`
- `P-TPT`
- `T-TPT`
- `P-MON-CKP`
- `T-JUS-CKP`

A saida padrao segue o contrato usado pelo pipeline de treinamento:

```text
timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, well, code
```

Esse formato permite que o mesmo dado gerado pelo gemeo digital seja consumido tanto pela dashboard quanto pelo `train_pipeline.py`.

## 3. Configuracao do tempo de execucao

O desenvolvimento comeca pela classe `RuntimeConfig`, que concentra todos os parametros operacionais do simulador:

- caminho do dataset 3W;
- codigo do evento de falha;
- origem dos dados da falha e da fase normal;
- quantidade de linhas normais e linhas de falha;
- frequencia de emissao das amostras;
- quantidade de linhas emitidas por ciclo de atualizacao;
- nivel de ruido gaussiano;
- semente aleatoria;
- caminho e tipo do modelo treinado;
- caminho do normalizador (`scaler`);
- limite maximo de historico armazenado.

Essa estrutura foi criada para separar configuracao de logica. Com isso, o simulador pode ser reiniciado com novos parametros sem alterar o codigo fonte.

Os valores padrao simulam 600 amostras normais e 600 amostras de falha, emitindo 5 linhas a cada 0,5 segundo. O modelo padrao e um DQN salvo em:

```text
models/Abrupt Increase of BSW/realtime_twin/DQN/150000/_DQN.zip
```

## 4. Estado do simulador em tempo real

A classe central do algoritmo e `RealtimeTwinState`. Ela representa o estado vivo do gemeo digital. Essa classe controla:

- se a simulacao esta executando ou pausada;
- a posicao atual dentro da linha do tempo;
- o ciclo atual de simulacao;
- o total de amostras ja emitidas;
- o historico recente de amostras;
- o modelo carregado;
- o normalizador usado nas observacoes;
- a ultima predicao realizada;
- os erros de modelo ou normalizacao.

Como a simulacao roda em uma thread separada da API Flask, o estado usa `threading.Lock`. Esse lock evita concorrencia indevida entre a thread que gera amostras e as rotas HTTP que consultam ou alteram o estado.

## 5. Geracao da linha do tempo operacional

A linha do tempo e criada pelo metodo `_build_timeline`. O processo e:

1. Instanciar `DigitalTwinWellSimulator` apontando para o dataset 3W.
2. Construir um `Scenario` com os parametros atuais.
3. Solicitar ao simulador uma sequencia de dados.
4. Converter timestamps para texto no formato `YYYY-MM-DD HH:MM:SS`.
5. Ajustar o normalizador online, se necessario.
6. Retornar apenas as colunas padronizadas.

No `well_simulator.py`, a simulacao de um cenario funciona assim:

1. Carrega um segmento de classe 0, representando operacao normal.
2. Carrega um segmento da classe de evento desejada, por exemplo evento 1 para aumento abrupto de BSW.
3. Concatena os dois segmentos.
4. Cria uma linha do tempo com frequencia de 1 segundo.
5. Define o nome do poco e o codigo do evento.
6. Opcionalmente adiciona ruido gaussiano proporcional ao desvio padrao de cada variavel.

Essa composicao e importante porque transforma arquivos historicos do 3W em uma narrativa operacional: primeiro o poco opera normalmente, depois entra em estado de falha.

## 6. Selecao dos dados no dataset 3W

O metodo `_load_segment` de `well_simulator.py` seleciona os arquivos candidatos de acordo com:

- classe desejada;
- fonte dos dados (`real`, `simulated`, `drawn` ou `any`);
- quantidade de linhas solicitada.

Depois, o algoritmo escolhe arquivos em ordem aleatoria controlada pela semente. Ele procura um arquivo que contenha todas as colunas necessarias, remove valores infinitos ou ausentes e extrai uma janela continua de linhas.

Se o arquivo escolhido tiver mais linhas que o necessario, uma janela aleatoria e recortada. Se tiver menos linhas, o conteudo e repetido ate completar o tamanho solicitado.

Para classe normal, o campo `class` e forçado para `0`. Para eventos de falha, qualquer rotulo normal remanescente e substituido pelo codigo de evento, garantindo que a fase de falha seja tratada como anomala.

## 7. Simulacao continua por ciclos

O metodo `_run` e o motor em tempo real do algoritmo. Ele roda continuamente em uma thread daemon criada no construtor de `RealtimeTwinState`.

O ciclo de execucao e:

1. Aguardar `tick_seconds`.
2. Entrar na secao protegida por lock.
3. Verificar se a simulacao esta em modo `running`.
4. Se a linha do tempo atual terminou, iniciar novo ciclo.
5. Selecionar um bloco de linhas de tamanho `rows_per_tick`.
6. Executar predicao do modelo para cada linha.
7. Adicionar as linhas processadas ao historico.
8. Remover linhas antigas caso o historico exceda `max_history_rows`.
9. Atualizar ponteiro e total emitido.

Esse desenho permite que o gemeo digital opere indefinidamente. Ao fim de cada linha do tempo, `_start_next_cycle` gera uma nova sequencia usando uma semente diferente e iniciando o timestamp logo apos a ultima amostra emitida.

## 8. Carregamento do modelo

O metodo `_load_model` carrega o modelo de acordo com `model_type`. O simulador suporta:

- `DQN`
- `PPO`
- `A2C`
- `RNA`

Para modelos de aprendizado por reforco, o carregamento usa Stable-Baselines3:

```python
DQN.load(path)
PPO.load(path)
A2C.load(path)
```

Para modelo neural supervisionado (`RNA`), o carregamento usa TensorFlow/Keras com o objeto customizado `F1Score`.

Antes de carregar, `_resolve_model_path` verifica se o caminho informado existe. Quando o usuario informa um caminho sem `.zip`, o algoritmo tambem procura automaticamente a versao com extensao `.zip`, o que facilita o uso de modelos salvos pelo Stable-Baselines3.

Se o modelo nao puder ser carregado, a simulacao continua emitindo dados. Nesse caso, as linhas recebem status de "sem modelo", e a dashboard informa que a predicao nao foi executada.

## 9. Normalizacao das observacoes

Antes de enviar uma amostra ao modelo, o algoritmo monta uma observacao numerica com as cinco variaveis principais. Esse processo ocorre em `_build_observation`.

O vetor e construido na ordem:

```text
P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP
```

Se existir um `scaler`, a observacao e normalizada antes da predicao. O simulador aceita dois modos:

- scaler carregado de arquivo via `joblib`;
- scaler online ajustado automaticamente com `MinMaxScaler(feature_range=(-1, 1))`.

O scaler online e ajustado sobre a linha do tempo completa do ciclo atual. Essa estrategia permite usar o simulador mesmo quando o scaler de treinamento nao foi salvo. Porem, o modo mais consistente com o treinamento e fornecer o mesmo scaler usado durante a criacao do modelo.

## 10. Predicao online

A predicao e feita linha a linha pelo metodo `_predict_row`.

O fluxo e:

1. Verificar se existe modelo carregado.
2. Construir a observacao com `_build_observation`.
3. Executar `_predict_action`.
4. Medir o tempo de predicao em milissegundos.
5. Calcular a acao esperada.
6. Comparar acao prevista e acao esperada.
7. Registrar status, validacao e metadados da predicao.

Para modelos `DQN`, `PPO` e `A2C`, a chamada segue a interface de aprendizado por reforco:

```python
action = model.predict(obs, deterministic=True)[0]
```

Para `RNA`, o modelo retorna probabilidades por classe, e a acao escolhida e o indice de maior probabilidade:

```python
probabilities = model.predict(np.atleast_2d(obs), verbose=0)
action = np.argmax(probabilities, axis=1)[0]
```

A acao esperada e binaria:

- classe real `0` significa operacao normal, logo a acao esperada e `0`;
- qualquer classe diferente de `0` significa falha, logo a acao esperada e `1`.

Essa conversao permite avaliar modelos binarios de deteccao de falha mesmo quando o dataset possui diferentes codigos de evento.

## 11. Validacao do modelo em tempo real

Cada linha emitida recebe campos adicionais:

- `model_action`: acao prevista pelo modelo;
- `expected_action`: acao esperada pela classe real;
- `model_status`: `normal`, `falha`, `sem modelo` ou `erro`;
- `model_validation`: `correto`, `incorreto`, `sem modelo` ou `erro`;
- `model_correct`: valor booleano indicando acerto;
- `prediction_time_ms`: tempo de inferencia.

Essas informacoes permitem que a dashboard calcule metricas acumuladas durante a execucao.

## 12. Metricas calculadas

O metodo `status` resume o estado atual do gemeo digital. Alem de dados operacionais, ele calcula metricas de desempenho do modelo:

- acuracia online;
- precisao;
- recall;
- F1 score;
- matriz de confusao;
- percentuais de TP, TN, FP e FN.

A matriz de confusao e calculada usando a interpretacao binaria:

- `TP`: a amostra era falha e o modelo previu falha;
- `TN`: a amostra era normal e o modelo previu normal;
- `FP`: a amostra era normal e o modelo previu falha;
- `FN`: a amostra era falha e o modelo previu normal.

Essas metricas sao recalculadas a partir do historico emitido sempre que a API consulta o status.

## 13. API HTTP

O Flask expõe uma API simples para controlar e consultar o gemeo digital.

As rotas principais sao:

- `GET /`: retorna a dashboard HTML.
- `GET /api/version`: informa versao da aplicacao, arquivo servido e PID.
- `GET /api/status`: retorna o estado atual do simulador.
- `GET /api/window?limit=300`: retorna as ultimas amostras da janela.
- `GET /api/samples`: retorna todas as amostras armazenadas, ou um limite definido.
- `POST /api/control`: executa `start`, `pause` ou `reset`.

Essa API tambem permite que o `train_pipeline.py` consuma dados vivos do simulador. No modo `--use-realtime-twin`, o pipeline inicia a simulacao, coleta amostras ate atingir um minimo configurado e transforma o resultado no mesmo formato numpy usado pelo treinamento.

## 14. Dashboard web

A dashboard e definida diretamente dentro de `realtime_simulator.py` como a string `DASHBOARD_HTML`.

Ela apresenta:

- controles de iniciar, pausar e resetar;
- selecao de evento;
- selecao da fonte dos dados;
- configuracao de ruido;
- selecao do tipo de modelo;
- caminho do modelo;
- caminho do scaler;
- status da simulacao;
- progresso do ciclo;
- classe atual;
- resultado da predicao;
- acuracia, precisao, recall e F1;
- matriz de confusao percentual;
- graficos das cinco variaveis de sensores;
- distribuicao das classes;
- detalhes da ultima predicao;
- tabela com a ultima amostra.

O frontend consulta `/api/window` periodicamente, redesenha os graficos em canvas e atualiza os indicadores. Dessa forma, o usuario acompanha a evolucao da simulacao e do modelo sem precisar interagir com o terminal.

## 15. Controle de reinicializacao

O metodo `reset` permite reiniciar o gemeo digital com novos parametros enviados pela interface ou pela API.

Durante o reset:

1. O algoritmo atualiza os campos de configuracao recebidos.
2. Pausa a execucao.
3. Zera ponteiro, ciclo, contador de linhas e historico.
4. Reposiciona o inicio temporal em `2026-01-01 00:00:00`.
5. Limpa erros e ultima predicao.
6. Recarrega modelo e scaler.
7. Reconstrói a linha do tempo.

Esse comportamento torna o simulador adequado para experimentos comparativos, por exemplo alternar entre eventos, fontes de dados, modelos e scalers.

## 16. Fluxo geral do algoritmo

O algoritmo completo pode ser resumido assim:

```text
Inicializar configuracao
Carregar modelo
Carregar scaler, se informado
Gerar linha do tempo normal + falha
Iniciar thread de simulacao

Enquanto o processo estiver ativo:
    aguardar intervalo de tick
    se simulacao estiver pausada:
        continuar
    se linha do tempo acabou:
        gerar novo ciclo
    selecionar proximo bloco de amostras
    para cada amostra:
        montar observacao
        normalizar observacao
        executar modelo
        comparar com classe real
        registrar predicao e validacao
    armazenar amostras no historico
    atualizar ponteiro e contadores

Enquanto isso:
    Flask responde API e dashboard
    Dashboard consulta estado periodicamente
    Pipeline externo pode coletar amostras via HTTP
```

## 17. Decisoes de desenvolvimento

Algumas decisoes importantes aparecem na implementacao:

- O simulador foi separado em dois niveis: `well_simulator.py` gera cenarios, enquanto `realtime_simulator.py` transforma esses cenarios em fluxo temporal e dashboard.
- A emissao de dados usa thread separada para manter a API responsiva.
- O historico tem tamanho maximo para evitar crescimento ilimitado de memoria.
- O modelo e opcional, permitindo usar a ferramenta tambem como simulador de sensores.
- O sistema aceita diferentes tipos de modelo, reaproveitando tanto aprendizado por reforco quanto rede neural supervisionada.
- A avaliacao online foi simplificada para problema binario: normal contra falha.
- O dashboard foi embutido no proprio arquivo para reduzir dependencias de frontend.
- A escolha automatica de porta livre evita confusao quando ja existe uma dashboard antiga rodando.

## 18. Relacao com o pipeline de treinamento

O `train_pipeline.py` foi adaptado para consumir o gemeo digital de duas formas:

- `--use-digital-twin`: gera um dataset sintetico completo via `DigitalTwinWellSimulator`.
- `--use-realtime-twin`: coleta amostras vivas da API do `realtime_simulator.py`.

No modo em tempo real, o pipeline chama `/api/control` para iniciar a simulacao, consulta `/api/samples` ate atingir o minimo de linhas e entao converte as amostras para numpy. Isso permite treinar e avaliar modelos com dados emitidos dinamicamente, mantendo o mesmo contrato de dados do 3W.

## 19. Limitacoes atuais

O algoritmo implementado e funcional, mas possui limitacoes importantes:

- O gemeo digital e baseado em replay de dados historicos, nao em uma simulacao fisica completa do poco.
- A transicao normal-falha e feita por concatenacao de segmentos, sem modelagem gradual entre regimes.
- A avaliacao online reduz todas as falhas para uma acao binaria, portanto nao mede classificacao multiclasse por tipo de evento.
- O scaler online pode divergir do scaler usado no treinamento, gerando diferenca entre distribuicao de treino e inferencia.
- A dashboard esta embutida como string HTML, o que facilita distribuicao, mas dificulta manutencao em projetos maiores.
- As metricas sao recalculadas sobre o historico armazenado, nao sobre um banco persistente.

## 20. Conclusao

O algoritmo de gemeo digital em `realtime_simulator.py` foi desenvolvido como uma camada operacional sobre o dataset 3W. Ele transforma segmentos historicos de pocos em um fluxo temporal continuo, adiciona controle em tempo real, integra modelos treinados e fornece validacao online por meio de uma dashboard web.

O resultado e uma ferramenta util para demonstrar comportamento de sensores, testar modelos de deteccao de falha, acompanhar metricas de inferencia e fornecer dados vivos para o pipeline de aprendizado por reforco. Embora nao seja um modelo fisico completo do sistema de producao, ele cumpre o papel de gemeo digital orientado a dados: reproduz estados operacionais observados, injeta eventos de falha e permite avaliar respostas de modelos em um ambiente controlado e interativo.
