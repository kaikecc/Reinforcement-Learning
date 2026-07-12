# 3W Digital Twin

Este diretório contém um gêmeo digital local para a base 3W. Ele não copia os
arquivos Parquet originais; ele cria uma camada de catálogo, perfil estatístico
e painel interativo em HTML a partir de `..\3W\dataset`.

## O que é gerado

- `outputs/twin_catalog.json`: catálogo de instâncias, com classe, fonte,
  período, duração, quantidade de linhas, cobertura de variáveis e distribuição
  de rótulos.
- `outputs/twin_summary.csv`: resumo por classe de evento.
- `outputs/variable_profiles.csv`: estatísticas globais por variável e classe.
- `outputs/dashboard.html`: painel interativo com visão por classe, fonte,
  cobertura de variáveis, linha do tempo e séries representativas.
- `well_simulator.py`: simulador operacional de poço. Ele cria sinais novos
  por equações físicas simplificadas, calibradas por estatísticas das
  instâncias reais, simuladas e desenhadas do 3W. O método de replay segue
  disponível em `simulate_replay()`, mas não é usado pelo fluxo padrão.
- `realtime_simulator.py`: simulador em tempo real com interface gráfica em
  navegador e API HTTP para o `train_pipeline.py` consumir janelas vivas da
  simulação.

## Como executar

Na raiz do repositório `Reinforcement-Learning`:

```powershell
python digital_twin/create_digital_twin.py
```

Para um teste rápido usando poucos arquivos por classe:

```powershell
python digital_twin/create_digital_twin.py --max-files-per-class 3
```

Depois abra:

```powershell
.\digital_twin\outputs\dashboard.html
```

## Simular um poço com falha

Exemplo com 2 cenários do evento 1, cada um com 1 hora normal e 1 hora de falha.
Por padrão, `--source any` usa perfis combinados de instâncias reais,
simuladas e desenhadas:

```powershell
python digital_twin/well_simulator.py --event-code 1 --source any --scenarios 2 --normal-rows 3600 --event-rows 3600 --output digital_twin/outputs/simulated_well.parquet
```

O arquivo gerado possui as colunas esperadas pelo pipeline de reinforcement
learning:

```text
timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, well, code
```

## Usar com `train_pipeline.py`

No repositório `Reinforcement-Learning`, rode:

```powershell
python src/train_pipeline.py --use-digital-twin --event-code 1 --models DQN --twin-source any --twin-scenarios 10 --twin-normal-rows 3600 --twin-event-rows 3600
```

Para validar rapidamente sem um treino DQN longo:

```powershell
python src/train_pipeline.py --use-digital-twin --event-code 1 --models IF --twin-scenarios 2 --twin-normal-rows 50 --twin-event-rows 50 --timesteps 1
```

## Simulação em tempo real

Inicie a interface gráfica. A simulação roda em ciclos contínuos de operação
normal seguida de falha e só para quando você clicar em `Pausar` ou encerrar o
processo Python:

```powershell
cd C:\Users\kaike\Documents\UFSC\CODE\Reinforcement-Learning
python digital_twin/realtime_simulator.py --port 8787 --event-code 1 --normal-rows 600 --event-rows 600 --rows-per-tick 5 --tick-seconds 0.5
```

Abra:

```text
http://127.0.0.1:8787
```

Para validar a operacao com um modelo treinado, informe o tipo e o caminho do
arquivo salvo pelo Stable-Baselines3. O caminho pode ser passado com ou sem
`.zip`:

```powershell
python digital_twin/realtime_simulator.py --port 8787 --event-code 1 --model-type DQN --model-path "models\Abrupt Increase of BSW\realtime_twin\DQN\150000\_DQN.zip"
```

A cada amostra emitida, o simulador monta `obs` com as variaveis do pipeline e
executa:

```python
action = model.predict(obs, deterministic=True)[0]
```

A interface compara essa acao com a acao esperada pela classe real
(`0 = normal`, `1 = falha`) e mostra acao do modelo, acao esperada, validacao e
acuracia online.

A interface mostra um gráfico separado para cada variável usada pelo pipeline:
`P-PDG`, `P-TPT`, `T-TPT`, `P-MON-CKP` e `T-JUS-CKP`. Em todos eles, o eixo X é
o tempo e o eixo Y é o valor da variável.

A API expõe a janela gerada em:

```text
http://127.0.0.1:8787/api/samples
```

No repositório `Reinforcement-Learning`, faça o pipeline consumir a simulação:

```powershell
cd C:\Users\kaike\Documents\UFSC\CODE\Reinforcement-Learning
python src/train_pipeline.py --use-realtime-twin --event-code 1 --models DQN --realtime-url http://127.0.0.1:8787 --realtime-min-rows 1000
```

Para validação rápida:

```powershell
python src/train_pipeline.py --use-realtime-twin --event-code 1 --models IF --realtime-url http://127.0.0.1:8787 --realtime-min-rows 100 --timesteps 1
```

## Interpretação

Este gêmeo digital é uma representação operacional do dataset: ele permite
inspecionar quais poços/eventos existem, como as classes estão distribuídas,
quais variáveis estão presentes, onde há lacunas de medição e como séries reais,
simuladas ou desenhadas se comportam em cada classe.

Ele serve como base para etapas futuras, como detecção de anomalias, treinamento
de modelos e comparação entre uma instância observada e o histórico do 3W.
