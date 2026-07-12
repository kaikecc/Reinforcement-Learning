# Reinforcement Learning para Detecção de Falhas 3W

Este repositório treina e avalia modelos para detecção de falhas em poços usando
o dataset 3W. Ele inclui um pipeline de treinamento, validação por instância,
um simulador/gêmeo digital e uma dashboard em tempo real.

## Estrutura

- `src/train_pipeline.py`: entrada principal para treino e avaliação.
- `src/classes/`: carregamento de dados, ambiente Gym, agentes e validação.
- `digital_twin/`: catálogo do dataset 3W, simulador físico e dashboard em tempo real.
- `analyses data/` e `src/notebook/`: notebooks exploratórios.

Os diretórios `logs/`, `models/`, `metrics/`, `img/` e `digital_twin/outputs/`
são artefatos gerados e não devem ser usados como fonte principal do código.

## Ambiente

Crie um ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O pipeline espera o dataset 3W em um diretório `3W/dataset` ao lado deste
repositório. Para usar outro caminho, passe `--dataset-path` ou defina a variável
de ambiente `THREE_W_ROOT`.

```powershell
$env:THREE_W_ROOT="C:\caminho\para\3W"
```

## Execução rápida

Validar o fluxo com Isolation Forest e gêmeo digital:

```powershell
python src/train_pipeline.py --use-digital-twin --event-code 1 --models IF --twin-scenarios 2 --twin-normal-rows 50 --twin-event-rows 50 --timesteps 1
```

Treinar DQN com gêmeo digital:

```powershell
python src/train_pipeline.py --use-digital-twin --event-code 1 --models DQN --timesteps 10000
```

Habilitar TensorBoard durante o treino:

```powershell
python src/train_pipeline.py --use-digital-twin --models DQN --tensorboard
```

## Gêmeo digital

Gerar catálogo e dashboard local do dataset:

```powershell
python digital_twin/create_digital_twin.py --max-files-per-class 3
```

Simular um poço:

```powershell
python digital_twin/well_simulator.py --event-code 1 --source any --scenarios 2 --normal-rows 3600 --event-rows 3600
```

Dashboard em tempo real:

```powershell
python digital_twin/realtime_simulator.py --port 8787 --event-code 1
```

Abra `http://127.0.0.1:8787`.

## Testes

```powershell
python -m pytest
python -m compileall -q src digital_twin
```
