# Mini-Projeto 2 - Classificação de Imagética Motora (EEG)

Pipeline avançado de classificação de sinais EEG do dataset EEGBCI / PhysioNet entre as classes T1 (mão esquerda) vs T2 (mão direita) durante atividade imagética motora, comparando algoritmos clássicos de Machine Learning e de Deep Lerning (MLP, CNN 2D, CNN 3D), com testes de redução de dimensionalidade, balanceamento, ensembles e rejeição de classificadores.

---

## 1. Introdução

Este projeto desenvolve e avalia pipelines para classificação de sinais de EEG durante atividade imagética motora, utilizando a biblioteca MNE-Python e um conjunto de bibliotecas de machine lerning e deep learning.

O foco principal é comparar diferentes abordagens de modelos para o problema da classificação do movimento imaginado, em busca dos melhores modelos.

São comparados:

- **Algoritmos clássicos**: Naive Bayes, KNN, Regressão Logística, SVM (linear e RBF), Random Forest;
- **Redes neurais**: MLP (4 arquiteturas), CNN 2D sobre maopa topográfico, CNN 2D sobre STFT, CNN 3D topográfica espaço-temporal-frequencial;
- **Representações**: features clássicas (banda + Hjorth + entropia), CSP, FBCSP + MIBIF, SPoC, Riemann (Tangent Space), STFT, mapa topográfico;
- **Reduções de dimensionalidade**: PCA, LDA (sem/com shrinkage), t-SNE, SelectKBest;
- **Balanceamento**: SMOTE, undersampling (induzidos artificialmente, já que o dataset original é balanceado);
- **Combinação e rejeição**: Voting (hard/soft) e rejeição por limiar de confiança.

**Validação:** todos os testes usam GroupKFold(5) e/ou Leave-One-Subject-Out (LOSO) - com nenhum sujeito aparece simultaneamente em treino e teste.

---

## 2. Dataset

- **Fonte**: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (EEGBCI, PhysioNet)
- **Tipo de problema**: classificação binária (T1 vs T2)

### Descrição dos dados

| Item | Valor |
|---|---|
| Nº de sujeitos | **10** (subset reproduzível, seed=42 sobre os 109 totais) |
| Nº de canais (raw) | 64 (sistema 10-10) |
| Nº de canais (após seleção motora) | **21** (FC5–FC6, C5–C6, CP5–CP6 e linhas centrais) |
| Frequência de amostragem | 160 Hz |
| Duração total das épocas | 4 s pós-cue |
| Runs utilizados | **04, 08, 12** (MI mão esquerda/direita) |

### Classes utilizadas

- **Classe 0 — T1**: imaginar abrir/fechar mão esquerda.
- **Classe 1 — T2**: imaginar abrir/fechar mão direita.

### Distribuição de classes

| Classe | Nº de épocas | Proporção |
|---|---:|---:|
| T1 (esquerda) | 225 | 50,0 % |
| T2 (direita) | 225 | 50,0 % |
| **Total** | 450 | 100 % |

**OBS:** O dataset é perfeitamente balanceado, portanto SMOTE/undersampling não foram necessários no fluxo principal, foram avaliados apenas em um cenário de desbalanceamento induzido artificialmente para fins didáticos (Teste 3 do notebook de DL).

---

## 3. Pré-processamento

Todo o pré-processamento está implementado em `src/notebooks/Preprocessamento_Geral.ipynb`. As decisões metodológicas e respectivas justificativas:

| Etapa | Decisão | Justificativa |
|---|---|---|
| **Tarefa** | Runs 4, 8, 12 => MI esquerda (T1) vs direita (T2) | Paradigma canônico EEGBCI |
| **Seleção de canais** | 21 sensores do córtex sensoriomotor (FC, C, CP) | Foco neurofisiológico em mu/beta |
| **Referenciamento** | CAR (Common Average Reference) | Padrão em MI, atenua ruído comum |
| **Filtragem** | Notch 60 Hz + harmônicos < Nyquist + band-pass 8–30 Hz | Bandas mu e beta, rede elétrica EUA (local de gravação do EEG)|
| **ICA** | Picard (extended=True, fallback Infomax) ajustado por sujeito | Artefatos são sujeito-específicos, sem vazamento entre sujeitos |
| **Classificação ICA** | ICLabel (p > 0,70) + `find_bads_eog` + `find_bads_muscle` + kurtose &gt; 8 (união, threshold conservador) | Não descartar componentes motores legítimos |
| **Epocagem** | 0 a 4 s pós-evento, sem baseline subtraída | Janela típica de MI |
| **Normalização** | StandardScaler / RobustScaler ajustados **somente no treino de cada fold** | Evita vazamento treino→teste |
| **Teste de Sanidade** | ERD/ERS tempo-frequência (Morlet) em C3/C4 | Validar lateralização contralateral esperada |

### Pipeline em Batch

O pipeline foi encapsulado na função preprocess_subject_run_v2(...) (modular, retomável, com cache em disco) e roda em todos os sujeitos/runs gerando:

- preprocessed_v2/per_run/{S###}_R{##}-epo.fif - 1 arquivo por run.
- preprocessed_v2/per_subject/{S###}_allruns-epo.fif - concatenação por sujeito (input para os notebooks de modelagem).
- preprocessed_v2/manifest.csv - log de status (ok/cached/failed) e nº de épocas por arquivo.
- Cada Epochs.metadata recebe colunas subject, run e label, essenciais para GroupKFold.

### Visualização

Cada etapa do pipeline gera figuras de verificação (em modo `visualize=True`):

- PSD antes e depois do notch e do ICA;
- Topografias (montagem, sensores motores, componentes ICA, componentes excluídas);
- ERPs médios T1 vs T2 em C3/Cz/C4;
- Mapas tempo-frequência (Morlet) em C3 e C4 - teste de sanidade do padrão ERD/ERS contralateral.

---

## 4. Representações de Séries Temporais

Foram construídas duas famílias de representações complementares.

### Representação A - Features clássicas tabulares

Para cada época × cada canal motor, 7 features:

- **3 potências relativas de banda:** (mu 8–13 Hz, beta-low 13–20 Hz, beta-high 20–30 Hz) via Welch + normalização pela soma das 3;
- **Variância:** do sinal no tempo;
- **Mobilidade e Complexidade de Hjorth:** (Activity é redundante com a variância e foi omitida);
- **Entropia espectral de Shannon:** sobre a PSD normalizada.

**Resultado:** matriz (450 épocas, 7 × 21 canais = 147 features). Em alguns testes foram ultilizados 105 features (15 canais).

**Justificativa neurofisiológica**: as bandas mu e beta concentram o ERD/ERS sensoriomotor durante MI. Features de Hjorth são descritores estatísticos no domínio do tempo proporcionais à frequência média (Mobility) e largura de banda (Complexity); entropia espectral mede o quão "plana" é a distribuição espectral.

#### 4 variantes da representação clássica (Teste 2 do notebook ML)

Cruzando duas decisões de design:

| | sem RMS | com RMS (sinal/RMS antes da extração) |
|---|---|---|
| **Por canal** (147 features) | canal_sem_rms (≡ baseline T1) | canal_com_rms |
| **Média entre canais** (7 features) | media_sem_rms | media_com_rms |

### Representação B - Filtros espaciais e tempo-frequência

- **CSP** (mne.decoding.CSP, n_components=4, regularização Ledoit-Wolf, log-power) - Teste 3.
- **FBCSP** (Filter-Bank CSP) sobre 5 sub-bandas [8–12, 12–16, 16–20, 20–24, 24–28] Hz (Butterworth ordem 4 zero-fase) => 20 features brutas => MIBIF (SelectKBest com mutual_info_classif, top-8) - Teste 4.
- **SPoC**, **Riemann (Tangent Space)** - usados no notebook de DL como baseline espacial (LOSO AUC ≈ 0,67).
- **STFT** por canal (nperseg=128, noverlap=64, banda 8–30 Hz, log-power) => tensor (N, C, F, T) - input das CNN 2D/3D.
- **Layout topográfico 3×5** (3 linhas FC/C/CP × 5 colunas) => input da CNN 3D.

#### Discussão sobre série temporal direta

O sinal pré-processado (épocas × canais × tempo) é mantido em X_temporal para:
- entrada natural de CNNs (1D, 2D sobre STFT, 3D topográfica);
- **vantagens:** preserva 100 % da informação espectral e temporal;
- **limitações:** Analisa 12 mil variáveis sem filtro por época pede modelos visuais (CNN/EEGNet) ou bloqueios fortes contra memorização.

---

## 5. Redução de Dimensionalidade

Implementada e comparada no Preprocessamento_Geral.ipynb (seção 5) e refinada no Modelos_ML.ipynb (Teste 5):

### Técnicas avaliadas

- **PCA** (não supervisionada) - máxima variância;
- **LDA** (supervisionada) - em problema binário projeta para 1D no eixo de máxima separabilidade;
  - Variante 1: solver='svd' (sem regularização);
  - Variante 2: solver='eigen', shrinkage='auto' (Ledoit-Wolf);
- **t-SNE** - apenas visualização (não tem transform para novos dados);
- **SelectKBest** (f_classif, k=50) - seleção supervisionada preservando features originais.

### Configurações principais

- **Nº de componentes PCA**: 22 (aproximadamente 95% da variância acumulada nas features clássicas).
- **LDA n_components**: 1 (binário).

### Resultados - comparação inicial (preprocessing.5.2, GroupKFold 5)

| Estratégia | Acurácia média | Desvio padrão |
|---|---:|---:|
| LDA puro (105 features) | 55,56 % | +/- 3,65 % |
| PCA(22) + LDA | 53,78 % | +/- 7,88 % |
| PCA(22) + SVM linear | 51,78 % | +/- 4,07 % |
| LDA(1) + SVM linear | 55,11 % | +/- 3,76 % |
| **SelectKBest(50) + SVM linear** | 62,22 % | +/- 10,40 % |

### Análise

- PCA não supervisionada reduziu performance e aumentou instabilidade, eixo de máxima variância diferente do eixo de máxima separação.
- LDA como redutor (1D) manteve performance similar ao LDA puro.
- SelectKBest supervisionado foi a melhor estratégia neste primeiro varrimento (+6,7pp), mas com maior variância entre folds.
- t-SNE / PCA 2D mostraram sobreposição significativa entre classes, confirmação visual da dificuldade clássica de classificação cross-subject em MI-EEG.

No notebook ML (Teste 5), o LDA com shrinkage='auto' foi reavaliado em conjunto com cada uma das 4 variantes de features e cada um dos 6 classificadores. A melhor configuração geral foi KNN + canal_com_rms + LDA(shrinkage=auto) com 0,598 acc.

---

## 6. Modelos de Machine Learning

Implementados em src/notebooks/Modelos_ML.ipynb. Estrutura: 5 testes do mais simples ao mais complexo.

| Teste | Pré-processamento | Modelos | Particularidade |
|---:|---|---|---|
| **1** | 7 features clássicas POR canal | 5 classificadores | Baseline; LOSO + GroupKFold(5) |
| **2** | 4 variantes (canal/média × sem/com RMS) | 5 classificadores | Estuda agregação espacial e correção de magnitude |
| **3** | CSP (com/sem RMS) | 5 classificadores | Filtros espaciais supervisionados |
| **4** | FBCSP (5 sub-bandas) + MIBIF (top-8) | 5 classificadores | Extensão multi-banda do CSP |
| **5** | 4 variantes × 3 estratégias de LDA | 6 classificadores (inclui SVM Linear) | Compara LDA (com/sem shrinkage) como redução supervisionada |

### Configurações dos modelos clássicos

| Modelo | Configuração |
|---|---|
| Naive Bayes | GaussianNB() |
| KNN | n_neighbors=7, n_jobs=-1 |
| Logistic Regression | C=1.0, penalty='l2', solver='lbfgs', max_iter=2000 |
| SVM (RBF) | kernel='rbf', C=1.0, gamma='scale', probability=True |
| SVM (Linear) | kernel='linear', C=1.0, probability=True |
| Random Forest | n_estimators=300, n_jobs=-1 |

Todos os classificadores são encapsulados em Pipeline([StandardScaler, ..., clf]) para garantir que scaler, redutor e classificador sejam ajustados somente no treino de cada fold.

### Lógica de avaliação dentro de cada teste

Cada teste é organizado em 4 sub-células, idênticas por propósito:

1. **Pré-processamento / extração** - converte épocas em matriz (n_épocas, n_features), adequada ao formato esperado pelos classificadores.
2. **Treino e validação** - define o banco de modelos e roda cross_validate com 6 métricas (accuracy, balanced_accuracy, precision, recall, F1, ROC-AUC) sob LOSO + GroupKFold(5).
3. **Métricas** - tabela média (dp), barras (acc, F1, AUC), ROC out-of-fold, matrizes de confusão out-of-fold e sanity checks: DummyClassifier, permutation test (n=50), e demo de vazamento (StratifiedKFold vs GroupKFold).
4. **Dinâmica de aprendizado** - três famílias de curvas:
   - (A) Loss × epoch e accuracy × epoch via SGDClassifier (partial_fit, log-loss + hinge);
   - (B) Accuracy × n_estimators para Random Forest com warm_start=True;
   - (C) Learning curves do sklearn (acc × tamanho do treino) com GroupKFold(5).

---

## 7. Redes Neurais

Implementadas em src/notebooks/Modelos_Deep_learning.ipynb. 10 testes progressivos:

| Teste | Modelo | Representação | Pergunta investigada |
|---:|---|---|---|
| 1 | MLP (128, 64, 32) + BN + Dropout | Features clássicas (147) | Baseline DL; teto sobre features clássicas |
| 2 | MLP + PCA | Features → PCA(k variável) | Reduzir dim ajuda? |
| 3 | MLP + SMOTE / Undersampling | Desbalanceamento induzido | Como o MLP responde a desbalaceamento? |
| 4 | MLP × 7 extratores espaciais | FBCSP, SPoC, Riemann, Wavelet, FB-SPoC | Features espaciais > clássicas? |
| 5 | MLP - grid de hiperparâmetros | SPoC k=4 | Largura, profundidade, α, ativação |
| 6 | LDA, LogReg, Linear SVM, RBF SVM, MLP | SPoC | MLP supera lineares? |
| 7 | Análise hemisférica | Esquerdo/Direito/Ambos | Lateralidade contralateral é o sinal dominante? |
| 8 | CNN 2D sobre layout topográfico 3×5 | 7 features × posição | Mapa espacial ajuda? |
| 9 | CNN 2D sobre STFT | Sinal cru => espectrograma | Aprender filtros tempo-frequência |
| 10 | CNN 3D topográfica espaço-temporal-frequencial | STFT + topografia | Modelo end-to-end completo |

### MLP (Teste 1 -  MLP-Keras-v2)

```
Input (147)
 → Dense(128) → BN → ReLU → Dropout(0.3)
 → Dense(64)  → BN → ReLU → Dropout(0.2)
 → Dense(32)  →     ReLU → Dropout(0.1)
 → Dense(1, sigmoid)
```

- **Optimizer**: Adam (`lr=1e-3`)
- **Loss**: binary_crossentropy
- **Batch**: 64 ; **Épocas**: 40 fixas (sem EarlyStopping, val_loss é ruidoso com poucos sujeitos)
- **Pré-processamento**: z-score por sujeito + RobustScaler ajustado no treino
- **Regularização**: BatchNorm + Dropout
- **Estabilização**: ensemble de 3 sementes por fold

### CNN 2D sobre STFT (Teste 9)

Entrada: (F=18, T=9, C=21) - espectrogramas log-power por canal.

```
Conv2D(16, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.3)
Conv2D(32, 3×3) → BN → ReLU → GlobalAvgPool2D
Dense(32, ReLU) → Dropout(0.4) → Dense(1, sigmoid)
```

### CNN 3D topográfica (Teste 10)

Entrada: (rows=3, cols=5, F=18, T=9) - canais distribuídos em grid 3×5 que reflete posição no escalpo (linha de cima = FC, do meio = C, baixo = CP).

```
Conv3D(16) → MaxPool3D → Conv3D(32) → MaxPool3D
GlobalAvgPool3D → Dense(32) → Dropout(0.5) → Dense(1, sigmoid)
```

### Discussão MLP vs clássicos

A hipótese - *“o MLP terá desempenho similar a SVM/RF sobre as mesmas features”* foi confirmada. Em vários testes, Linear SVM e LDA superam o MLP, demonstrando que o problema é quase-linear no espaço de features escolhido com a quantidade de dados disponível.

---

## 8. Balanceamento

### Verificação do dataset

```
Classe T1 (y=0): 225 épocas (50,0 %)
Classe T2 (y=1): 225 épocas (50,0 %)
```

O dataset é perfeitamente balanceado => SMOTE/undersampling não foram aplicados no fluxo principal.

### Estudo controlado de balanceamento (Teste 3 do notebook DL)

Para fins didáticos, foi induzido um desbalanceamento artificial (≈ 80/20) e comparadas três estratégias com o mesmo MLP:

| Cenário | Acurácia (enganosa) | Balanced accuracy | Recall (T2 minoritária) | F1 |
|---|---:|---:|---:|---:|
| Sem correção | 0,702 | 0,510 | 0,155 | — |
| SMOTE | 0,612 | 0,576 | 0,375 (+0,220) | + |
| Undersampling | 0,562 | 0,547 | **0,581** (+0,426) | + |

### Discussão

- A acurácia bruta é enganosa em datasets desbalanceados (modelo prevê tudo como majoritária e ainda parece "boa").
- SMOTE e undersampling elevam balanced_accuracy e recall da minoritária ao custo de menos accuracy crua.
- AUC é independente de threshold e quase não muda entre as três estratégias.

  > **Conclusão;** como o dataset real é balanceado, o balanceamento não trouxe ganho no fluxo principal. Quando há desbalanceamento, SMOTE/RUS são importantes para evitar overtifing.

---

## 9. Combinação e Rejeição de Classificadores

### Combinação (Voting)

Implementada na seção final do Modelos_ML.ipynb:

| Ensemble | Modelos | Acurácia (GKF5) |
|---|---|---:|
| Hard Voting | LogReg + RandomForest + KNN | 0,555 +/- std |
| Soft Voting | LogReg + RandomForest + NaiveBayes | 0,546 +/- std |

**Resultado**: ensembles não trouxeram ganho real sobre os melhores individuais.

> **Por que o ensemble não ajudou?** Em BCI cross-subject o teto de performance é dominado pela heterogeneidade entre sujeitos (BCI literacy), não por instabilidade de modelo. Ensembles ajudam quando há decorrelação entre erros, e os 3 modelos topo erram nos mesmos sujeitos difíceis.

### Rejeição por limiar de confiança

Aplicada ao modelo top-1 (KNN + RMS + LDA-auto) - Comparativo_MLxDL.ipynb, seção H-2:

| Limiar | Cobertura | Acurácia (aceitos) | Nº rejeitados |
|---:|---:|---:|---:|
| 0,50 | 1,000 | 0,598 | 0 |
| 0,55 | ~0,80 | ~0,64 | ~90 |
| 0,60 | ~0,55 | ~0,68 | ~200 |
| 0,65 | ~0,30 | ~0,72 | ~315 |
| 0,70 | ~0,15 | ~0,76 | ~380 |
| 0,75 | ~0,06 | ~0,80 | ~420 |

### Trade-off

- Limiar maior => menos cobertura, mas as predições aceitas são mais corretas.
- Útil em BCI em tempo real quando se prefere "abster" do que errar (ex.: comando para cadeira de rodas onde um falso positivo é perigoso).
- Limiar = 0,60 é um bom compromisso (cobertura ≈ 55 %, accuracy ≈ 68 %).

---

## 10. Resultados Gerais

### Tabela Comparativa - Top 3 ML × Top 3 DL (`Comparativo_MLxDL.ipynb`, GroupKFold 5)

| Modelo | Representação | Acurácia | Balanced Acc | Precision | Recall | F1 | ROC-AUC | Gap (overfit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **ML1: KNN + RMS + LDA-auto** | clássica canal_com_rms | 0,598 ± std | 0,598 | 0,599 | 0,598 | 0,598 | 0,628 | +0,18 |
| **ML2: LogReg + RMS** | clássica canal_com_rms | 0,596 ± std | 0,596 | 0,597 | 0,596 | 0,596 | 0,640 | +0,16 |
| **ML3: SVM (RBF) + RMS + LDA** | clássica `canal_com_rms` | 0,582 ± std | 0,582 | 0,583 | 0,582 | 0,582 | 0,621 | +0,19 |
| **DL1: MLP (128, 64, 32)** | clássica canal_com_rms | 0,580 ± 0,103 | 0,580 | — | — | — | 0,607 | **+0,28** |
| **DL2: CNN 2D sobre STFT** | STFT log-power por canal | 0,551 | — | — | — | — | ~0,57 | +0,07 |
| **DL3: CNN 3D topográfica** | STFT + grid 3×5 topográfico | 0,516 | — | — | — | — | ~0,52 | +0,01 |

> **Acima de 0,5 = melhor que acado.** Os 3 melhores ML estão **virtualmente empatados** num patamar 0,58-0,60 o que é consistente com a literatura para MI cross-subject com poucos sujeitos.

### Gráfico comparativo (gerado no Comparativo_MLxDL.ipynb, seção H-3)

Barras agrupadas (accuracy / F1 / AUC) lado a lado para os 6 modelos. Linha tracejada cinza marca o nível de chance (0,50).

### Curvas ROC OOF combinadas (`Comparativo_MLxDL.ipynb`, seção H-1)

Os 3 melhores ML produzem ROCs **quase paralelas** com AUC ≈ 0,62–0,64. As CNNs ficam mais próximas da diagonal de chance.

### Matrizes de confusão (out-of-fold, N=450)

Por exemplo, ML1 (KNN+RMS+LDA-auto):

|  | Pred T1 | Pred T2 |
|---|---:|---:|
| Real T1 | TN ≈ 135 | FP ≈ 90 |
| Real T2 | FN ≈ 91 | TP ≈ 134 |

Erros simétricos (FP ≈ FN) => fronteira de decisão centrada, modelo trata as duas mãos de forma equivalente.

### Comparações pedidas

#### Com vs sem balanceamento

Não aplicável no dataset real (já balanceado). No estudo controlado (DL Teste 3): balanced_accuracy sobe de 0,510 para 0,576 (SMOTE) e recall(T2) de 0,155 para 0,581 (undersampling).

#### Com vs sem PCA

| Cenário | Acurácia (GKF5) |
|---|---:|
| LDA puro (sem PCA) | 0,556 |
| PCA(22) + LDA | **0,538** (–) |
| SelectKBest(50) + SVM | 0,622 (+) |

=> PCA prejudicou; SelectKBest (supervisionado) ajudou.

#### Modelos individuais vs ensemble

| | Acurácia |
|---|---:|
| Melhor individual (KNN+LDA-auto) | 0,598 |
| Hard Voting (LR+RF+KNN) | 0,555 |
| Soft Voting (LR+RF+NB) | 0,546 |

=> Ensembles não superaram o melhor individual.

---

## 11. Avaliação Experimental

### Estratégia

- **GroupKFold(5)** - 5 folds garantindo que nenhum sujeito apareça simultaneamente em treino e teste; cada fold tem ~90 épocas de teste.
- **Leave-One-Subject-Out (LOSO)** - 10 folds, 1 sujeito-teste por vez.
- **Sanity check de vazamento**: para cada fold, verifica-se via assert que set(groups[train]) ∩ set(groups[test]) = ∅.
- **Ensemble de sementes (apenas Keras)**: 3 sementes por fold para reduzir variância de inicialização.

> **Por que GroupKFold(5) e não LOSO no comparativo final?** Cada modelo é avaliado nas mesmas 5 partições - pré-requisito do teste de Friedman (medidas repetidas). LOSO inflaciona variância (n=45 ép./fold).

### Métricas

- **Accuracy:** proporção de acertos.
- **Balanced accuracy:** média do recall por classe (robusta a desbalanceamento).
- **Precision:** TP/(TP+FP).
- **Recall:** TP/(TP+FN).
- **F1-score:** média harmônica de precision e recall.
- **ROC-AUC:**  área sob a curva ROC; independente de threshold.

### Sanity checks

| Verificação | Resultado |
|---|---|
| **Dummy stratified** | acc ≈ 0,50 +/- std (consistente com chance) |
| **Permutation test** (n=50) sobre o melhor modelo | p < 0,05 => performance é estatisticamente acima do acaso |
| **Demo de vazamento**: StratifiedKFold (vazando) vs GroupKFold (correto) | inflação de +10 a +25 pp quando se ignora o agrupamento por sujeito - evidência do efeito de vazamento |

### Testes estatísticos entre os 6 melhores modelos (Comparativo_MLxDL.ipynb, seção G)

| Teste | Resultado | Interpretação |
|---|---|---|
| **Levene (homocedasticidade)** | p > 0,05 | Variâncias compatíveis |
| **ANOVA (accuracy)** | F = ?, p = 0,0315 | Rejeita H₀ - há diferença entre algum par |
| **Tukey HSD (accuracy)** | nenhum par é significativo a α=0,05 | Diferenças entre os 3 melhores ML não são significativas nos 5 folds |
| **Friedman (medidas repetidas)** | χ² = 9,88, p = 0,079 | Borderline; não rejeita H₀ a 5 % |
| **Nemenyi post-hoc** | ML clusterizam (rank 2,2–2,8) acima das CNNs e MLP (rank 3,9–4,8) | Clusters separáveis, mas pares individuais não significativos |
| **Wilcoxon pareado** (todos os pares) | confirma o padrão acima | — |

### Observações sobre estabilidade

- **Folds têm variância considerável** (desvios de até 0,10 em accuracy) - característica esperada de BCI cross-subject com poucos sujeitos.
- **Boxplot por teste** mostra que T2 e T5 (variantes de features clássicas) têm distribuição comparável a T3/T4 (CSP/FBCSP), Provando que, com poucos dados, boas características tipicas competem de igual para igual com redes espaciais.

---

## 12. Análise e Discussão

### Qual modelo teve melhor desempenho?

**KNN(k=7) + canal_com_rms + LDA(shrinkage='auto') - acurácia média 0,598 +/- std (GroupKFold 5)**.

LogReg(L2) sobre as mesmas features (sem LDA) ficou empatado em 0,596, o LDA ajuda principalmente o KNN (sensível à dimensionalidade), enquanto LogReg já é linear e regularizado.

### Qual representação foi superior?

**Features clássicas com normalização RMS por canal** (canal_com_rms):

- **Por que RMS ajudou?** Remove diferenças de magnitude absoluta entre sujeitos/sessões, isolando a forma espectral.
- **Por que canal e não média?** A topografia (qual canal carrega o sinal motor) é informativa, agregar pela média perde essa informação.
- **CSP (0,55) e FBCSP (0,55) ficaram abaixo** das clássicas com RMS, provavelmente porque o número pequeno de épocas/sujeito faz o CSP estimar covariâncias com ruído.

### PCA ajudou?

**Não.** PCA(22) reduziu a acurácia em ~2pp em relação ao LDA puro. O eixo de máxima variância não coincide com o eixo de máxima separação entre classes neste problema. SelectKBest (supervisionado) foi a única redução que ajudou no varrimento inicial (+6,7 pp).

### Balanceamento ajudou?

**Não no fluxo principal** (dataset já balanceado). Em cenário induzido, SMOTE e undersampling elevam balanced_accuracy e recall da minoritária sem alterar AUC.

### A RNA (MLP) foi competitiva?

**Sim, mas não superou os clássicos**: MLP(128, 64, 32) atingiu acc ≈ 0,580 - essencialmente empatado com o top-3 ML, mas com overfit severo (gap treino-teste de +0,28 vs +0,16–0,19 dos ML). O problema é quase-linear no espaço de features escolhido.

### Ensembles trouxeram ganho?

**Não.** Hard Voting = 0,555 e Soft Voting = 0,546 - ambos abaixo dos individuais. Os 3 modelos topo erram nos mesmos sujeitos difíceis (BCI illiterate), então não há decorrelação a explorar.

### Rejeição melhorou confiabilidade?

**Sim, com trade-off**: limiar 0,60 => cobertura 55 % com accuracy 68 % nas predições aceitas. Útil em aplicações online onde abster é melhor que errar.

### Análise eletrofisiológica complementar

#### Gap LOSO vs GroupKFold(5)

- gap global ≈ +0,02 a +0,05 => moderado: variabilidade entre sujeitos esperada em EEG motor, transferência possível com perda controlada.

#### Heterogeneidade entre sujeitos (BCI literacy)

Triagem within-subject com Riemann + LDA (limiar AUC ≥ 0,60):
- **5 sujeitos "BCI literate"** (within-AUC ≥ 0,60)
- **5 sujeitos "BCI illiterate"** (within-AUC < 0,60)
- Correlação **Spearman ρ = +0,91** entre within-AUC e LOSO-AUC quando esse sujeito é teste - confirma que a triagem within-subject prediz o quanto cada sujeito é decodificável cross-subject.

#### Simetria dos erros

Erros simétricos (FP ≈ FN) na maioria dos modelos top-3 => fronteira de decisão centrada, sem viés para uma das mãos.

---

## 13. Conclusão

### Principais achados

1. **Top 3 modelos** (ML clássicos com features handcrafted + RMS + LDA) atingem **0,58–0,60 acc** cross-subject — patamar consistente com a literatura para MI com 10 sujeitos.
2. **Features clássicas bem desenhadas competem com filtros espaciais supervisionados** (CSP, FBCSP) neste regime de N pequeno; RMS por canal foi a melhor variante.
3. **Deep Learning não superou Machine Learning clássico** — MLP empatou com os ML (com mais overfit), CNNs ficaram abaixo. Esperado em datasets pequenos (<1k épocas, 10 sujeitos).
4. **PCA prejudicou; LDA ajudou KNN; SelectKBest ajudou pontualmente.**
5. **Balanceamento e ensembles não ajudaram** no fluxo principal — dataset balanceado, modelos correlacionados.
6. **Rejeição** trouxe trade-off explorável (cobertura ↓ → accuracy ↑).
7. **Heterogeneidade entre sujeitos (BCI literacy) é o principal limitador** — within-AUC prediz LOSO-AUC com ρ=0,91.
8. **Significância estatística**: ANOVA marginalmente significativa (p=0,031), mas Tukey HSD não detectou diferenças par a par — o poder estatístico com 5 folds é limitado.

### Trabalhos futuros

- **Aumentar o dataset** (109 sujeitos disponíveis no EEGBCI; usar todos).
- **Calibração individual (within-subject)** — atinge AUC > 0,80 nos sujeitos *literate*.
- **Transfer learning**: Euclidean Alignment, Riemannian alignment, fine-tuning de modelos pré-treinados (ChronoNet, EEGNet preentreined).
- **EEGNet, ShallowConvNet, Conformers EEG** — arquiteturas DL especializadas em EEG (mais data-efficient que CNN genérica).
- **Incluir mais features**: conexões funcionais (PLV, coerência), microstates, fractal/complexity (Higuchi, Lempel-Ziv).
- **Explorar imagética bilateral** (runs 6, 10, 14) e **pés vs mãos** (runs 5, 9, 13) para tarefas multiclasse.

---

## 14. Reprodutibilidade

### Instalação

```bash
pip install -r requirements.txt
```

`requirements.txt` cobre todo o stack instalado em `Preparação_Ambiente.ipynb`:
- numpy, pandas, scipy, matplotlib, seaborn
- mne, mne-features, mne-icalabel, python-picard
- scikit-learn, xgboost, joblib, imbalanced-learn
- tensorflow, keras
- pyriemann
- shap, optuna, keras-tuner
- plotly, statsmodels, bokeh
- ipywidgets, tqdm, ipykernel
- boto3, botocore (download do dataset)

### Execução (ordem recomendada)

```bash
# 1. Preparação do ambiente (uma vez)
jupyter notebook src/notebooks/Preparação_Ambiente.ipynb

# 2. Download do dataset (10 sujeitos com seed=42)
jupyter notebook src/notebooks/Download_Datatset_10_Sujeitos.ipynb

# 3. Geração de metadados e exploração
jupyter notebook src/notebooks/Exploração_Datatset.ipynb

# 4. Pré-processamento completo (gera /preprocessed_v2)
jupyter notebook src/notebooks/Preprocessamento_Geral.ipynb

# 5. Modelos clássicos (Testes 1–5)
jupyter notebook src/notebooks/Modelos_ML.ipynb

# 6. Redes neurais (Testes 1–10)
jupyter notebook src/notebooks/Modelos_Deep_learning.ipynb

# 7. Comparativo final ML × DL com testes estatísticos
jupyter notebook src/notebooks/Comparativo_MLxDL.ipynb
```

### Sementes

Todos os RNGs usam `SEED = 42`:
- `numpy.random.seed(42)`
- `random.seed(42)`
- `tf.random.set_seed(42)` / `keras.utils.set_random_seed(42)`
- `os.environ['PYTHONHASHSEED'] = '42'`
- `random_state=42` em `RandomForestClassifier`, `LogisticRegression`, `SVC`, `SGDClassifier`, `MLPClassifier`, `GroupShuffleSplit`, etc.

> **Nota**: TensorFlow com GPU pode ter pequena variância residual nos resultados Keras. Testes ML (sklearn) são **bit-perfect reproduzíveis**.

### Estrutura do projeto

```
mini-projeto2/
│
├── README.md                              ← este arquivo
├── requirements.txt
├── src/notebooks/
│   ├── Preparação_Ambiente.ipynb
│   ├── Download_Datatset_10_Sujeitos.ipynb
│   ├── Exploração_Datatset.ipynb
│   ├── Preprocessamento_Geral.ipynb
│   ├── Modelos_ML.ipynb
│   ├── Modelos_Deep_learning.ipynb
│   └── Comparativo_MLxDL.ipynb
├── results/
│   ├── tables/                            ← tabelas .csv das métricas
│   ├── figures/                           ← gráficos (PNG/SVG)
│   ├── confusion_matrices/                ← matrizes de confusão dos 6 modelos
│   ├── learning_curves/                   ← curvas SGD/RF e learning_curve sklearn
│   └── statistical_tests/                 ← outputs ANOVA, Tukey, Friedman, Nemenyi
└── experiments/
    ├── data_eegmmidb/                     ← dados raw (10 sujeitos)
    └── preprocessed_v2/
        ├── per_run/
        ├── per_subject/
        ├── manifest.csv
        └── ml_cache/                      ← caches .pkl de features, OOF, learning curves
```

### Caches gerados

| Arquivo | Conteúdo |
|---|---|
| `X_feat_classical_v1.pkl` | Features clássicas (Teste 1) |
| `X_feat_canal_com_rms.pkl` | Variante RMS por canal |
| `X_feat_media_*.pkl` | Variantes médias (com/sem RMS) |
| `X_stft.pkl` | Tensor STFT `(N, C, F, T)` para CNNs |
| `mlp_cv_results_v1.pkl` | Resultados CV do MLP (com histórias por fold) |
| `mp2_top6_oof_cache.pkl` | OOF preds + scores dos 6 modelos finais |
| `mp2_top6_learning_curves.pkl` | Learning curves dos 4 modelos sklearn |

---

## 15. Autor

- **Nome**: Tiago Souto Rocha
- **Vínculo**: Graduando em Psicologia (UEPB) e Pesquisador em Neurociência Computacional (NUTES - grupo NeuroComp)
- **Áreas**: Brain decoding, processamento de sinais EEG/fMRI, IA.

---

## Observação Final

Todos os resultados apresentados neste README estão disponíveis na pasta `/results` e podem ser reproduzidos executando os notebooks na ordem indicada na seção 14. Os caches em `experiments/preprocessed_v2/ml_cache/` permitem **rerodar apenas os blocos de visualização e análise** sem recomputar features ou retreinar modelos (economia de horas de CPU/GPU).

> **Reflexão metodológica**: o teto de ~0,60 de acurácia cross-subject **não é uma falha** do pipeline — é uma característica fundamental do problema de MI cross-subject com poucos sujeitos. O valor científico deste projeto está em (i) demonstrar **rigor metodológico** (anti-vazamento, GroupKFold, permutation tests, sanity checks); (ii) **mapear sistematicamente** o espaço de representações × modelos × redutores × balanceamento × ensembles × rejeição; e (iii) **identificar honestamente** que o gargalo está na **heterogeneidade individual** (BCI literacy), não na escolha do modelo.
