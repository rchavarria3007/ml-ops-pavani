# TRABALHO DE CONCLUSÃO DE DISCIPLINA

## [cite_start]Machine Learning Aplicado: HR Analytics Challenge [cite: 2]

[cite_start]**Disciplina**: Data Science Experience [cite: 3]
[cite_start]**Professor**: Matheus H. P. Pacheco [cite: 4]
[cite_start]**Data de Entrega**: 17/07/2025 [cite: 5]
[cite_start]**Valor**: 10 pontos [cite: 6]

## [cite_start]CONTEXTO DO PROBLEMA [cite: 7]

[cite_start]A TechCorp Brasil, uma das maiores empresas de tecnologia do país com mais de 50.000 funcionários, está enfrentando um problema crítico: sua taxa de *attrition* (rotatividade de funcionários) aumentou 35% no último ano, gerando custos estimados em R$ 45 milhões. [cite: 8]

[cite_start]Cada funcionário que deixa a empresa representa não apenas custos de demissão e contratação (estimados em 1,5x o salário anual), mas também: [cite: 9]
* [cite_start]Perda de conhecimento institucional [cite: 9]
* [cite_start]Impacto na produtividade das equipes [cite: 9]
* [cite_start]Diminuição da moral dos colaboradores [cite: 9]
* [cite_start]Atrasos em projetos críticos [cite: 9]

[cite_start]Você foi contratado como Cientista de Dados para desenvolver um sistema preditivo que identifique funcionários com alto risco de deixar a empresa, permitindo que o RH tome ações preventivas. [cite: 10]

## [cite_start]OBJETIVO DO TRABALHO [cite: 11]

[cite_start]Desenvolver um pipeline completo de Machine Learning para prever *attrition* de funcionários, demonstrando domínio das técnicas aprendidas na disciplina e criatividade na solução do problema. [cite: 12]

### [cite_start]Entregáveis Obrigatórios: [cite: 13]
1.  [cite_start]Código Python completo e documentado (Jupyter Notebook ou scripts .py) [cite: 14]
2.  [cite_start]Relatório técnico (10-15 páginas) detalhando toda a solução [cite: 15]
3.  [cite_start]Dashboard interativo ou visualizações que comuniquem os resultados [cite: 16]

## [cite_start]SOBRE O DATASET [cite: 18]

[cite_start]O dataset fornecido contém informações de 1 milhão de funcionários (sintético baseado no IBM HR Analytics) com 35 variáveis: [cite: 19]

### [cite_start]Variáveis Disponíveis: [cite: 20]
* [cite_start]**Demográficas**: Age, Gender, MaritalStatus, Education, EducationField [cite: 21]
* [cite_start]**Profissionais**: Department, JobRole, JobLevel, JobInvolvement, Years At Company [cite: 21]
* [cite_start]**Compensação**: MonthlyIncome, Percent Salary Hike, StockOptionLevel [cite: 22]
* [cite_start]**Satisfação**: JobSatisfaction, Environment Satisfaction, RelationshipSatisfaction [cite: 23]
* [cite_start]**Work-Life**: OverTime, WorkLifeBalance, Business Travel, DistanceFrom Home [cite: 24]
* [cite_start]**Performance**: Performance Rating, Training TimesLast Year [cite: 25]
* [cite_start]**Target**: Attrition (Yes/No) [cite: 26]

[cite_start]**IMPORTANTE**: O dataset é altamente desbalanceado (~16% attrition) [cite: 27]

## [cite_start]CRITÉRIOS DE AVALIAÇÃO [cite: 28]

### [cite_start]1. Análise Exploratória (2 pontos) [cite: 29]
* [cite_start]Análise estatística completa das variáveis [cite: 30]
* [cite_start]Identificação de padrões e correlações [cite: 31]
* [cite_start]Visualizações criativas e informativas [cite: 32]
* [cite_start]Insights de negócio relevantes [cite: 33]
* [cite_start]Tratamento de dados faltantes/outliers [cite: 35]

### [cite_start]2. Feature Engineering (2 pontos) [cite: 36]
* [cite_start]Criação de no mínimo 10 novas features [cite: 37]
* [cite_start]Justificativa técnica e de negócio para cada feature [cite: 39]
* [cite_start]Análise do impacto das novas features [cite: 41]
* [cite_start]Uso de técnicas avançadas (polynomial features, embeddings, etc.) [cite: 43]

### [cite_start]3. Modelagem (2 pontos) [cite: 44]
* [cite_start]Implementação de pelo menos 4 algoritmos diferentes [cite: 45]
* [cite_start]Tratamento adequado do desbalanceamento [cite: 46]
* [cite_start]Otimização de hiperparâmetros (Grid/Random Search, Bayesian, etc.) [cite: 47]
* [cite_start]Validação cruzada apropriada [cite: 48]
* [cite_start]Análise de ensemble methods [cite: 49]

### [cite_start]4. Avaliação e Interpretação (2 pontos) [cite: 51]
* [cite_start]Métricas apropriadas para desbalanceamento [cite: 52]
* [cite_start]Análise de erro detalhada [cite: 53]
* [cite_start]Análise de viés e fairness [cite: 54]
* [cite_start]Recomendações de threshold ótimo [cite: 55]

### [cite_start]5. Implementação e Comunicação (2 pontos) [cite: 56]
* [cite_start]Código limpo e bem documentado [cite: 57]
* [cite_start]Pipeline reproduzível [cite: 58, 61]
* [cite_start]Visualizações profissionais [cite: 59, 62]
* [cite_start]Comunicação clara dos resultados [cite: 60, 63]
* [cite_start]Proposta de implementação em produção [cite: 64]

## [cite_start]DESAFIOS EXTRAS (Pontos Bônus) [cite: 65]

### [cite_start]Desafio: Deployment (3 pontos) [cite: 66]
[cite_start]Crie uma API REST ou aplicação web que permita: [cite: 67]
* [cite_start]Upload de dados de novos funcionários [cite: 67]
* [cite_start]Predição em tempo real [cite: 67]
* [cite_start]Dashboard de monitoramento [cite: 67]
* [cite_start]Sistema de alertas [cite: 67]

## [cite_start]DICAS E RECURSOS [cite: 68]

### [cite_start]Bibliotecas Recomendadas: [cite: 69]

[cite_start]**Essenciais** [cite: 70]
```python
[cite_start]import pandas as pd [cite: 71]
[cite_start]import numpy as np [cite: 72]
[cite_start]import matplotlib.pyplot as plt [cite: 73]
[cite_start]import seaborn as sns [cite: 74]
