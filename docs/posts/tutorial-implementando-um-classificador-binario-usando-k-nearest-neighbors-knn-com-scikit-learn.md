# Tutorial: Implementando um Classificador Binário usando K-Nearest Neighbors (KNN) com Scikit-Learn

Neste tutorial, iremos explorar a implementação de um classificador binário utilizando o algoritmo de K-Vizinhos Mais Próximos (KNN) a partir do pacote Scikit-Learn. Iremos passar por todos os passos necessários, desde a conceitualização até a avaliação do classificador, usando um conjunto de dados simulado de comentários sobre o atendimento ao cliente de uma loja virtual de eletrônicos.

## 1. Conceitualização

### Classificação de Dados
A classificação de dados é uma tarefa de aprendizado de máquina onde o objetivo é atribuir rótulos (classes) a dados com base em características específicas.

### Classificação Binária
A classificação binária é um tipo de classificação onde os dados são divididos em duas classes distintas. Neste tutorial, vamos classificar comentários como "positivos" ou "negativos".

### Classificação com K-Vizinhos Mais Próximos (KNN)
O algoritmo KNN classifica um ponto de dados com base na classe majoritária dos seus K vizinhos mais próximos. Ele calcula a distância entre pontos de dados para encontrar os vizinhos mais próximos e toma uma decisão com base nas classes desses vizinhos.

## 2. Criação da Amostra de Dados

Vamos criar uma amostra de dados simulada com comentários sobre atendimento ao cliente de uma loja virtual de eletrônicos. A coluna "textos" contém os textos brutos dos comentários, e a coluna "label" contém as classes (positivo ou negativo).

```python
import pandas as pd
import random

random.seed(42)

data = {
    "textos": [
        "O atendimento foi excelente, estou muito satisfeito!",
        "Péssimo serviço ao cliente, não recomendo.",
        "Demoraram para responder às minhas perguntas.",
        # ... (outros comentários simulados)
    ],
    "label": ["POSITIVO", "NEGATIVO", "NEGATIVO", ...]
}

df = pd.DataFrame(data)
```

## 3. Pré-processamento de Dados

Antes de treinar nosso classificador, precisamos pré-processar os dados para remover ruídos, acentuações, stopwords, etc.

```python
import re
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Função para pré-processamento de texto
def preprocess_text(text):
    text = text.lower()  # Converter para minúsculas
    text = re.sub(r'\d+', '', text)  # Remover números
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remover pontuações
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])  # Remover stopwords
    return text

# Aplicar pré-processamento aos textos
df['textos'] = df['textos'].apply(preprocess_text)

# Dividir o dataset em treino e teste (estratificado)
X_train, X_test, y_train, y_test = train_test_split(df['textos'], df['label'], test_size=0.25, stratify=df['label'], random_state=42)
```

## 4. Extração de Características com TF-IDF

Vamos usar a abordagem TF-IDF (Term Frequency-Inverse Document Frequency) para vetorizar os textos e extrair características relevantes.

```python
# Criar um vetor TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limitando a 1000 características
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

## 5. Treinamento e Avaliação do Classificador KNN

Agora, vamos treinar o classificador KNN e avaliá-lo usando métricas como acurácia, F1-macro e F1-micro.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Treinar o classificador KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_tfidf, y_train)

# Prever classes para o conjunto de teste
y_pred = knn_classifier.predict(X_test_tfidf)

# Avaliar o classificador
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

print(f"Acurácia: {accuracy:.2f}")
print(f"F1-Macro: {f1_macro:.2f}")
print(f"F1-Micro: {f1_micro:.2f}")
```

## 6. Análise e Considerações Finais

A abordagem com TF-IDF demonstrou ser eficaz na extração de características relevantes dos textos, permitindo ao classificador KNN fazer previsões razoavelmente boas. No entanto, essa abordagem tem algumas limitações, como a falta de compreensão semântica das palavras.

Em futuras melhorias, consideraríamos:

- **Word Embeddings**: Utilização de word embeddings pré-treinados, como Word2Vec ou GloVe, para capturar relações semânticas entre as palavras.

- **Modelos de Linguagem**: Explorar modelos de linguagem poderosos, como BERT, que capturam contextos complexos e significados em frases.

- **Ajuste de Parâmetros**: Experimentar diferentes valores para os parâmetros do KNN e do vetorizador TF-IDF para melhorar ainda mais o desempenho.

- **Tratamento de Dados Desbalanceados**: Caso o conjunto de dados apresente classes desbalanceadas, aplicar técnicas para lidar com esse desafio.

Neste tutorial, você aprendeu como implementar um classificador binário usando o KNN com o pacote Scikit-Learn, desde a criação dos dados simulados até a avaliação do modelo. Além disso, exploramos considerações para melhorias futuras, visando abordagens mais avançadas de processamento de linguagem natural.
