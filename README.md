# Chat ENEM - Assistente Virtual para Estudantes do ENEM

## Autor
Carlos Alberto Santos de Souza; Mestrando em Ciências da Computação 2025.1; Orientador: Glauco Carneiro.

## Sobre o Projeto
Este projeto é uma iniciativa desenvolvida como parte do trabalho de mestrado para a disciplina de Tópicos Especiais Em Engenharia de Software I da Universidade Federal de Sergipe no ano de 2025.

O Chat ENEM é um assistente virtual projetado para apoiar estudantes a aumentarem seu rendimento nas provas do ENEM. Utilizando tecnologias modernas de Inteligência Artificial e Processamento de Linguagem Natural, o sistema oferece um ambiente interativo onde os estudantes podem fazer perguntas, receber questões personalizadas e obter explicações detalhadas sobre os diferentes conteúdos do exame.

## Funcionalidades do Projeto

### 1. Upload e Processamento de Documentos (RN001)
- Permite que os usuários façam upload de documentos em formato PDF ou DOCX
- Processa automaticamente os documentos enviados
- Converte o conteúdo em embeddings para busca semântica
- Armazena as informações em um banco de dados vetorial (FAISS)
- Permite consultas contextuais sobre o conteúdo dos documentos

### 2. Agente Recomendador de Questões (RN002)
O sistema utiliza uma arquitetura de agentes inteligentes composta por:

- **Recommender Agent**: Agente principal que analisa as perguntas dos usuários e as direciona para o Agent Item mais apropriado
- **Agent Items Especializados**: Quatro agentes especializados, cada um dedicado a uma área específica do ENEM:
  - Agent Item 1: Linguagens, Códigos e suas Tecnologias
  - Agent Item 2: Ciências Humanas e suas Tecnologias
  - Agent Item 3: Ciências da Natureza e suas Tecnologias
  - Agent Item 4: Matemática e suas Tecnologias

Cada Agent Item possui seu próprio dataset de questões e pode fornecer respostas contextualizadas baseadas em sua área de especialidade.

## Como Executar em sua Máquina Local

### Pré-requisitos
- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/chat-enem.git
cd chat-enem
```

2. **Crie e ative um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # No macOS/Linux
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
Crie um arquivo `.env` na raiz do projeto e adicione suas chaves de API:
```plaintext
OPENAI_API_KEY=sua_chave_da_openai
HUGGINGFACEHUB_API_TOKEN=sua_chave_do_huggingface
```

5. **Execute a aplicação**
```bash
streamlit run chat_enem.py
```

A aplicação estará disponível em `http://localhost:8501`

### Uso da Aplicação

1. Acesse a interface web através do seu navegador
2. Use o painel lateral para fazer upload de documentos (opcional)
3. Digite suas perguntas na caixa de texto principal
4. O sistema identificará automaticamente a área de conhecimento e fornecerá respostas relevantes

### Estrutura de Dados Necessária

O sistema espera encontrar arquivos JSON com questões do ENEM nas seguintes pastas:
- `data_agent_1/` - Questões de Linguagens
- `data_agent_2/` - Questões de Ciências Humanas
- `data_agent_3/` - Questões de Ciências da Natureza
- `data_agent_4/` - Questões de Matemática

Cada arquivo JSON deve seguir o formato:
```json
{
    "enunciado": "Texto da questão",
    "label": "área de conhecimento",
    "cor_prova": "COR"
}
```

## Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
