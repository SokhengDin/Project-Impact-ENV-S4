# Analyse de l'Impact Sociétal des Articles ICML

Analyse automatique des déclarations d'impact sociétal dans **406 articles** de recherche en IA publiés aux conférences **ICML 2023, 2024 et 2025**, à l'aide d'un pipeline LLM + RAG.

---

## Comment ça marche

Les articles (PDF) sont placés dans `papers/{année}/`. Le pipeline s'exécute en trois étapes principales :

### Étape 1 — Extraction du texte d'impact (`extract_impact_section`)

Pour chaque PDF, le système utilise **PyMuPDF** (`fitz`) pour parcourir le document bloc par bloc et détecter une section d'impact sociétal via des expressions régulières. Les titres recherchés incluent :

- *Broader Impact*, *Societal Impact*, *Impact Statement*
- *Ethics Statement*, *Ethical Considerations*
- *Risks and Limitations*, *Broader Context*

Lorsqu'un titre correspondant est trouvé, les 5 blocs de texte suivants sont capturés.

**Si aucune section explicite n'est trouvée**, le pipeline active un mécanisme de repli en deux niveaux :

1. **RAG (Retrieval-Augmented Generation)** — Le PDF est découpé en chunks de 1000 caractères (avec chevauchement de 200) via `RecursiveCharacterTextSplitter`. Les chunks contenant des mots-clés liés à l'impact sont indexés dans un vecteur store **FAISS** avec des embeddings OpenAI. Une recherche de similarité sémantique extrait les 6 passages les plus pertinents.

2. **Texte complet** — En dernier recours, l'intégralité du texte du PDF est envoyée au modèle.

---

### Étape 2 — Analyse par LLM (`analyse_paper`)

Le texte extrait est transmis à **GPT-4o-mini** via une chaîne LangChain (`ChatPromptTemplate | llm.with_structured_output`). Le modèle répond à **4 questions** avec une sortie structurée enforced par un schéma Pydantic (`ImpactAnalysis`) :

| Question | Description |
|----------|-------------|
| **Q1** | La déclaration d'impact est-elle un texte générique/par défaut sans signification concrète ? |
| **Q2** | L'article décrit-il un impact sociétal ou environnemental **positif** ? |
| **Q3** | L'article décrit-il un impact sociétal ou environnemental **négatif** ? |
| **Q4** | Les impacts positifs l'emportent-ils sur les négatifs ? |

Le modèle produit également des listes concrètes d'exemples positifs et négatifs extraits du texte.

---

### Étape 3 — Sauvegarde (`analyse_folder`)

- Chaque article génère un fichier `results/{année}/{article}.json`
- Un résumé par année est écrit dans `results/{année}/summary.json`
- Un résumé global est agrégé dans `results/summary.json`
- Les résultats déjà traités sont ignorés (reprise possible sans recalcul)
- Une API **FastAPI** expose les résultats via `/data` et sert le tableau de bord interactif sur `/`

---

## Architecture du projet

```
projet/
├── main.py              # Pipeline complet + API FastAPI
├── index.html           # Tableau de bord de visualisation (charts, KPIs)
├── papers/
│   ├── 2023/            # PDFs ICML 2023
│   ├── 2024/            # PDFs ICML 2024
│   └── 2025/            # PDFs ICML 2025
├── results/
│   ├── 2023/summary.json
│   ├── 2024/summary.json
│   ├── 2025/summary.json
│   └── summary.json     # Résumé global (406 articles ICML 2023–2025)
└── pyproject.toml
```

---

## Installation et lancement

### Prérequis

- Python 3.12+
- Une clé API OpenAI dans un fichier `.env` :
  ```
  OPENAI_API_KEY=sk-...
  ```

> **Alternative open-source (sans clé API)** — Il est possible de remplacer GPT-4o-mini par un modèle local via [Ollama](https://ollama.com). Dans `main.py`, décommenter les lignes `ChatOllama` / `OllamaEmbeddings` et commenter les lignes OpenAI. Modèles compatibles :
> - `qwen3.5:4b`, `qwen3:8b` — Qwen3 (Alibaba)
> - `gemma3:4b`, `gemma3:12b` — Gemma 3 (Google)
> - `gemma4:12b` — Gemma 4 (Google, dernière génération)
>
> ```bash
> ollama pull qwen3.5:4b
> ollama pull gemma4:12b
> ```

**Windows :**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS :**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Installer les dépendances et lancer :**
```bash
uv sync
source .venv/bin/activate
fastapi run main.py --port 8000
```

Ouvrir ensuite [http://localhost:8000](http://localhost:8000)

---

### Lancer l'analyse

Via la ligne de commande :

```bash
python main.py papers/ --output results/
```

Via l'API (serveur démarré) :

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"papers": "papers", "output": "results"}'
```

---

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET`   | `/`   | Tableau de bord interactif |
| `GET`   | `/data` | Résultats JSON globaux |
| `POST`  | `/run` | Lancer l'analyse des PDFs |

---

## Technologies utilisées

| Composant | Outil |
|-----------|-------|
| Extraction PDF | PyMuPDF (`fitz`) |
| Découpage texte | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vecteur store | FAISS |
| LLM | GPT-4o-mini (OpenAI) |
| Orchestration | LangChain |
| Sortie structurée | Pydantic + `with_structured_output` |
| API | FastAPI |
