
---

# **Assignment** **3**

**A Text-Based Recommender System with LLMs**

Course: Advanced Analytics in Business [D0S07a] — KU Leuven

## **1.** **Introduction** **and** **Objective**

The objective of this assignment was to build an LLM-driven game recommender system for the Steam platform. The starting point was a scaffold Flask application (`recommender.py`) that returned random games regardless of the user’s query. Our task was to replace this dummy logic with a functional Retrieval-Augmented Generation (RAG) pipeline that understands natural-language queries, retrieves relevant games from a database of 5,000 Steam titles, and generates **human-readable recommendations.

The data consists of a SQLite database (`steam_games_reviews_25.sqlite`) containing two tables: `games` (metadata such as name, genres, tags, description, price, and release date) and `reviews` (up to 500 English user reviews per game). Only games with more than 25 reviews are included.

## **2. Architecture** **Overview**

Our recommender system follows a four-stage RAG pipeline. Each stage corresponds to a function in the original scaffold and maps to specific course concepts:

| **Step** | **Function**        | **Method**                                                                    | **Course Reference**                                     |
| -------------- | ------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 1              | `retrieve_candidates()` | Hybrid BM25 + dense embedding retrieval, merged via Reciprocal Rank Fusion (top-30) | BM25: 07-DL2, slides 13–14. Embeddings: 07-DL2, slides 23–39 |
| 2              | `rank_candidates()`     | Cross-encoder reranking with min-max normalized scores (top-10)                     | Transformers/self-attention: 07-DL2, slides 75–82             |
| 3              | `generate_answer()`     | Local LLM generation via Ollama with review-augmented context                       | GPT architecture: 07-DL2, slides 88–90                        |
| 4              | Review integration        | Player review summaries injected into embeddings and LLM prompt                     | Data enrichment (beyond base slides)                           |


---

The overall pattern—retrieving relevant context and feeding it to a generative model—is known as **Retrieval-****Augmented Generation (RAG)**. While RAG as an explicit pattern was not covered in the course slides, every underlying building block (BM25, embeddings, transformers, generative pretrained models) is drawn directly from the *Deep Learning II: Text and Recurrency* lecture (07-DeepLearning2).

## **3. Technology Stack**

| **Component** | **Technology**                         | **Purpose**                                         |
| ------------------- | -------------------------------------------- | --------------------------------------------------------- |
| Embedding model     | `all-MiniLM-L6-v2` (sentence-transformers) | Encode games and queries as 384-dim vectors               |
| Vector database     | `ChromaDB` (persistent, local)             | Store and retrieve game embeddings by cosine similarity   |
| Sparse retrieval    | `BM25Okapi` (rank-bm25)                    | Keyword-based retrieval for exact-match and genre queries |
| Reranker            | `ms-marco-MiniLM-L-6-v2` (cross-encoder)   | Precise pairwise relevance scoring of query–game pairs   |
| LLM                 | `gemma2:2b` via Ollama (local)             | Natural-languagerecommendation generation                 |
| Web framework       | `Flask`                                    | API backend + web frontend (unchanged from scaffold)      |
| Database            | SQLite                                       | Game metadata + user reviews storage                      |


---

## **4. Component Details**

### **4.1 Offline** **Indexing**

Before serving any queries, we run a one-time indexing step (`build_index.py`). This script loads all 5,000 games and, for each game, constructs a **game_text** string combining the game’s name, genres, tags, short description, and a review summary. The review summary is built by selecting the top 3 most helpful positive reviews and the top 1 most helpful negative review per game, truncated to 150 words each. This enriched text is then embedded using` all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection.

**Index statistics:** 5,000 games indexed; 4,994 with review data; average review summary length of 2,307 characters; total build time approximately 46 seconds.

Course link: The embedding step applies the word embedding concept from slides 23–39 (07-DL2) at document level. As the slides state: “the goal is to construct a dense vector of real values” where “distance metrics can be used to define a notion of relatedness” (slide 23). We use cosine similarity as the distance metric, matching the course’s recommendation.

### **4.2** **Hybrid** **Retrieval (BM25 +** **Dense**)

Our retrieval stage combines two complementary approaches. **Dense retrieval** encodes the user query with the same embedding model and queries ChromaDB for the 30 nearest neighbours by cosine distance. This excels at semantic matching: a query like “relaxing game to unwind after work” can find games described as “peaceful” or “cozy” even without exact word overlap.

**Sparse retrieval** uses BM25Okapi, a classical information retrieval ranking function. The BM25 index is built once at startup over the same game_text corpus. BM25 scores documents based on term frequency, inverse document frequency, and document length normalization (slides 13–14, 07-DL2). This catches exact keyword matches that dense retrieval might miss. For example, specific game titles like “Counter-Strike” or niche genre terms like “roguelike deckbuilder.”

Both retrieval lists (each top-30) are merged using **Reciprocal Rank Fusion (RRF)**, a simple yet effective fusion technique: for each game appearing in either list, its fused score is the sum of 1/(k + rank) across all lists where it appears, with k = 60. The top-30 from the fused ranking proceed to reranking.

Course link: The slides explicitly mention that BM25 “sometimes even works better than embedding models” (slide 14, 07-DL2). Our empirical results confirm this for keyword-heavy queries, while dense retrieval dominates for semantic queries. Hybrid retrieval captures both modes.

### **4.3 Cross-Encoder** **Reranking**

The 30 candidates from retrieval are passed through a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`). Unlike the bi-encoder used for retrieval—which encodes query and game separately, the cross-encoder processes the (query, game_text) pair jointly through a transformer, allowing full self-attention between query tokens and game tokens. This yields much more precise relevance scores at the cost of higher latency. 

The raw cross-encoder outputs are logits that can be negative and vary widely in magnitude. We apply **min-max normalization** within each batch to scale scores to a 0–10 range, ensuring the best match scores approximately 10 and the weakest approximately 0. The top 10 are retained and passed to the generation stage.

Course link: The cross-encoder is a direct application of the transformer self-attention mechanism (slides 79–82, 07-DL2). The slide states: “for each token, the model asks: what other tokens should I look at?” By feeding both query and game text as a single input sequence, the cross-encoder lets query tokens attend to game description tokens, enabling fine-grained semantic matching.

### 4.4 LLM-Based **Answer** **Generation**

The top-ranked games are formatted into a structured prompt for `gemma2:2b`, a 2-billion parameter language model running locally via Ollama. The prompt includes the game name, genres, short description, and a condensed player feedback summary for each candidate. The LLM is instructed to select the 3–5 best matches and explain why they fit, referencing both game metadata and player opinions.

Using a local model via Ollama eliminates the need for cloud API access, avoids rate limits and costs, and ensures reproducibility. The `gemma2:2b` model can be swapped for alternatives such as `phi3.5` or `llama3.2:3b` by changing a single line.

Course link: The LLM is a generative pretrained transformer (slide 88, 07-DL2), trained on text completion given a prompt. By constraining the LLM’s context to only the retrieved games, we mitigate hallucination: the model can only recommend games that exist in our database.

### **4.5 Review Integration**

The assignment explicitly encouraged incorporating the `reviews` table. We integrate reviews at two levels:

* **Embedding enrichment:** Each game’s embedding text includes a review summary (top-3 positive, top-1 negative) so that player sentiment is encoded into the vector representation. This means retrieval is influenced by what players actually say, not just the store description.
* **Prompt enrichment:** The LLM prompt includes player feedback snippets for each candidate, enabling the generated answer to reference real player opinions (e.g., “players praise the relaxing gameplay” or “reviewers note the steep learning curve”).

Reviews are selected using a SQL window function query sorted by `votes_up` (helpfulness) and filtered by language (English only). Each review is truncated to 150 words to keep the total embedding text within the model’s 512-token context limit.

## **5. Files** **Modified**

| File                     | **Changes**                                                                                                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_index.py`                  | New file. Loads games, builds review summaries, embeds game texts, and stores them in a persistent ChromaDB collection.                                                   |
| `recommender.py`                  | Fully rewritten. Implements hybrid BM25+dense retrieval, cross-encoder reranking with min-max normalization, and Ollama LLM generation with **review-**augmented prompts. |
| `steam_sqlite.py`                 | Added get_review_summaries() function using SQL window functions for efficient review extraction.                                                                         |
| `pyproject.toml`                  | Added sentence-transformers and chromadb as dependencies. Ollama and rank-bm25 were already present.                                                                      |
| `app.py`                          | Unchanged. The Flask API and frontend were left intact.                                                                                                                   |


---

## **6.** **Results**

### **6.1 Retrieval Source Analysis**

To demonstrate the value of hybrid retrieval, we tested three queries that stress different retrieval modes:

| **Query**             | **Example** **Result** | **Source** | **Observation**                                     |
| --------------------------- | ---------------------------------- | ---------------- | --------------------------------------------------------- |
| “Counter-Strike”          | Counter-Strike: Source             | BOTH             | Exact name match → both systems agree                    |
| “Counter-Strike”          | Counter-Strike Nexon               | BM25 only        | Dense missed this variant; BM25 caught it via keyword     |
| “roguelikedeckbuilder”    | Guild of Dungeoneering             | Dense only       | No keyword match; semantic similarity rescued it          |
| “roguelikedeckbuilder”    | Roguelands / Overture              | BM25 only        | Keyword “rogue” matched; dense would not retrieve these |
| “relaxinggame after work” | LOOP: A Tranquil Puzzle Game       | Dense only       | Pure semantic query; embeddings dominate                  |


---

Keyobservation**:** BM25 adds the most value for exact-name queries and genre keyword queries, while embeddings dominate for purely semantic queries. Reciprocal Rank Fusion handles the transition gracefully without manual tuning. This aligns with the course material’s note that BM25 and embeddings have complementary strengths (slide 14, 07-DL2).

## **7.** **Discussion**

### **7.1** **Strengths**

* **Full RAG pipeline:** The system implements all three stages (retrieve, rerank, generate) rather than a simpler **direct-**embedding approach.**
* **Hybrid retrieval:** Combining BM25 with dense retrieval improves recall across different query types, as demonstrated empirically.
* **Review grounding:** Incorporating player reviews makes recommendations more trustworthy and allows the LLM to reference real player experiences.
* **Fully local:** All models run locally via Ollama and sentence-transformers, requiring no cloud APIs, no API keys, and no costs.

### **7.2** **Limitations** **and** **Future** **Work**

* **Model size:** The gemma2:2b model occasionally produces generic or repetitive answers. A larger model (e.g., llama3.2:3b or phi3.5) could improve generation quality.
* **Structured query parsing:** The system does not extract structured constraints (price, year, platform) from the user query. An LLM-based query parser could enable hard filters before retrieval, improving precision for queries like “cheap indie games from 2024.”
* **Embedding context window:** The all-MiniLM-L6-v2 model truncates input at 256 tokens. Games with long descriptions and review summaries may lose information. A model with a larger context (e.g., all-mpnet-base-v2 with 384 tokens) could help.
* **Formal evaluation:** We did not implement automated evaluation metrics (e.g., overlap with Steam’s **“similar games” lists). A future improvement would be to benchmark against Steam’s own recommendations or ChatGPT’s suggestions on identical queries.*

## **8.** **Conclusion**

We replaced the scaffold’s random game selection with a complete RAG pipeline that combines hybrid retrieval (BM25 + dense embeddings), cross-encoder reranking, review-augmented context, and local LLM generation. The system understands natural-language queries, retrieves semantically relevant games, and produces human-readable explanations grounded in both game metadata and player feedback. All components run locally without cloud dependencies, making the system fully reproducible

The architecture directly applies concepts from the course material: BM25 for sparse ranking (slides 13–14), word/document embeddings for dense representation (slides 23–39), transformer self-attention for precise reranking (slides 79–82), and generative pretrained models for natural-language output (slides 88–90)—all from the *Deep Learning II: Text and Recurrency lecture*
