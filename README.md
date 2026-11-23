# ESCO Skill Matcher

Upload a resume (PDF/DOCX/TXT) or paste skills; they are matched to ESCO skills (`skills_en.csv`).

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**How it works**
1. First use of an embedding model embeds all ~100k ESCO labels (preferred + alternative) and saves them to `.cache/vector_stores/`. This is one-time per model (long on CPU, fast on GPU).
2. Every later run only embeds the candidate's phrases and does a matrix search against the stored vectors (~ms).
3. Optional cross-encoder rerank of the top 15 candidates per phrase; final score = 0.7 * rerank + 0.3 * embedding similarity.

Code lives in `skillmatch/` (config, esco, embedder, store, extract, matcher); `app.py` is the UI.
