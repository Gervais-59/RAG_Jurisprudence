"""
PHASE 7 — INTERFACE STREAMLIT (version déployable)
====================================================
Pourquoi cette étape : un livrable démontrable. Le README dit "voici les
scores" ; l'interface prouve "voici comment ça marche".

Architecture : réutilise EXACTEMENT le pipeline des phases 4-5 (retrieval
hybride + routeur + Groq 70B). Streamlit n'apporte que l'habillage.

Différence avec la version notebook :
  - L'index ChromaDB est RECONSTRUIT automatiquement au premier démarrage
    s'il est absent (cas du déploiement Streamlit Cloud, où l'index n'est
    pas versionné — trop lourd pour git).
  - La clé Groq est lue via st.secrets EN PRIORITÉ (Streamlit Cloud), avec
    fallback sur .env (développement local).

Lancement (terminal PyCharm) :
    streamlit run app.py
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import chromadb
import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================ CONFIG
load_dotenv()

RACINE = Path(__file__).parent
CHUNKS = RACINE / "corpus" / "chunks.json"
INDEX_DIR = RACINE / "corpus" / "index_chroma"
MODELE_EMB = "intfloat/multilingual-e5-base"
MODELE_LLM = "llama-3.3-70b-versatile"


def get_groq_key():
    """Récupère la clé Groq : secrets Streamlit Cloud en priorité, sinon .env local.

    Pourquoi ce fallback : en développement local, la clé vient de .env
    (chargé par python-dotenv). En déploiement Streamlit Cloud, elle vient
    de st.secrets (onglet Secrets de l'interface Cloud). Le même code
    fonctionne dans les deux environnements sans modification.
    """
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ["GROQ_API_KEY"]


# ============================================================ CHARGEMENT (une seule fois)
@st.cache_resource(show_spinner="Chargement des modèles et de l'index…")
def initialiser():
    """..."""
    # === 1. Extraction du ZIP en PREMIER (avant tout appel réseau)
    import zipfile
    zip_path = RACINE / "corpus" / "index_chroma.zip"
    if not INDEX_DIR.exists() and zip_path.exists():
        st.info("Extraction de l'index vectoriel...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RACINE / "corpus")

    # === 2. Authentification HuggingFace (débloque le téléchargement du modèle)
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except (KeyError, FileNotFoundError):
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token  # nom alternatif que certaines versions utilisent

    # === 3. Chargement du modèle (peut télécharger si absent en cache)
    modele = SentenceTransformer(MODELE_EMB)

    # === 4. Suite normale de l'initialisation
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    passages = [
        f"{c['reference']}, Art. {c['article']} — {c['contenu']}"
        for c in chunks
    ]

    # 3. ChromaDB : ouverture ou reconstruction selon présence de l'index
    client_chroma = chromadb.PersistentClient(path=str(INDEX_DIR))
    noms_collections = [c.name for c in client_chroma.list_collections()]

    if "droit_ivoirien" not in noms_collections:
        # Index absent → reconstruction complète (cas du déploiement Streamlit Cloud)
        st.info("🔨 Premier démarrage : construction de l'index vectoriel "
                "(~2 min, une seule fois)…")
        embeddings = modele.encode(
            [f"passage: {p}" for p in passages],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        collection = client_chroma.create_collection(
            name="droit_ivoirien",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings.tolist(),
            documents=passages,
            metadatas=[{
                "section": c["section"],
                "statut": c["statut"],
                "texte_parent": c["texte_parent"] or "",
                "reference": c["reference"],
                "article": c["article"],
            } for c in chunks],
        )
    else:
        # Index déjà construit sur disque → simple ouverture
        collection = client_chroma.get_collection("droit_ivoirien")

    # 4. Index BM25 (rapide, en mémoire, refait à chaque démarrage)
    tokeniser = lambda t: re.findall(r"\d+\.\d+|\w+", t.lower())
    bm25 = BM25Okapi([tokeniser(p) for p in passages])
    ids_ordre = [c["id"] for c in chunks]

    # 5. Client LLM
    client_llm = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=get_groq_key(),
    )
    return modele, collection, bm25, ids_ordre, tokeniser, client_llm


modele, collection, bm25, ids_ordre, tokeniser, client_llm = initialiser()


# ============================================================ RETRIEVAL
def recherche_hybride(question, k=8, k_moteur=20, K_rrf=60):
    """Retrieval hybride dense + BM25, fusion RRF (identique à la Phase 4)."""
    q_emb = modele.encode([f"query: {question}"], normalize_embeddings=True)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=k_moteur)
    rangs_dense = {id_: r for r, id_ in enumerate(res["ids"][0])}

    scores_bm25 = bm25.get_scores(tokeniser(question))
    top_bm25 = sorted(range(len(scores_bm25)), key=lambda i: -scores_bm25[i])[:k_moteur]
    rangs_bm25 = {ids_ordre[i]: r for r, i in enumerate(top_bm25)}

    candidats = set(rangs_dense) | set(rangs_bm25)
    scores = {
        id_: sum(1 / (K_rrf + r[id_]) for r in (rangs_dense, rangs_bm25) if id_ in r)
        for id_ in candidats
    }
    tops = sorted(scores, key=scores.get, reverse=True)[:k]

    docs = collection.get(ids=tops)
    par_id = {i: (d, m) for i, d, m in zip(docs["ids"], docs["documents"], docs["metadatas"])}
    return [(id_, scores[id_], par_id[id_][0], par_id[id_][1]) for id_ in tops]


RE_REF = re.compile(r"article\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
INDICES_SECTION = {
    "code du travail": "loi",
    "convention collective": "convention",
    "convention": "convention",
}


def recherche(question, k=8, max_exacts=3):
    """Routeur de références exactes + hybride (identique à la Phase 4d)."""
    resultats, deja = [], set()
    m = RE_REF.search(question)
    if m:
        q_low = question.lower()
        section = next((s for ind, s in INDICES_SECTION.items() if ind in q_low), None)
        where = ({"$and": [{"article": {"$eq": m.group(1)}}, {"section": {"$eq": section}}]}
                 if section else {"article": {"$eq": m.group(1)}})
        exacts = collection.get(where=where, include=["documents", "metadatas", "embeddings"])
        n = len(exacts["ids"])
        if n > 0:
            if n > max_exacts:
                q_emb = modele.encode([f"query: {question}"], normalize_embeddings=True)[0]
                sims = np.array(exacts["embeddings"]) @ q_emb
                ordre = np.argsort(-sims)[:max_exacts]
            else:
                ordre = range(n)
            for i in ordre:
                resultats.append((exacts["ids"][i], 1.0,
                                  exacts["documents"][i], exacts["metadatas"][i]))
                deja.add(exacts["ids"][i])
    for id_, s, doc, meta in recherche_hybride(question, k=k):
        if id_ not in deja and len(resultats) < k:
            resultats.append((id_, s, doc, meta))
    return resultats[:k]


# ============================================================ GENERATION
SYSTEM = """Tu es un assistant documentaire spécialisé en droit du travail ivoirien.

RÈGLES IMPÉRATIVES :
1. Tu réponds UNIQUEMENT à partir des extraits fournis. Tu n'utilises JAMAIS tes connaissances générales.
2. Chaque affirmation cite sa source entre crochets : [Code du travail, Art. 14.5] ou [Décret n°96-195, Art. 2].
3. Si les extraits ne permettent pas de répondre, tu réponds EXACTEMENT :
   "Les textes dont je dispose ne me permettent pas de répondre à cette question."
4. Si plusieurs textes se complètent (loi + décret + convention), présente-les dans cet ordre hiérarchique.
5. Termine TOUJOURS par : "⚠️ Information documentaire — ne constitue pas un conseil juridique. Consultez un professionnel du droit pour votre situation."

Réponds de façon concise (5 à 15 lignes, sauf question à réponse hiérarchisée).
6. Si une question est posée dans une langue compatible au modèle de langue dans projet, tu réponds exactement dans cette langue.
"""


def generer_reponse(question, passages):
    """Appelle Groq 70B avec le prompt de fidélité stricte."""
    contexte = "\n\n".join(f"[{i+1}] {doc}" for i, (_, _, doc, _) in enumerate(passages))
    message = f"EXTRAITS :\n\n{contexte}\n\nQUESTION : {question}"
    r = client_llm.chat.completions.create(
        model=MODELE_LLM,
        temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": message}],
    )
    return r.choices[0].message.content


# ============================================================ INTERFACE
st.set_page_config(page_title="Assistant juridique — Droit du travail ivoirien", page_icon="⚖️")

st.title("⚖️ Assistant juridique")
st.caption("Droit du travail ivoirien — Code, décrets, convention collective")

with st.expander("ℹ️ Comment ça marche"):
    st.markdown("""
Cet assistant s'appuie sur un corpus de **1 093 articles** (Code du travail 2015 plus une mise à jour avec les derniers textes législatifs et réglementaires en matière du droit de travail ivoirien, 27 décrets, arrêtés, convention collective). 
Chaque réponse cite ses sources et le système s'abstient s'il ne trouve pas la réponse dans les textes.

**Exemples de questions à essayer :**
- Quelle est la durée maximale de la période d'essai ?
- Une femme enceinte peut-elle être licenciée ?
- Que dit l'article 14.5 du Code du travail ?
- Comment se calcule l'indemnité de licenciement ?
    """)

question = st.text_input(
    "Votre question",
    placeholder="Ex : Combien de jours de congés par mois ?",
)

if question:
    with st.spinner("🔎 Recherche dans le corpus…"):
        passages = recherche(question, k=8)

    with st.spinner("✍️ Rédaction de la réponse en cours"):
        reponse = generer_reponse(question, passages)

    st.markdown("### Réponse")
    st.markdown(reponse)

    with st.expander(f"📚 {len(passages)} passages consultés"):
        for i, (id_, s, doc, meta) in enumerate(passages, start=1):
            marque = "🎯 " if s == 1.0 else ""
            st.markdown(f"**{marque}[{i}] {meta['reference']}, Art. {meta['article']}**")
            st.caption(doc[:300] + ("…" if len(doc) > 300 else ""))
            st.markdown("---")

st.markdown("---")
st.caption("⚠️ Information documentaire uniquement. Ne constitue pas un conseil juridique. "
           "Consultez un professionnel du droit pour toute situation réelle.")