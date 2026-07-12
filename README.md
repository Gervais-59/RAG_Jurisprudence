# ⚖️ Assistant juridique RAG — Droit du travail ivoirien

> Un assistant documentaire qui répond à des questions en langage naturel sur le droit du travail ivoirien, avec citations d'articles vérifiables et abstention explicite hors du corpus.

**Corpus indexé** : Code du travail (loi n°2015-532), 27 décrets d'application, 3 arrêtés, 1 ordonnance, 2 lois annexées (traite des enfants, délais), Convention collective interprofessionnelle du 19 juillet 1977 — soit **516 pages, 1 093 articles** structurés en chunks citables.

---

## 🎯 Le problème

Le droit ivoirien est numériquement peu doté : textes dispersés, jurisprudence difficile d'accès, aucun dataset propre, aucun modèle spécialisé. Un simple LLM généraliste, questionné sur ce droit, hallucine des règles empruntées au droit français ou invente des chiffres. Pour un domaine où la citation exacte fait la valeur du produit, c'est rédhibitoire.

## 💡 La solution

Un système **RAG** (Retrieval-Augmented Generation) qui sépare la connaissance (dans le corpus contrôlé) du rédacteur (LLM). Chaque question déclenche : (1) une recherche des passages pertinents, (2) leur transmission au LLM avec des règles strictes, (3) une réponse sourcée article par article — avec obligation de s'abstenir quand la réponse n'est pas dans le corpus.

---

## 🏗️ Architecture

Chaîne de traitement en 6 étages, chaque composant justifié par un test qui l'a rendu nécessaire :

| Étage | Rôle | Choix technique |
|---|---|---|
| 1. Extraction | PDF 516 pages → texte propre | `pdfplumber` (après comparaison avec pypdf/PyMuPDF) |
| 2. Chunking structurel | Texte → 1 093 articles + métadonnées | Regex multi-format (`Art. X.Y`, `ARTICLE PREMIER`...) |
| 3. Indexation dense | Chunks → vecteurs 768d | `intfloat/multilingual-e5-base`, ChromaDB |
| 4. Indexation lexicale | Chunks → index BM25 | `rank-bm25`, tokens nombres-pointés |
| 5. Retrieval hybride | Question → 8 passages | Fusion RRF + routeur de références exactes |
| 6. Génération | Passages → réponse sourcée | Llama 3.3 70B via Groq (API OpenAI-compatible) |

---

## 📊 Résultats mesurés

Évaluation sur un jeu de 15 questions annotées à la main (5 couvertes, 5 difficiles, 5 hors-corpus) :

| Métrique | Score | Interprétation |
|---|---|---|
| **Recall@8** — questions couvertes | **100%** (5/5) | L'article pertinent est toujours dans les 8 passages transmis au LLM |
| **Recall@8** — questions difficiles | **80%** (4/5) | 4/5 — l'échec restant appelle une reformulation LLM (voir *Limites*) |
| **Recall@1** — global | **60%** | Le meilleur passage est souvent le bon dès le premier rang |
| **Abstention** — hors-corpus prononcée | **5/5** | La phrase d'abstention exacte est produite |
| **Abstention** — hors-corpus pure | **5/5** | Aucune tergiversation ni hallucination avant l'abstention |
| **Fidélité** des réponses | **6/10 fidèles**, 4/10 approximatives, **0/10 fausses** | Zéro invention, approximations liées majoritairement au retrieval |
| **Latence** | **1 à 4 s / réponse** | Compatible avec un usage interactif |

**Comparaison de LLM sur les mêmes 3 questions de test** :

| Modèle | Latence | Fidélité | Abstention |
|---|---|---|---|
| Llama 3.2 3B (local, Ollama) | ~5 min/réponse | Correcte | Hallucine des taux inventés sur la TVA |
| Llama 3.1 8B (local, Ollama) | ~8 min/réponse | S'abstient à tort quand la réponse est en position 1 | Correcte |
| **Llama 3.3 70B (Groq)** | **~2 s/réponse** | **Nuancée, hiérarchisée, complète** | **Parfaite** |

Enseignement transposable : dans un RAG, le LLM n'est pas interchangeable — un même retrieval donne des réponses de qualité radicalement différente selon le rédacteur. Le compromis coût/qualité/confidentialité est un vrai choix d'architecture.

---

## 🔧 Décisions de design (les choix non triviaux et pourquoi)

### 1. Chunking structurel par article

**Pourquoi** : le découpage naïf par blocs de N caractères est le défaut des tutoriels. Un texte juridique a une unité naturelle — l'article — qui est à la fois sémantiquement cohérente et directement citable. Résultat : chunks de 250-860 caractères médians selon la section, chacun accompagné de son texte parent (décret d'origine) pour une citation exacte comme *« Art. 2, Décret n°96-195 du 7 mars 1996 »*.

### 2. Recherche hybride (dense + BM25 + routeur)

**Pourquoi trois moteurs** : une batterie de tests a révélé trois familles d'échecs du retrieval dense seul :
- « Article 14.5 » remontait 14.4, 14.1, 14.2 — les identifiants exacts sont l'angle mort des embeddings.
- « Virer sans préavis » ne remontait aucun article pertinent dans le top 10 — l'argot est absent du corpus.
- « TVA » (hors-corpus) obtenait une similarité *supérieure* à des questions légitimes — la similarité ne détecte pas le hors-sujet.

**Remèdes** : BM25 pour les termes exacts, routeur avec consultation directe par métadonnées pour les références (« article X.Y »), désambiguïsation sémantique quand plusieurs textes ont un « article 2 » (44 candidats dans le corpus).

### 3. Enrichissement des passages avec leur référence

Chaque passage indexé est préfixé de sa référence : `« Code du travail, Art. 14.5 — [contenu] »`. La référence devient du texte indexable : BM25 matche « 14.5 » littéralement, le dense gagne le contexte du texte d'appartenance.

### 4. Métadonnées honnêtes contre citations mensongères

Le corpus contient des lois annexées après le code (loi 2010-272 sur le travail des enfants, loi 96-670 sur les délais) dont les articles étaient initialement étiquetés « Code du travail ». Correction : reconnaissance de « LOI N° … » comme texte parent, champ `reference` calculé au chunking. Sur un outil juridique, citer le mauvais texte est le péché capital.

### 5. Abstention côté LLM, pas côté similarité

Un seuil sur les scores de similarité aurait rejeté des questions légitimes et accepté la TVA (score plus élevé). L'abstention est donc confiée au LLM, avec une phrase d'abstention exacte imposée par le system prompt. Mesuré : 5/5 sur hors-corpus, pas de faux négatifs sur les questions couvertes.

---

## ⚠️ Limites connues et améliorations futures

Un projet portfolio honnête nomme ses limites — et donne le remède pour chacune :

| Limite constatée | Cause | Amélioration prioritaire |
|---|---|---|
| Question « télétravail encadré ? » sort du top 10 | Question abstraite, articles concrets → mots-clés ne matchent pas | Reformulation de la question par le LLM avant retrieval |
| Recall@1 à 60% (mais R@8 à 100%) | Le système trouve, mais classe imparfaitement | Ajout d'un re-ranker cross-encoder sur le top 20 |
| Réponse au calcul de l'indemnité de licenciement basée sur un décret abrogé (96-201) | Le corpus contient l'ancien ET le nouveau texte sans hiérarchie temporelle | Enrichir les chunks avec un champ `en_vigueur` / `abroge_par` |
| Registre argotique partiellement problématique | Le corpus n'a pas ce registre | Reformulation LLM (même remède que la limite 1) |

**Hors périmètre V1 mais souhaitable** : élargissement à la jurisprudence de la CCJA, aux Actes uniformes OHADA, à la Constitution ivoirienne.

**Avertissement d'usage** : cet outil fournit de l'information documentaire et non un conseil juridique. Les textes indexés sont ceux publiés jusqu'à 2022 — vérifier les évolutions récentes avant toute décision. Consulter un professionnel du droit pour toute situation réelle.

---

## 📁 Structure du dépôt