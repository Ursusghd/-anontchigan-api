# 🩷 ANONTCHIGAN API — Chatbot Béninois pour la Sensibilisation au Cancer du Sein 🇧🇯

**ANONTCHIGAN** est une application de chatbot intelligente développée avec **Streamlit**, **FAISS** et **Sentence Transformers**, spécialisée dans la sensibilisation au **cancer du sein** au Bénin 🇧🇯.
Elle combine la recherche sémantique (RAG) et la génération de texte via **Llama 3.1 (Groq)**, avec une personnalité 100% béninoise : chaleureuse, naturelle et éducative.

---

## 🚀 Fonctionnalités

* 🔍 Recherche de similarité avec **FAISS**
* ⚡ Génération ultra-rapide avec **Groq (Llama 3.1)**
* 💬 Style de réponse **authentiquement béninois**
* 🧠 Données médicales fiables (issues de `cancer_sein.json`)
* 🔒 Gestion sécurisée des clés API via `.env`
* 🌐 Compatible avec tout front-end (React, Vue, etc.)

---

## 🧰 Installation

### 1️⃣ Cloner le dépôt

```bash
git clone  https://github.com/Ursusghd/-anontchigan-api
cd anontchigan-api
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Créer un fichier `.env`

Crée un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=ta_cle_api_groq
```

👉 Pour obtenir ta clé : [https://console.groq.com](https://console.groq.com)

---

## ▶️ Lancer l'application

```bash
streamlit run app.py
```

Application disponible sur :
➡️ [http://localhost:8501](http://localhost:8501)

Mode API (via URL) :
➡️ [http://localhost:8501/?api=true&question=Votre+question](http://localhost:8501/?api=true&question=Votre+question)

---

## 💬 Exemple d’utilisation

### Interface Web
Ouvrez simplement votre navigateur et chattez avec ANONTCHIGAN ! 💬

### Mode API (GET)
```bash
curl "http://localhost:8501/?api=true&question=Quels+sont+les+signes+du+cancer+du+sein"
```

Réponse typique 👇

```json
{
  "success": true,
  "answer": "Les signes incluent une bosse dans le sein, un écoulement anormal du mamelon, des changements de la peau... 💗",
  "method": "json_direct",
  "similarity_score": 0.85,
  "user_id": "user_1234"
}
```

---

## 📁 Structure du projet

```
anontchigan-api/
│
├── app.py                 # Code principal Streamlit
├── cancer_sein.json       # Base de connaissances
├── requirements.txt       # Dépendances Python
├── .env                   # Configuration d'environnement (à créer)
├── .env.example           # Exemple de configuration
├── .gitignore             # Fichiers à ignorer par Git
├── Procfile               # Configuration déploiement
└── README.md              # Documentation du projet
```

---

## 🧑🏽‍💻 Auteur

**Projet ANONTCHIGAN 💗**
Développé par   Judicaël Karol DOBOEVI, 
                Hornel Ursus GBAGUIDI, 
                Abel Kocou KPOKOUTA, 
                Josaphat ADJELE

Membres du **Club IA ENSGMM 🇧🇯**

---

## ⚖️ Licence

Ce projet est distribué sous licence **MIT** — vous pouvez l’utiliser, le modifier et le partager librement.

---

> 💡 *« La connaissance, c’est la prévention. Et la prévention, c’est la vie ! »* — ANONTCHIGAN 🌸🇧🇯
