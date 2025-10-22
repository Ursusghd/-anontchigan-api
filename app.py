import json
import os
import logging
from typing import Dict, List, Optional
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from streamlit.web import cli as stcli
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ANONTCHIGAN API",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ANONTCHIGAN")

class Config:
    """Configuration optimisée"""
    SIMILARITY_THRESHOLD = 0.75
    MAX_HISTORY_LENGTH = 8
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 600
    FAISS_RESULTS_COUNT = 3
    MIN_ANSWER_LENGTH = 30

# ============================================
# SERVICE GROQ
# ============================================

class GroqService:
    def __init__(self):
        self.client = None
        self.available = False
        self._initialize_groq()
    
    def _initialize_groq(self):
        try:
            from groq import Groq
            
            # Essayer d'abord st.secrets (Streamlit Cloud), puis .env (local)
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except:
                api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                logger.warning("Clé API Groq manquante - Veuillez configurer GROQ_API_KEY dans Streamlit Secrets ou le fichier .env")
                return
            
            self.client = Groq(api_key=api_key)
            
            # Test de connexion
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="llama-3.1-8b-instant",
                max_tokens=5,
            )
            self.available = True
            logger.info("✓ Service Groq initialisé")
            
        except Exception as e:
            logger.warning(f"Service Groq non disponible: {str(e)}")
    
    def generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        """Génère une réponse complète sans coupure"""
        if not self.available:
            raise RuntimeError("Service Groq non disponible")
        
        try:
            context_short = self._prepare_context(context)
            messages = self._prepare_messages(question, context_short, history)
            
            logger.info("🤖 Génération avec Groq...")
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                top_p=0.9,
            )
            
            answer = response.choices[0].message.content.strip()
            answer = self._clean_response(answer)
            
            if not self._is_valid_answer(answer):
                raise ValueError("Réponse trop courte")
                
            answer = self._ensure_complete_response(answer)
            
            logger.info(f"✓ Réponse générée ({len(answer)} caractères)")
            return answer
            
        except Exception as e:
            logger.error(f"Erreur Groq: {str(e)}")
            raise
    
    def _prepare_context(self, context: str) -> str:
        lines = context.split('\n')[:5]
        context_short = '\n'.join(lines)
        if len(context_short) > Config.MAX_CONTEXT_LENGTH:
            context_short = context_short[:Config.MAX_CONTEXT_LENGTH-3] + "..."
        return context_short
    
    def _prepare_messages(self, question: str, context: str, history: List[Dict]) -> List[Dict]:
        system_prompt = f"""Tu es ANONTCHIGAN, assistante IA professionnelle spécialisée dans la sensibilisation au cancer du sein au Bénin.

CONTEXTE À UTILISER :
{context}

RÈGLES CRITIQUES :
1. FOURNIR DES RÉPONSES COMPLÈTES - ne JAMAIS couper une phrase en milieu de mot
2. Si tu donnes un conseil de prévention, aller deux fois à la ligne AVANT de le donner
3. Tes réponses doivent se terminer naturellement par un point final
4. Tes créateurs sont Judicaël Karol DOBOEVI, Ursus Hornel GBAGUIDI, Abel Kokou KPOCOUTA et Josaphat ADJELE, tous membres du club d'IA de l'ENSGMM (Ecole Nationale Supérieure de Génie Mathématique et Modélisation) d'Abomey.

STYLE :
- Professionnel, clair, empathique
- Réponses directes sans formules introductives
- CONCIS mais COMPLET
- Humour humain léger 1 fois sur 3 réponses
- Emojis à utiliser fréquemment : 💗 🌸 😊 🇧🇯

STRUCTURE DES RÉPONSES :
1. Réponse principale basée sur le contexte
2. N'invente PAS d'informations
3. Si le contexte ne couvre pas la question, recommande de consulter un professionnel de santé au Bénin
4. ENSGMM se définit comme l'Ecole Nationale Supérieure de Génie Mathématique et Modélisation
5. Utilise des connecteurs logiques pour fluidifier et produire des réponses naturelles et cohérentes

DIRECTIVES ANTI-COUPURE :
- Vérifie que ta réponse est complète avant de terminer
- Ne coupe PAS en milieu de phrase ou de mot
- Utilise "Atassa!" ou "Atassaaaaa!" en début de phrase en cas d'humour et d'étonnement extrême
- Termine par un point final approprié
- Si tu mentionnes des noms (créateurs, etc.), assure-toi qu'ils sont COMPLETS

Conseils de prévention : seulement si pertinents et si demandés."""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history[-4:]:
            messages.append(msg)
        
        messages.append({
            "role": "user", 
            "content": f"QUESTION: {question}\n\nIMPORTANT : Réponds de façon COMPLÈTE sans couper ta réponse. Termine par un point final. Si conseil de prévention, va à la ligne avant."
        })
        
        return messages
    
    def _clean_response(self, answer: str) -> str:
        unwanted_intros = [
            'bonjour', 'salut', 'coucou', 'hello', 'akwè', 'yo', 'bonsoir', 'hi',
            'excellente question', 'je suis ravi', 'permettez-moi', 'tout d abord',
            'premièrement', 'pour commencer', 'en tant qu', 'je suis anontchigan'
        ]
        
        answer_lower = answer.lower()
        for phrase in unwanted_intros:
            if answer_lower.startswith(phrase):
                sentences = answer.split('.')
                if len(sentences) > 1:
                    answer = '.'.join(sentences[1:]).strip()
                    if answer:
                        answer = answer[0].upper() + answer[1:]
                break
        
        return answer.strip()
    
    def _is_valid_answer(self, answer: str) -> bool:
        return (len(answer) >= Config.MIN_ANSWER_LENGTH and 
                not answer.lower().startswith(('je ne sais pas', 'désolé', 'sorry')))
    
    def _ensure_complete_response(self, answer: str) -> str:
        if not answer:
            return answer
            
        cut_indicators = [
            answer.endswith('...'),
            answer.endswith(','),
            answer.endswith(';'),
            answer.endswith(' '),
            any(word in answer.lower() for word in ['http', 'www.', '.com']),
            '...' in answer[-10:]
        ]
        
        if any(cut_indicators):
            logger.warning("⚠️  Détection possible de réponse coupée")
            
            last_period = answer.rfind('.')
            last_exclamation = answer.rfind('!')
            last_question = answer.rfind('?')
            
            sentence_end = max(last_period, last_exclamation, last_question)
            
            if sentence_end > 0 and sentence_end >= len(answer) - 5:
                answer = answer[:sentence_end + 1]
            else:
                answer = answer.rstrip(' ,;...')
                if not answer.endswith(('.', '!', '?')):
                    answer += '.'
        
        prevention_phrases = [
            'conseil de prévention',
            'pour prévenir',
            'je recommande',
            'il est important de',
            'n oubliez pas de'
        ]
        
        has_prevention_advice = any(phrase in answer.lower() for phrase in prevention_phrases)
        
        if has_prevention_advice:
            lines = answer.split('. ')
            if len(lines) > 1:
                for i, line in enumerate(lines[1:], 1):
                    if any(phrase in line.lower() for phrase in prevention_phrases):
                        lines[i] = '\n' + lines[i]
                        answer = '. '.join(lines)
                        break
        
        return answer

# ============================================
# SERVICE RAG
# ============================================

class RAGService:
    def __init__(self, data_file: str = 'cancer_sein.json'):
        self.questions_data = []
        self.embedding_model = None
        self.index = None
        self.embeddings = None
        self._load_data(data_file)
        self._initialize_embeddings()
    
    def _load_data(self, data_file: str):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                self.questions_data.append({
                    'question_originale': item['question'],
                    'question_normalisee': item['question'].lower().strip(),
                    'answer': item['answer']
                })
            
            logger.info(f"✓ {len(self.questions_data)} questions chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        try:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'
            
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            all_texts = [
                f"Q: {item['question_originale']} R: {item['answer']}"
                for item in self.questions_data
            ]
            
            self.embeddings = self.embedding_model.encode(all_texts, show_progress_bar=False)
            self.embeddings = np.array(self.embeddings).astype('float32')
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            
            logger.info(f"✓ Index FAISS créé ({len(self.embeddings)} vecteurs)")
            
        except Exception as e:
            logger.error(f"Erreur initialisation embeddings: {str(e)}")
            raise
    
    def search(self, query: str, k: int = Config.FAISS_RESULTS_COUNT) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.questions_data):
                    similarity = 1 / (1 + distances[0][i])
                    results.append({
                        'question': self.questions_data[idx]['question_originale'],
                        'answer': self.questions_data[idx]['answer'],
                        'similarity': similarity,
                        'distance': distances[0][i]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur recherche FAISS: {str(e)}")
            return []

# ============================================
# FONCTION DE TRAITEMENT DES QUESTIONS
# ============================================

def process_question(question: str, history: List[Dict], groq_service, rag_service):
    """Traite une question et retourne la réponse"""
    
    # Salutations
    salutations = ["cc", "bonjour", "salut", "coucou", "hello", "akwe", "yo", "bonsoir", "hi"]
    question_lower = question.lower().strip()
    
    if any(salut == question_lower for salut in salutations):
        responses = [
            "Je suis ANONTCHIGAN, assistante pour la sensibilisation au cancer du sein. Comment puis-je vous aider ? 💗",
            "Bonjour ! Je suis ANONTCHIGAN. Que souhaitez-vous savoir sur le cancer du sein ? 🌸",
            "ANONTCHIGAN à votre service. Posez-moi vos questions sur la prévention du cancer du sein. 😊"
        ]
        return {
            "answer": random.choice(responses),
            "method": "salutation",
            "score": None
        }
    
    # Recherche FAISS
    logger.info("🔍 Recherche FAISS...")
    faiss_results = rag_service.search(question)
    
    if not faiss_results:
        return {
            "answer": "Les informations disponibles ne couvrent pas ce point spécifique. Je vous recommande de consulter un professionnel de santé au Bénin pour des conseils adaptés. 💗",
            "method": "no_result",
            "score": None
        }
    
    best_result = faiss_results[0]
    similarity = best_result['similarity']
    
    logger.info(f"📊 Meilleure similarité: {similarity:.3f}")
    
    # Décision : Réponse directe vs Génération
    if similarity >= Config.SIMILARITY_THRESHOLD:
        logger.info(f"✅ Haute similarité → Réponse directe")
        answer = best_result['answer']
        
        if len(answer) > Config.MAX_ANSWER_LENGTH:
            answer = answer[:Config.MAX_ANSWER_LENGTH-3] + "..."
        
        return {
            "answer": answer,
            "method": "json_direct",
            "score": float(similarity)
        }
    
    else:
        logger.info(f"🤖 Similarité modérée → Génération Groq")
        
        # Préparer le contexte
        context_parts = []
        for i, result in enumerate(faiss_results[:3], 1):
            answer_truncated = result['answer']
            if len(answer_truncated) > 200:
                answer_truncated = answer_truncated[:197] + "..."
            context_parts.append(f"{i}. Q: {result['question']}\n   R: {answer_truncated}")
        
        context = "\n\n".join(context_parts)
        
        # Génération avec Groq
        try:
            if groq_service.available:
                answer = groq_service.generate_response(question, context, history)
                method = "groq_generated"
            else:
                answer = "Je vous recommande de consulter un professionnel de santé pour cette question spécifique. La prévention précoce est essentielle. 💗"
                method = "fallback"
        except Exception as e:
            logger.warning(f"Génération échouée: {str(e)}")
            answer = "Pour des informations précises sur ce sujet, veuillez consulter un médecin ou un centre de santé spécialisé au Bénin. 🌸"
            method = "error_fallback"
        
        return {
            "answer": answer,
            "method": method,
            "score": float(similarity)
        }

# ============================================
# INITIALISATION DES SERVICES (CACHE)
# ============================================

@st.cache_resource
def load_services():
    """Charge les services une seule fois"""
    logger.info("🚀 Chargement des services...")
    groq = GroqService()
    rag = RAGService()
    logger.info("✓ Services chargés")
    return groq, rag

groq_service, rag_service = load_services()

# ============================================
# GESTION DES PARAMÈTRES URL
# ============================================

# Vérifier si c'est un appel API via les query params
query_params = st.query_params

if "api" in query_params and query_params["api"] == "true":
    # MODE API - Pas d'interface, juste réponse JSON
    if "question" in query_params:
        question = query_params["question"]
        user_id = query_params.get("user_id", f"user_{random.randint(1000, 9999)}")
        
        try:
            result = process_question(question, [], groq_service, rag_service)
            
            response_data = {
                "success": True,
                "answer": result["answer"],
                "method": result["method"],
                "similarity_score": result["score"],
                "user_id": user_id
            }
            
            st.json(response_data)
            st.stop()
            
        except Exception as e:
            error_data = {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
            st.json(error_data)
            st.stop()
    else:
        st.json({
            "success": False,
            "error": "Paramètre 'question' manquant"
        })
        st.stop()

# ============================================
# INTERFACE STREAMLIT NORMALE
# ============================================

# CSS personnalisé moderne et joyeux
st.markdown("""
<style>
    /* Animation de pulsation pour le header */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Header principal avec animation */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #ff6b9d 0%, #ffa07a 25%, #ff69b4 50%, #ff1493 75%, #c44569 100%);
        background-size: 200% 200%;
        animation: gradient-shift 8s ease infinite;
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 107, 157, 0.3);
        border: 3px solid rgba(255, 255, 255, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        animation: float 3s ease-in-out infinite;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Cartes statistiques avec hover effect */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        cursor: pointer;
    }
    
    .stat-box:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stat-box h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-box p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Carte API avec style moderne */
    .api-info {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #00acc1;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0, 172, 193, 0.2);
        transition: all 0.3s ease;
    }
    
    .api-info:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0, 172, 193, 0.3);
    }
    
    .api-info h4 {
        color: #00838f;
        margin-top: 0;
    }
    
    .api-code {
        background: linear-gradient(135deg, #2d3436 0%, #000000 100%);
        color: #00ff88;
        padding: 12px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        overflow-x: auto;
        border: 1px solid #00ff88;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
    }
    
    /* Message de bienvenue */
    .welcome-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 20px rgba(253, 203, 110, 0.3);
        border: 2px solid #fdcb6e;
    }
    
    .welcome-box h3 {
        color: #d63031;
        margin-top: 0;
        font-size: 1.5rem;
    }
    
    .welcome-box p {
        color: #2d3436;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Bouton personnalisé */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b9d 0%, #c44569 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.5);
    }
    
    /* Style des messages de chat */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Input de chat stylisé */
    .stChatInputContainer {
        border-radius: 25px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar moderne - Masquée par défaut */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Masquer complètement la sidebar et son bouton */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Expander stylisé */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Animation de chargement */
    .stSpinner > div {
        border-color: #ff6b9d transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Header avec animation
st.markdown("""
<div class="main-header">
    <h1>💗 ANONTCHIGAN 💗</h1>
    <p>✨ Votre Assistante IA Béninoise pour la Sensibilisation au Cancer du Sein 🇧🇯 ✨</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">🌸 Prévention • Information • Accompagnement 🌸</p>
</div>
""", unsafe_allow_html=True)

# Message de bienvenue joyeux
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-box">
        <h3>👋  Bienvenue chez ANONTCHIGAN !</h3>
        <p><strong>🎯 Je suis là pour vous aider !</strong></p>
        <p>💬 Posez-moi toutes vos questions sur le cancer du sein : symptômes, prévention, dépistage, traitement...</p>
        <p>🌟 <strong>Ensemble, prévenons et sensibilisons !</strong> La connaissance sauve des vies. 💗</p>
        <p>🇧🇯 Développée avec amour par le Club IA de l'ENSGMM Abomey</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Tableau de Bord")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>📚 {len(rag_service.questions_data)}</h3>
            <p>Questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        groq_status = "✅ Actif" if groq_service.available else "❌ Inactif"
        groq_color = "#667eea" if groq_service.available else "#e74c3c"
        st.markdown(f"""
        <div class="stat-box" style="background: linear-gradient(135deg, {groq_color} 0%, {groq_color}dd 100%);">
            <h3>{groq_status}</h3>
            <p>🤖 Groq AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Documentation API
    st.markdown("### 🔗 Utiliser l'API")
    
    # Récupérer l'URL de l'app
    try:
        app_url = st.secrets.get("app_url", "https://votre-app.streamlit.app")
    except:
        app_url = "https://votre-app.streamlit.app"
    
    st.markdown(f"""
    <div class="api-info">
        <h4>Méthode GET</h4>
        <p>Envoyez vos questions via URL :</p>
        <div class="api-code">
{app_url}/?api=true&question=Votre+question
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="api-info">
        <h4>📝 Exemple JavaScript</h4>
        <div class="api-code">
const question = "Symptômes cancer sein";<br>
const url = `{URL}/?api=true&question=${encodeURIComponent(question)}`;<br>
<br>
fetch(url)<br>
&nbsp;&nbsp;.then(res => res.json())<br>
&nbsp;&nbsp;.then(data => console.log(data.answer));
        </div>
    </div>
    """.replace("{URL}", app_url), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="api-info">
        <h4>🐍 Exemple Python</h4>
        <div class="api-code">
import requests<br>
import urllib.parse<br>
<br>
question = "Symptômes cancer sein"<br>
url = f"{URL}/?api=true&question={urllib.parse.quote(question)}"<br>
<br>
response = requests.get(url)<br>
data = response.json()<br>
print(data['answer'])
        </div>
    </div>
    """.replace("{URL}", app_url), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👥 Équipe de Développement
    
    🎓 **Club IA - ENSGMM Abomey** 🇧🇯
    
    💻 **Développeurs** :
    - 🌟 Judicaël Karol DOBOEVI
    - 🌟 Ursus Hornel GBAGUIDI
    - 🌟 Abel Kokou KPOCOUTA
    - 🌟 Josaphat ADJELE
    
    💗 *Développé avec passion pour sauver des vies*
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔄 Nouvelle Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"
        st.session_state.conversation_history = []
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); border-radius: 10px; margin-top: 1rem;">
        <p style="margin: 0; color: #2d3436; font-weight: bold;">🌸 La prévention sauve des vies 🌸</p>
        <p style="margin: 0.5rem 0 0 0; color: #636e72; font-size: 0.85rem;">Version 2.3.0</p>
    </div>
    """, unsafe_allow_html=True)

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{random.randint(1000, 9999)}"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if question := st.chat_input("Posez votre question sur le cancer du sein..."):
    # Ajouter la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Traiter la question
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            try:
                result = process_question(
                    question, 
                    st.session_state.conversation_history,
                    groq_service,
                    rag_service
                )
                
                answer = result["answer"]
                method = result["method"]
                score = result["score"]
                
                # Afficher la réponse
                st.markdown(answer)
                
                # Ajouter à l'historique
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Mettre à jour l'historique de conversation
                st.session_state.conversation_history.append({"role": "user", "content": question})
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                
                # Limiter l'historique
                if len(st.session_state.conversation_history) > Config.MAX_HISTORY_LENGTH * 2:
                    st.session_state.conversation_history = st.session_state.conversation_history[-Config.MAX_HISTORY_LENGTH * 2:]
                
            except Exception as e:
                error_message = f"❌ Erreur: {str(e)}"
                st.error(error_message)
                logger.error(error_message)

# Footer minimaliste
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p style="margin: 0; font-size: 0.9rem;">ANONTCHIGAN v2.3.0 • 2025</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">Développé par le Club IA de l'ENSGMM Abomey 🇧🇯</p>
</div>
""", unsafe_allow_html=True)

