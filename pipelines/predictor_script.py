import torch
import pandas as pd
import logging

import hopsworks

from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


class Predict(object):
    """
    A predictor class for Kserver that handles retrieval augmented generation for
    Rättsmedicinalverket's documentation.
    """

    # Configuration constants
    DOWNLOAD_PATH = "data"
    RERANKER = 'BAAI/bge-reranker-large'
    GPT_MODEL = "gpt-4o-mini-2024-07-18"
    MODEL_SENTENCE_TRANSFORMER = 'all-MiniLM-L6-v2'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        """Initialize the predictor with necessary models and connections."""
        self.setup_feature_store()
        self.setup_models()
        # Configure the logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # Create a logger instance
        self.logger = logging.getLogger()

    def setup_feature_store(self):
        """Connect to Hopsworks and set up the feature store."""
        self.project = hopsworks.login()
        self.fs = self.project.get_feature_store()
        self.feedback_fg = self.fs.get_feature_group(
            name="user_feedback",
            version=1
        )
        self.feature_view = self.fs.get_feature_view(
            name="administrative_protocols",
            version=1
        )

    def setup_models(self):
        """Initialize the necessary ML models."""
        # Initialize reranker
        self.reranker = self.get_reranker(self.RERANKER)

        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer(
            self.MODEL_SENTENCE_TRANSFORMER
        ).to(self.DEVICE)

    def get_reranker(self, reranker_model: str) -> FlagReranker:
        """
        Initialize and return a FlagReranker model.

        Args:
            reranker_model: The name/path of the reranker model.

        Returns:
            An initialized FlagReranker instance.
        """
        return FlagReranker(
            reranker_model,
            use_fp16=True,
        )

    def get_source(self, neighbors: List[Tuple[str, str, int, int]]) -> str:
        """
        Generates a formatted string for the sources of the provided context.

        Args:
            neighbors: List of tuples representing document information.

        Returns:
            Formatted string containing document names, links, pages, and paragraphs.
        """
        return '\n\nReferences:\n' + '\n'.join(
            [
                f' - {neighbor[0]}({neighbor[1]}): Page: {neighbor[3]}, Paragraph: {neighbor[4]}'
                for neighbor in neighbors
            ]
        )

    def get_context(self, neighbors: List[Tuple[str]]) -> str:
        """
        Generates a formatted string for the context based on the provided neighbors.

        Args:
            neighbors: List of tuples representing context information.

        Returns:
            Formatted string containing context information.
        """
        return '\n\n'.join([neighbor[-1] for neighbor in neighbors])

    def get_neighbors(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the k closest neighbors for a given query using sentence embeddings.

        Args:
            query: The input query string.
            k: Number of neighbors to retrieve. Default is 10.

        Returns:
            A list of tuples containing the neighbor context.
        """
        question_embedding = self.sentence_transformer.encode(query)

        # Retrieve closest neighbors
        neighbors = self.feature_view.find_neighbors(
            question_embedding,
            k=k,
        )

        return neighbors

    def rerank(self, query: str, neighbors: List[str], k: int = 3) -> List[str]:
        """
        Rerank a list of neighbors based on a reranking model.

        Args:
            query: The input query string.
            neighbors: List of neighbor contexts.
            k: Number of top-ranked neighbors to return. Default is 3.

        Returns:
            The top-ranked neighbor contexts after reranking.
        """
        # Compute scores for each context using the reranker
        scores = [self.reranker.compute_score([query, context[5]]) for context in neighbors]

        combined_data = [*zip(scores, neighbors)]

        # Sort contexts based on the scores in descending order
        sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)

        # Return the top-k ranked contexts
        return [context for score, context in sorted_data][:k]

    def get_context_and_source(self, user_query: str, year: Optional[int] = None, k: int = 10) -> Tuple[str, str]:
        """
        Retrieve context and source based on user query using embedding, feature view, and reranking.

        Args:
            user_query: The user's input query string.
            year: Filter to select only findings of this particular year.
            k: Number of nearest neighbors to find

        Returns:
            A tuple containing the retrieved context and source.
        """
        # Retrieve closest neighbors
        neighbors = self.get_neighbors(
            user_query,
            k=k,
        )

        # Rerank the neighbors to get top-k
        context_reranked = self.rerank(
            user_query,
            neighbors,
            k=3,
        )

        # Retrieve source
        source = self.get_source(context_reranked)

        return context_reranked, source

    def build_prompt(self, query, context):
        """
        Build a multi-shot prompt for LLM to provide detailed and relevant answers
        about medical and administrative procedures at Rättsmedicinalverket's
        forensic psychiatric investigation units.

        Args:
            query: The user's query
            context: Retrieved context information

        Returns:
            A formatted prompt string
        """
        # Multi-shot examples
        example_q1 = "Vilka är de första åtgärderna vid en anafylaktisk reaktion?"
        example_a1 = (
            "Enligt Rättsmedicinalverkets rutin för anafylaktisk reaktion (Dokument-ID: RMV0-629561176-1083, "
            "sida 2) är de första åtgärderna vid anafylaxi: 1) Lägg patienten ned och höj fotändan för att "
            "förebygga blodtrycksfall, 2) Administrera Adrenalin (Epipen 300 mikrog) intramuskulärt på lårets "
            "utsida, 3) Ge syrgas minst 5 l/min, 4) Etablera intravenös infart och ge Ringer Acetat."
        )

        example_q2 = "Hur ska läkemedelsbiverkningar rapporteras?"
        example_a2 = (
            "Enligt rutinen för anmälan av läkemedelsbiverkan (Dokument-ID: RMV0-815988144-564, "
            "sida 1-2) ska misstänkta läkemedelsbiverkningar snarast rapporteras till Läkemedelsverket. "
            "Nyupptäckt läkemedelsbiverkan rapporteras av dagansvarig specialistläkare, och biverkningar "
            "ska dokumenteras i patientjournalen. Alla allvarliga biverkningar samt biverkningar av nya "
            "läkemedel (<2 år) ska rapporteras, medan banala biverkningar av äldre läkemedel inte behöver "
            "rapporteras."
        )

        example_q3 = "Vilka krav gäller för att en intagen ska få tillstånd till telefonsamtal?"
        example_a3 = (
            "Enligt rutinen för ansökan om besök och tillstånd att telefonera (Dokument-ID: RMV0-815988144-687, "
            "sida 2-3) måste en intagen som önskar få tillstånd att telefonera till en viss person: 1) Ansöka på "
            "en särskild blankett från RMV, 2) Hämta in samtycke från den som samtalet avser, 3) Invänta beslut "
            "från RMV som gör en egen prövning. Undantag från samtyckeskravet gäller för offentlig försvarare, "
            "vissa myndighetspersoner, och vid telefonsamtal av stor vikt för den intagnes frigivningsförberedelser "
            "eller sociala situation."
        )

        # Build references to context sections
        snippet_texts = []
        for c in context:
            ref_str = ""
            # Identify the source in a more explicit, structured format
            ref_str = f"[källa: Rättsmedicinalverkets rutin, Dokument-ID: {c[1]}, sida {c[3]}, stycke {c[4]}]"

            snippet_texts.append(f"Kontextavsnitt:\n{ref_str}\n{c[5]}\n")

        # Combine all section texts
        context_snippets = "\n".join(snippet_texts)

        # Construct the enhanced prompt
        prompt = f"""
Du är en kunnig assistent som refererar till Rättsmedicinalverkets rutiner och riktlinjer för att hjälpa 
personal vid rättspsykiatriska undersökningsenheter med medicinska och administrativa frågor.

Ditt mål är att ge ett koncist men specifikt svar på användarens fråga, med fokus på:
- Referenser till relevanta dokument-ID, sidor eller avsnitt från Rättsmedicinalverkets rutiner som nämns i avsnitten.
- Detaljerad information om korrekta medicinska och administrativa procedurer enligt rutinerna.

Nedan följer flera exempel på frågor och svar som visar referensstilen:

EXEMPEL F1:
{example_q1}

EXEMPEL S1:
{example_a1}

EXEMPEL F2:
{example_q2}

EXEMPEL S2:
{example_a2}

EXEMPEL F3:
{example_q3}

EXEMPEL S3:
{example_a3}

Här är kontextavsnitt från din kunskapsbas (Rättsmedicinalverkets rutiner):

{context_snippets}

ANVÄNDARENS FRÅGA: {query}

Följ dessa steg:
1. Identifiera vilket/vilka avsnitt som kan vara relevanta för frågan.
2. Sammanfatta nyckelinformation från dessa avsnitt som besvarar frågan.
3. Ge ett koncist, direkt svar med referenser till specifika dokument-ID och sidor om avsnitten nämner dem.
4. Om avsnitten endast delvis besvarar frågan, ge en sammanfattning baserad på tillgänglig information och notera eventuella kunskapsluckor.
5. Om absolut inget avsnitt behandlar användarens fråga kan du svara: "Jag kan inte besvara frågan baserat på tillgänglig information."

Kom ihåg: Använd den tillhandahållna kontexten när det är möjligt.
Om frågan går utöver de tillhandahållna avsnitten, påpeka att
kontexten inte fullt ut besvarar frågan.

VIKTIGT: Ditt svar måste vara på svenska och anpassat för medicinsk och administrativ personal vid Rättsmedicinalverkets rättspsykiatriska undersökningsenheter.
"""
        return prompt

    def get_answer_from_gemini(self, query: str, context: str, source: str) -> str:
        """
        Get an answer from Google Gemini API.

        Args:
            query: The user query
            context: Retrieved context
            source: Source information

        Returns:
            Response from Gemini
        """
        prompt_text = self.build_prompt(query, context)
        gemini_key = os.environ.get("GEMINI_KEY")

        if not gemini_key:
            raise ValueError("GEMINI_KEY environment variable is not set")

        # Gemini Endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent?key={gemini_key}"

        # Gemini payload structure
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0  # You can set this value between 0.0 and 1.0
            }
        }

        # Set headers
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Make the POST request
            import requests
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            # Parse the JSON response
            response_data = response.json()

            try:
                return response_data['candidates'][0]["content"]["parts"][0]["text"] + "\n\n" + source
            except (KeyError, IndexError):
                # If the structure is different, you might need to dig deeper or return the full object.
                return str(response_data)

        except requests.RequestException as e:
            # Handle network or HTTP errors
            print(f"Gemini API request failed: {e}")
            return None

    def store_feedback(self, feedback_data):
        """
        Store user feedback in the feature store.

        Args:
            feedback_data: User feedback data

        Returns:
            Status of feedback storage operation
        """
        try:
            # Extract user feedback
            user_feedback = feedback_data["user_feedback"]
            self.logger.info(f"user_feedback is {user_feedback}.")

            # Create a DataFrame with the feedback data
            feedback_df = pd.DataFrame({
                'feedback_id': [user_feedback["feedback_id"]],
                'user_query': [user_feedback["user_query"]],
                'assistant_response': [user_feedback["assistant_response"]],
                'like': [user_feedback["like"]],
                'feedback': [user_feedback["feedback"]],
                'session_id': [user_feedback.get("session_id", "unknown")],
                'timestamp': [datetime.fromisoformat(user_feedback["timestamp"])
                              if "timestamp" in user_feedback
                              else datetime.now()]
            })

            # Insert the data into the feature group
            self.feedback_fg.multi_part_insert(feedback_df)

            self.logger.info(f"feedback_fg.insert")

        except Exception as e:
            self.logger.info(f"Error while trying to store feedback: {e}")

    def predict(self, inputs):
        """
        Main prediction function that handles the entire RAG pipeline.

        Args:
            query: The user's query string
            model: Which LLM to use ("gpt" or "gemini")

        Returns:
            A dictionary containing the response and metadata
            :param inputs:
        """
        # Check if this is a feedback submission
        if "user_feedback" in inputs[0]:
            self.logger.info(f"user_feedback in predict is {inputs[0]}.")
            return self.store_feedback(inputs[0])

        query = inputs[0]["user_query"]
        # Retrieve context and source
        context, source = self.get_context_and_source(query)

        # Get response from selected model
        response = self.get_answer_from_gemini(query, context, source)

        return response
