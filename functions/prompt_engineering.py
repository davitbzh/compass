import os
import requests
import json_repair

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from hsfs import feature_view


def get_reranker(reranker_model: str) -> FlagReranker:
    reranker = FlagReranker(
        reranker_model,
        use_fp16=True,
    )
    return reranker


def get_source(neighbors: List[Tuple[str, str, int, int]]) -> str:
    """
    Generates a formatted string for the sources of the provided context.

    Args:
        neighbors (List[Tuple[str, str, int, int]]): List of tuples representing document information.

    Returns:
        str: Formatted string containing document names, links, pages, and paragraphs.
    """
    return '\n\nReferences:\n' + '\n'.join(
        [
            f' - {neighbor[0]}({neighbor[1]}): Page: {neighbor[2]}, Paragraph: {neighbor[3]}'
            for neighbor
            in neighbors
        ]
    )


def get_context(neighbors: List[Tuple[str]]) -> str:
    """
    Generates a formatted string for the context based on the provided neighbors.

    Args:
        neighbors (List[Tuple[str]]): List of tuples representing context information.

    Returns:
        str: Formatted string containing context information.
    """
    return '\n\n'.join([neighbor[-1] for neighbor in neighbors])


def get_neighbors(query: str, sentence_transformer: SentenceTransformer, feature_view, k: int = 10) -> List[
    Tuple[str, float]]:
    """
    Get the k closest neighbors for a given query using sentence embeddings.

    Parameters:
    - query (str): The input query string.
    - sentence_transformer (SentenceTransformer): The sentence transformer model.
    - feature_view (FeatureView): The feature view for retrieving neighbors.
    - k (int, optional): Number of neighbors to retrieve. Default is 10.

    Returns:
    - List[Tuple[str, float]]: A list of tuples containing the neighbor context.
    """
    question_embedding = sentence_transformer.encode(query)

    # Retrieve closest neighbors
    neighbors = feature_view.find_neighbors(
        question_embedding,
        k=k,
    )

    return neighbors


def rerank(query: str, neighbors: List[str], reranker, k: int = 3) -> List[str]:
    """
    Rerank a list of neighbors based on a reranking model.

    Parameters:
    - query (str): The input query string.
    - neighbors (List[str]): List of neighbor contexts.
    - reranker (Reranker): The reranking model.
    - k (int, optional): Number of top-ranked neighbors to return. Default is 3.

    Returns:
    - List[str]: The top-ranked neighbor contexts after reranking.
    """
    # Compute scores for each context using the reranker
    scores = [reranker.compute_score([query, context[5]]) for context in neighbors]

    combined_data = [*zip(scores, neighbors)]

    # Sort contexts based on the scores in descending order
    sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)

    # Return the top-k ranked contexts
    return [context for score, context in sorted_data][:k]


def get_context_and_source(user_query: str, sentence_transformer: SentenceTransformer,
                           feature_view: feature_view.FeatureView, reranker: FlagReranker, year: int = None,
                           k: int = 10) -> Tuple[str, str]:
    """
    Retrieve context and source based on user query using a combination of embedding, feature view, and reranking.

    Parameters:
    - user_query (str): The user's input query string.
    - sentence_transformer (SentenceTransformer): The sentence transformer model.
    - feature_view (FeatureView): The feature view for retrieving neighbors.
    - reranker (Reranker): The reranking model.
    - year: filter to select only findigs of this particular year.
    - k: number of nearest neighbors to find

    Returns:
    - Tuple[str, str]: A tuple containing the retrieved context and source.
    """
    # Retrieve closest neighbors
    neighbors = get_neighbors(
        user_query,
        sentence_transformer,
        feature_view,
        k=k,
    )
    
    print(neighbors)

    # Rerank the neighbors to get top-k
    context_reranked = rerank(
        user_query,
        neighbors,
        reranker,
        k=3,
    )

    # Retrieve source
    source = get_source(context_reranked)
    
    print(source)

    return context_reranked, source


def build_prompt(query, context):
    """
    Skapa en multi-shot prompt för LLM för att ge detaljerade och relevanta svar
    om medicinska och administrativa rutiner vid Rättsmedicinalverkets
    rättspsykiatriska undersökningsenheter.
    """
    # -- MULTI-SHOT EXEMPEL --
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

    # Bygg referenser till kontextavsnitt
    snippet_texts = []
    for c in context:
        ref_str = ""
        # Identifiera källan i ett mer explicit, strukturerat format
        ref_str = f"[källa: Rättsmedicinalverkets rutin, Dokument-ID: {c[1]}, sida {c[3]}, stycke {c[4]}]"

        snippet_texts.append(f"Kontextavsnitt:\n{ref_str}\n{c[5]}\n")

    # Kombinera alla avsnittstexter
    context_snippets = "\n".join(snippet_texts)

    # Konstruera den förbättrade prompten
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

Kom ihåg: Använd den tillhandahållna kontexten när det är möjligt.
Om frågan går utöver de tillhandahållna avsnitten, påpeka att
kontexten inte fullt ut besvarar frågan.

VIKTIGT: Ditt svar måste vara på svenska och anpassat för medicinsk och administrativ personal vid Rättsmedicinalverkets rättspsykiatriska undersökningsenheter.
"""
    return prompt

def get_answer_from_gemini(query: str, context: str, source: str, api_key: str):
    """
    Calls the Google Gemini REST API with the given prompt_text.
    Returns the response as a string (if parsing is successful) or a raw JSON fallback.
    
    Example usage:
        response = call_gemini_api("Explain how AI works")
        print("Gemini response:", response)
    """
    prompt_text = build_prompt(query, context)

    # Gemini Endpoint 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent?key={api_key}"

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
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        # Parse the JSON response
        response_data = response.json()

        try:
            return response_data['candidates'][0]["content"]["parts"][0]["text"] + "\n\n" + source
        except (KeyError, IndexError):
            # If the structure is different, you might need to dig deeper or return the full object.
            return response_data

    except requests.RequestException as e:
        # Handle network or HTTP errors
        print(f"Gemini API request failed: {e}")
        return None


# Function to query the OpenAI API
def get_answer_from_gpt(query: str, context: str, source: str, gpt_model: str, client) -> str:
    # Build the prompt
    prompt = build_prompt(query, context)

    # Create a chatbot
    completion = client.chat.completions.create(
        model=gpt_model,
        # Pre-define conversation messages for the possible roles 
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    response = json_repair.loads(completion.choices[0].message.content) + "\n\n" + str(source)

    return response
