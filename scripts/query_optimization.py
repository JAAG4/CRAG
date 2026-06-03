import os
import json
import openai

OPENAI_MODEL = "llama-3.1-8b-instant"

openai_key = os.getenv("OPENAI_API_KEY")


def expand_query2doc(
    original_query: str,
    client,
    retrieval_type: str = "dense",
) -> str:
    """
    Expande una consulta de búsqueda utilizando el paradigma Query2Doc.

    Args:
        original_query (str): La consulta original del usuario.
        retrieval_type (str): El tipo de sistema de recuperación ("dense" o "sparse").
        api_key (str): Tu API key de Groq (por defecto busca la variable de entorno GROQ_API_KEY).

    Returns:
        str: El query expandido listo para ser inyectado en tu sistema de recuperación.
    """
    # Inicializamos el cliente de OpenAI apuntando a la API de Groq

    # Instrucción Zero-Shot (Q2D/ZS)
    SYSTEM_PROMPT = (
        "You are an expert knowledge base. "
        "Write a highly descriptive paragraph that answers the query directly. "
        "Do not include conversational filler, just the factual response."
    )

    try:
        # Generación del pseudo-documento utilizando Groq
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": original_query},
            ],
            temperature=0.0,  # Temperatura baja para reducir alucinaciones
            max_tokens=150,
        )

        pseudo_document = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generando el pseudo-documento: {e}")
        # Como fallback, retornamos el query original
        return original_query

    # Reformulación matemática basada en el tipo de recuperación
    if retrieval_type.lower() == "sparse":
        # Repetición del query original 5 veces para ponderación de BM25
        repeated_query = " ".join([original_query] * 5)
        expanded_query = f"{repeated_query} {pseudo_document}"
    else:
        # Concatenación simple para Bi-Encoders densos
        expanded_query = f"{original_query} [SEP] {pseudo_document}"

    return expanded_query


def is_class_ii_query(query: str, client) -> bool:
    """
    Clasifica si un query es de Clase II (requiere descomposición por pedir
    múltiples fuentes, comparaciones o agregaciones).
    """
    SYSTEM_PROMPT = """You are a query classifier for a RAG system. Determine if the user's query requires aggregating or comparing information from multiple independent sources or entities.
Examples: 'Compare the GDP of Japan and Germany', 'What are the differences between Python and Rust?'. 
Examples of Simple: 'What is the capital of France?', 'How does backpropagation work?'. 
Output EXACTLY and ONLY the word TRUE if it is Class II, or FALSE if it is not."""

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
        return "TRUE" in result
    except Exception as e:
        print(f"Error en clasificación: {e}")
        return False


def decompose_query(query: str, client) -> list:
    """
    Descompone un query complejo en una lista de sub-queries atómicos.
    Retorna una lista de strings.
    """
    SYSTEM_PROMPT = (
        "You are an expert query planner. Your task is to break down a complex query "
        "into simpler, atomic sub-queries that can be searched independently. "
        "Output ONLY a valid JSON array of strings containing the sub-queries, with no markdown formatting or extra text. "
        'Example: ["What is the GDP of Japan over the last decade?", "What is the GDP of Germany over the last decade?"]'
    )

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            # Forzamos salida JSON si el modelo lo soporta, o confiamos en el prompt estricto
            response_format={"type": "json_object"},
        )

        # Parsear el string JSON retornado a una lista de Python
        raw_json = response.choices[0].message.content.strip()
        # Si el modelo devuelve un diccionario con una llave, lo extraemos.
        # Adaptado para asegurar robustez en el parseo:
        parsed_data = json.loads(raw_json)

        if isinstance(parsed_data, list):
            return parsed_data
        elif isinstance(parsed_data, dict):
            # Extrae la primera lista que encuentre en los valores del dict
            for val in parsed_data.values():
                if isinstance(val, list):
                    return val

        return [query]  # Fallback
    except Exception as e:
        print(f"Error en descomposición: {e}")
        return [query]  # En caso de error, devolvemos el query original en una lista


def optimize_pipeline(
    original_query: str, client, retrieval_type: str = "dense"
) -> list:
    """
    Pipeline principal: Clasifica, descompone (si es necesario) y expande.
    Retorna siempre una lista de queries listos para el motor de búsqueda.
    """
    final_queries = []

    # 1. Clasificación
    if is_class_ii_query(original_query, client):
        print("-> Query clasificado como Clase II. Iniciando descomposición...")
        # 2. Descomposición
        sub_queries = decompose_query(original_query, client)
        print(f"->{len(sub_queries)} Sub-queries generados: {sub_queries}")

        # 3. Expansión para cada sub-query
        for sq in sub_queries:
            expanded = expand_query2doc(sq, client, retrieval_type)
            final_queries.append(expanded)
    else:
        print("-> Query simple. Pasando directo a expansión...")
        # 3. Expansión del query original
        expanded = expand_query2doc(original_query, client, retrieval_type)
        final_queries.append(expanded)

    return final_queries


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Asegúrate de configurar: os.environ["GROQ_API_KEY"] = "tu_clave_aqui"

    testquery = "What is the capital of France?"

    # Para motores basados en vectores / embeddings (Dense)
    # dense_query = expand_query2doc(query, client, retrieval_type="dense")
    # print(f"Dense Expansion:\n{dense_query}\n")

    # Para motores léxicos como  BM25 (Sparse)
    # sparse_query = expand_query2doc(query, client, retrieval_type="sparse")
    # print(f"Sparse Expansion:\n{sparse_query}")
