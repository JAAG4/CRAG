import os
import json
import openai
from tracing_phx import tracer
from tenacity import retry, wait_fixed
OPENAI_MODEL = "llama-3.1-8b-instant"

openai_key = os.getenv("OPENAI_API_KEY")


@tracer.chain
@retry(wait=wait_fixed(2))
def expand_query2doc(
    original_query: str,
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
        "If a query is written as a claim or statement, rephrase it as a web-searchable query."
        "then Write a highly descriptive paragraph that answers the query directly. "
        "Do not include conversational filler, just the factual response."
    )

    with tracer.start_as_current_span(
        "expand_query2doc", openinference_span_kind="chain"
    ) as span:
        span.set_input(
            {"original_query": original_query, "retrieval_type": retrieval_type}
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
            span.set_attributes({"error": str(e)})
            span.set_output(
                {"result": original_query, "fallback": True, "reason": "openai_error"}
            )
            return original_query

        # Reformulación matemática basada en el tipo de recuperación
        if retrieval_type.lower() == "sparse":
            # Repetición del query original 5 veces para ponderación de BM25
            repeated_query = " ".join([original_query] * 5)
            expanded_query = f"{repeated_query} {pseudo_document}"
        else:
            # Concatenación simple para Bi-Encoders densos
            expanded_query = f"{original_query} [SEP] {pseudo_document}"

        span.set_output(
            {
                "pseudo_document": pseudo_document,
                "expanded_query": expanded_query,
            }
        )
        return expanded_query


@tracer.chain
@retry(wait=wait_fixed(2))
def is_class_ii_query(query: str) -> bool:
    """
    Clasifica si un query es de Clase II (requiere descomposición por pedir
    múltiples fuentes, comparaciones o agregaciones).
    """
    SYSTEM_PROMPT = """You are a query classifier for a RAG system. Determine if the user's query requires aggregating or comparing information from multiple independent sources or entities.
Examples: 'Compare the GDP of Japan and Germany', 'What are the differences between Python and Rust?'. 
Examples of Simple: 'What is the capital of France?', 'How does backpropagation work?'. 
Output EXACTLY and ONLY the word TRUE if it is Class II, or FALSE if it is not."""

    with tracer.start_as_current_span(
        "is_class_ii_query", openinference_span_kind="chain"
    ) as span:
        span.set_input({"query": query})

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
            is_class_ii = "TRUE" in result
            span.set_output({"result": is_class_ii, "raw_model_response": result})
            return is_class_ii
        except Exception as e:
            print(f"Error en clasificación: {e}")
            span.set_attributes({"error": str(e)})
            span.set_output({"result": False, "fallback": True})
            return False


@tracer.chain
@retry(wait=wait_fixed(2))
def decompose_query(query: str) -> list:
    """
    Descompone un query complejo en una lista de sub-queries atómicos.
    Retorna una lista de strings.
    """
    SYSTEM_PROMPT = (
        "You are an expert query planner. Your task is to break down a complex query "
        "into simpler, atomic sub-queries that can be searched independently. "
        "If a query is written as a claim or statement, rephrase it as a web-searchable query."
        "Output ONLY a valid JSON array of strings containing the sub-queries, with no markdown formatting or extra text. "
        'Example: ["What is the GDP of Japan over the last decade?", "What is the GDP of Germany over the last decade?"]'
    )

    with tracer.start_as_current_span(
        "decompose_query", openinference_span_kind="chain"
    ) as span:
        span.set_input({"query": query})

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
                span.set_output({"result": parsed_data})
                return parsed_data
            elif isinstance(parsed_data, dict):
                for val in parsed_data.values():
                    if isinstance(val, list):
                        span.set_output({"result": val})
                        return val

            span.set_output(
                {"result": [query], "fallback": True, "reason": "parsed_not_list"}
            )
            return [query]  # Fallback
        except Exception as e:
            print(f"Error en descomposición: {e}")
            span.set_attributes({"error": str(e)})
            span.set_output({"result": [query], "fallback": True})
            return [
                query
            ]  # En caso de error, devolvemos el query original en una lista


@retry(wait=wait_fixed(2))
def optimize_pipeline(original_query: str, retrieval_type: str = "dense") -> list:
    """
    Pipeline principal: Clasifica, descompone (si es necesario) y expande.
    Retorna siempre una lista de queries listos para el motor de búsqueda.
    """
    final_queries = []

    with tracer.start_as_current_span(
        "optimize_pipeline", openinference_span_kind="chain"
    ) as span:
        span.set_input(
            {"original_query": original_query, "retrieval_type": retrieval_type}
        )

        if is_class_ii_query(original_query):
            print("-> Query clasificado como Clase II. Iniciando descomposición...")
            span.set_attributes({"classification": "Class II"})

            with tracer.start_as_current_span(
                "optimize_pipeline.decompose", openinference_span_kind="chain"
            ) as decomp_span:
                decomp_span.set_input({"query": original_query})
                sub_queries = decompose_query(original_query)
                decomp_span.set_output({"sub_queries": sub_queries})

            print(f"->{len(sub_queries)} Sub-queries generados: {sub_queries}")

            with tracer.start_as_current_span(
                "optimize_pipeline.expand_subqueries", openinference_span_kind="chain"
            ) as expansion_span:
                expansion_span.set_input(
                    {
                        "sub_query_count": len(sub_queries),
                        "retrieval_type": retrieval_type,
                    }
                )
                for sq in sub_queries:
                    expanded = expand_query2doc(sq, retrieval_type)
                    final_queries.append(expanded)
                expansion_span.set_output(
                    {"expanded_queries_count": len(final_queries)}
                )
        else:
            print("-> Query simple. Pasando directo a expansión...")
            span.set_attributes({"classification": "simple"})
            with tracer.start_as_current_span(
                "optimize_pipeline.expand_single", openinference_span_kind="chain"
            ) as expansion_span:
                expansion_span.set_input(
                    {"query": original_query, "retrieval_type": retrieval_type}
                )
                expanded = expand_query2doc(original_query, retrieval_type)
                final_queries.append(expanded)
                expansion_span.set_output({"expanded_query": expanded})

        span.set_output({"optimized_queries": final_queries})

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
